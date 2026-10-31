"""Self-contained zarr -> mp4 converter for one Squid recording store.

Depends only on numpy, tensorstore and imageio-ffmpeg, nothing from the viewer,
so Squid can import this module (or copy this one file) and run the same
conversion as soon as a recording store finalizes. Both sides then write the
same files to the same place: `~/Downloads/mp4-<experiment folder name>/`.

A recording store is `recording/t{t}/{well}/fov_{k}[_z{j}].ome.zarr`, a Zarr v3
array of shape (T, 1, 1, Y, X) at the store root with a `_squid` attribute
block. The store is opened read-only and the mp4 keeps as much of it as the
format allows: full pixel resolution, every captured frame, the real-time
playback rate, a near-lossless H.264 encode. What mp4 cannot hold is 16-bit
data: player-compatible H.264 is 8-bit, so every frame goes through ONE
linear window per video, the video's own min and max (clips nothing). The
window is returned and written into the mp4's `comment` metadata, so the
mapping can be read back from the file alone. The 8-bit value then picks a
colour on the black-to-channel-colour ramp the viewer uses, the channel colour
being the store's own `omero.channels[0].color`; a white channel stays gray.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import imageio_ffmpeg
import numpy as np
import tensorstore as ts

CRF = 12                 # near-lossless; still plays everywhere
PRESET = "veryfast"      # 2048x2048 bench: 3.6x faster than medium at the same CRF and PSNR, smaller file
CANCELLED = "cancelled"  # ConvertResult.error of a job stopped by the caller
_FALLBACK_FPS = 10.0     # when the store carries no timing at all


@dataclass
class ConvertResult:
    store: Path
    out: Path
    frames: int = 0                            # frames written to the mp4
    window: tuple[float, float] | None = None  # the 16-to-8 window used
    color: str | None = None                   # `#RRGGBB` ramp applied after the window; None = gray
    error: str | None = None                   # set when no file was written

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        """One log line: `A1_fov_0.mp4: 30 frames, window 4–65534`."""
        if self.error == CANCELLED:
            return f"{self.out.name}: cancelled"
        if self.error is not None:
            return f"{self.out.name}: FAILED: {self.error}"
        lo, hi = self.window
        color = f", color {self.color}" if self.color else ""
        return f"{self.out.name}: {self.frames} frames, window {lo:g}–{hi:g}{color}"


# ----------------------------------------------------------------- naming

def default_out_dir(experiment_dir: str | Path) -> Path:
    """`~/Downloads/mp4-<experiment folder name>/`, the one place both the viewer
    and Squid write to; outside the data folder, one folder per run."""
    return Path.home() / "Downloads" / f"mp4-{Path(experiment_dir).name}"


def video_name(well: str, fov: int, acq_t: int, z: int, n_t: int, n_z: int) -> str:
    """The store's own name, prefixed by its well: `recording/t0/A1/fov_0.ome.zarr`
    becomes `A1_fov_0.mp4`, a plane store `fov_0_z1.ome.zarr` becomes
    `A1_fov_0_z1.mp4`. `_t{t}` is inserted after the well only when the run has
    several timepoints."""
    name = f"{well}_t{acq_t}" if n_t > 1 else well
    name += f"_fov_{fov}"
    if n_z > 1:
        name += f"_z{z}"
    return name + ".mp4"


# ------------------------------------------------------------ store facts

def squid_attrs(store: Path) -> dict:
    """The `_squid` block of the store's `zarr.json` (empty when absent)."""
    meta = json.loads((Path(store) / "zarr.json").read_text())
    return (meta.get("attributes") or {}).get("_squid") or {}


def frames_written(store: Path, declared: int) -> int:
    """Frames present: `_squid.captured_frames`, else the `c/` chunk-dir count
    (one chunk dir per frame under the default full-frame chunking), else the
    declared T when the store is sharded (presence is not readable per frame)."""
    attrs = squid_attrs(store)
    if attrs.get("captured_frames") is not None:
        return int(attrs["captured_frames"])
    if str(attrs.get("compression", "")).upper() in ("BALANCED", "BEST"):
        return declared
    try:
        with os.scandir(Path(store) / "c") as it:
            return sum(1 for e in it if e.is_dir())
    except FileNotFoundError:
        return 0


def default_fps(store: Path) -> float:
    """Real-time playback: the rate squid recorded at, else 1 / frame interval."""
    attrs = squid_attrs(store)
    if attrs.get("effective_fps"):
        return float(attrs["effective_fps"])
    dt = float(attrs.get("time_increment_s") or 0.0)
    return round(1.0 / dt, 3) if dt > 0 else _FALLBACK_FPS


def parse_color(raw: object) -> str | None:
    """`#RRGGBB` from a hex colour with or without '#'; None when unparsable or
    white, meaning plain gray (a white ramp IS gray, and that is what the viewer
    shows for a white channel)."""
    s = str(raw or "").lstrip("#").upper()
    if len(s) != 6 or any(c not in "0123456789ABCDEF" for c in s) or s == "FFFFFF":
        return None
    return f"#{s}"


def channel_color(store: Path) -> str | None:
    """The store's own channel colour (OME `omero.channels[0].color`, which Squid
    writes without the '#'), via `parse_color`."""
    meta = json.loads((Path(store) / "zarr.json").read_text())
    channels = ((((meta.get("attributes") or {}).get("ome") or {}).get("omero") or {})
                .get("channels") or [])
    return parse_color((channels[0] if channels else {}).get("color"))


def color_ramp(color: str) -> np.ndarray:
    """256 x 3 uint8 lookup, black -> `color`: the viewer's channel colormap.
    Indexed by the 8-bit value AFTER the 16-to-8 window, so contrast is untouched."""
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    return np.rint(np.arange(256, dtype=np.float32)[:, None] * rgb / 255.0).astype(np.uint8)


def open_frames(store: Path) -> ts.TensorStore:
    """The store's array, read-only, lazily."""
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(store)},
                    "open": True, "create": False}).result()


class UnsupportedStore(ValueError):
    """The store is not a recording this converter knows how to read faithfully."""


def check_store(store: Path) -> dict:
    """Refuse anything but the recording layout this converter was written for,
    so a format change fails loudly instead of converting the wrong thing.

    Expects a Zarr v3 ARRAY node at the store root, five dimensions (T, C, Z, Y,
    X) with C and Z of size 1, an integer dtype, and `_squid.structure ==
    "5D-TCZYX"` when `_squid` is present. Returns the parsed `zarr.json`.
    """
    store = Path(store)
    try:
        meta = json.loads((store / "zarr.json").read_text())
    except FileNotFoundError:
        raise UnsupportedStore(f"{store.name}: no zarr.json (not a zarr store)") from None
    problems = []
    if meta.get("zarr_format") != 3:
        problems.append(f"zarr_format {meta.get('zarr_format')!r}, expected 3")
    if meta.get("node_type") != "array":
        problems.append(f"node_type {meta.get('node_type')!r}, expected 'array' at the store root")
    shape = meta.get("shape") or []
    if len(shape) != 5 or shape[1] != 1 or shape[2] != 1:
        problems.append(f"shape {shape}, expected (T, 1, 1, Y, X)")
    if not str(meta.get("data_type", "")).startswith(("uint", "int")):
        problems.append(f"data_type {meta.get('data_type')!r}, expected an integer type")
    structure = ((meta.get("attributes") or {}).get("_squid") or {}).get("structure")
    if structure not in (None, "5D-TCZYX"):
        problems.append(f"_squid.structure {structure!r}, expected '5D-TCZYX'")
    if problems:
        raise UnsupportedStore(f"{store.name}: store format not supported: " + "; ".join(problems))
    return meta


# -------------------------------------------------------------- conversion

def data_window(arr: ts.TensorStore, n_frames: int) -> tuple[float, float]:
    """The video's own `(min, max)` over every frame, one read pass."""
    lo, hi = np.inf, -np.inf
    for f in range(n_frames):
        plane = arr[f, 0, 0].read().result()
        lo, hi = min(lo, float(plane.min())), max(hi, float(plane.max()))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _window(window: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(window[0]), float(window[1])
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _to_uint8(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (plane.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def _ffmpeg_command(width: int, height: int, fps: float, out: Path,
                    window: tuple[float, float], dtype: str, color: str | None) -> list[str]:
    comment = (f"squid record-zstack export: gray8 = ({dtype} - {window[0]:g}) * 255 / "
               f"{window[1] - window[0]:g}; window {window[0]:g}-{window[1]:g}"
               + (f"; color {color} (rgb = gray8 * color / 255)" if color else ""))
    return [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24" if color else "gray", "-s", f"{width}x{height}",
            "-r", f"{fps:.3f}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-metadata", f"comment={comment}", "-f", "mp4", str(out)]


def convert_store(store: str | Path, out: str | Path, *, fps: float | None = None,
                  window: tuple[float, float] | None = None, color: str | None = None,
                  on_progress: Callable[[int, int], None] | None = None,
                  cancelled: Callable[[], bool] | None = None) -> ConvertResult:
    """Encode one recording store to the mp4 file `out`.

    `window` (lo, hi) and `color` ("#RRGGBB") default to the store's own min
    and max and the store's channel colour, so a conversion with no viewer in
    the loop (CLI, Squid at acquisition time) is fully determined by the zarr.
    The viewer passes what is on screen instead, and the mp4 is a copy of the
    display. Either way both are written into the mp4's comment metadata, so
    the file states how it was made and can be regenerated from the zarr.

    Atomic: a `.tmp` file is renamed on success and removed on any failure,
    cancel or interrupt. `fps` defaults to real time. An odd side is cropped by
    one pixel (yuv420p needs even sides); nothing else is resampled. Frames are
    written exactly as the store reads them: a slot the recorder never wrote
    reads as the fill value and becomes a black frame. A coloured channel (see
    `channel_color`) is painted on the viewer's black-to-colour ramp after the
    window; a white one is written as gray. `cancelled()` is checked
    before every frame: a cancelled job is discarded, `error == CANCELLED`.
    """
    store, out = Path(store), Path(out)
    result = ConvertResult(store, out)
    proc: subprocess.Popen | None = None
    tmp = out.with_name(out.name + ".tmp")
    result.error = "interrupted"  # cleared only when every frame went through
    try:
        check_store(store)
        arr = open_frames(store)
        n_frames = frames_written(store, int(arr.shape[0]))
        if n_frames == 0:
            result.error = "no frames written"
            return result
        height, width = (int(s) - int(s) % 2 for s in arr.shape[-2:])
        fps = fps or default_fps(store)
        lo, hi = result.window = _window(window) if window is not None else data_window(arr, n_frames)
        color = result.color = parse_color(color) if color is not None else channel_color(store)
        ramp = color_ramp(color) if color else None
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(_ffmpeg_command(width, height, fps, tmp, (lo, hi), arr.dtype.name, color),
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in range(n_frames):
            if cancelled is not None and cancelled():
                result.error = CANCELLED
                break
            plane = arr[f, 0, 0, :height, :width].read().result()
            gray8 = _to_uint8(plane, lo, hi)
            proc.stdin.write((ramp[gray8] if ramp is not None else gray8).tobytes())
            result.frames += 1
            if on_progress is not None:
                on_progress(f + 1, n_frames)
        else:
            result.error = None
    except Exception as exc:  # torn zarr.json mid-write, tensorstore read error, broken pipe
        result.error = str(exc) or type(exc).__name__
    finally:
        if proc is not None:
            _, stderr = proc.communicate()  # closes stdin (EPIPE-safe), drains stderr, waits
            if result.error is None and proc.returncode != 0:
                result.error = stderr.decode(errors="replace").strip() or f"ffmpeg exited with {proc.returncode}"
        if result.error is None:
            os.replace(tmp, out)
        else:
            tmp.unlink(missing_ok=True)
    return result
