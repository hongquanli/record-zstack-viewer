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
mapping can be read back from the file alone.
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
CANCELLED = "cancelled"  # ConvertResult.error of a job stopped by the caller
_FALLBACK_FPS = 10.0     # when the store carries no timing at all


@dataclass
class ConvertResult:
    store: Path
    out: Path
    frames: int = 0                            # frames written to the mp4
    held: int = 0                              # never-written slots filled with the previous frame
    window: tuple[float, float] | None = None  # the 16-to-8 window used
    error: str | None = None                   # set when no file was written

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        """One log line: `A1_fov0.mp4: 30 frames, window 4–65534`."""
        if self.error == CANCELLED:
            return f"{self.out.name}: cancelled"
        if self.error is not None:
            return f"{self.out.name}: FAILED: {self.error}"
        held = f", {self.held} held frame(s)" if self.held else ""
        lo, hi = self.window
        return f"{self.out.name}: {self.frames} frames{held}, window {lo:g}–{hi:g}"


# ----------------------------------------------------------------- naming

def default_out_dir(experiment_dir: str | Path) -> Path:
    """`~/Downloads/mp4-<experiment folder name>/`, the one place both the viewer
    and Squid write to; outside the data folder, one folder per run."""
    return Path.home() / "Downloads" / f"mp4-{Path(experiment_dir).name}"


def video_name(well: str, fov: int, acq_t: int, z: int, n_t: int, n_z: int,
               partial: bool) -> str:
    """`A1_fov0.mp4`, with `_t{t}` / `_z{z}` only when that axis has more than one
    entry, and `_partial` when the store never completed."""
    name = f"{well}_fov{fov}"
    if n_t > 1:
        name += f"_t{acq_t}"
    if n_z > 1:
        name += f"_z{z}"
    if partial:
        name += "_partial"
    return name + ".mp4"


# ------------------------------------------------------------ store facts

def squid_attrs(store: Path) -> dict:
    """The `_squid` block of the store's `zarr.json` (empty when absent)."""
    meta = json.loads((Path(store) / "zarr.json").read_text())
    return (meta.get("attributes") or {}).get("_squid") or {}


def is_complete(store: Path) -> bool:
    return bool(squid_attrs(store).get("acquisition_complete", False))


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


def open_frames(store: Path) -> ts.TensorStore:
    """The store's array, read-only, lazily."""
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(store)},
                    "open": True, "create": False}).result()


# -------------------------------------------------------------- conversion

def data_window(arr: ts.TensorStore, n_frames: int) -> tuple[float, float]:
    """The video's own `(min, max)` over every written frame, one read pass."""
    lo, hi = np.inf, -np.inf
    for f in range(n_frames):
        plane = arr[f, 0, 0].read().result()
        if plane.any():  # a never-written slot is all fill value, not data
            lo, hi = min(lo, float(plane.min())), max(hi, float(plane.max()))
    if not np.isfinite(lo):
        lo, hi = 0.0, 0.0
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _to_uint8(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (plane.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def _ffmpeg_command(width: int, height: int, fps: float, out: Path,
                    window: tuple[float, float], dtype: str) -> list[str]:
    comment = (f"squid record-zstack export: gray8 = ({dtype} - {window[0]:g}) * 255 / "
               f"{window[1] - window[0]:g}; window {window[0]:g}-{window[1]:g}")
    return [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
            "-r", f"{fps:.3f}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-metadata", f"comment={comment}", "-f", "mp4", str(out)]


def convert_store(store: str | Path, out: str | Path, *, fps: float | None = None,
                  on_progress: Callable[[int, int], None] | None = None,
                  cancelled: Callable[[], bool] | None = None) -> ConvertResult:
    """Encode one recording store to the mp4 file `out`.

    Atomic: a `.tmp` file is renamed on success and removed on any failure,
    cancel or interrupt. `fps` defaults to real time. An odd side is cropped by
    one pixel (yuv420p needs even sides); nothing else is resampled. A frame
    slot that was never written (all fill value) repeats the previous frame
    rather than flashing black; the count is reported as `held` (a never-written
    slot before the first real frame stays black). `cancelled()` is checked
    before every frame: a cancelled job is discarded, `error == CANCELLED`.
    """
    store, out = Path(store), Path(out)
    result = ConvertResult(store, out)
    proc: subprocess.Popen | None = None
    tmp = out.with_name(out.name + ".tmp")
    previous: np.ndarray | None = None
    result.error = "interrupted"  # cleared only when every frame went through
    try:
        arr = open_frames(store)
        n_frames = frames_written(store, int(arr.shape[0]))
        if n_frames == 0:
            result.error = "no frames written"
            return result
        height, width = (int(s) - int(s) % 2 for s in arr.shape[-2:])
        fps = fps or default_fps(store)
        lo, hi = result.window = data_window(arr, n_frames)
        out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(_ffmpeg_command(width, height, fps, tmp, (lo, hi), arr.dtype.name),
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in range(n_frames):
            if cancelled is not None and cancelled():
                result.error = CANCELLED
                break
            plane = arr[f, 0, 0, :height, :width].read().result()
            if not plane.any() and previous is not None:
                plane, result.held = previous, result.held + 1
            previous = plane
            proc.stdin.write(_to_uint8(plane, lo, hi).tobytes())
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
