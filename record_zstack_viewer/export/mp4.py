"""Export recording stores to H.264 mp4 files.

One recording store, i.e. one `(well, fov, acq_t, z-plane)`, becomes one mp4.
Frames stream through the existing `FovSource.read_plane`, one at a time, into
an ffmpeg pipe (the `imageio-ffmpeg` wheel ships the binary), so memory does
not grow with the video length.

The zarr is the ground truth and is opened read-only. The mp4 keeps as much
of it as the format allows: full pixel resolution, every captured frame, the
real-time playback rate, and a near-lossless H.264 encode. What mp4 cannot
hold is 16-bit data: player-compatible H.264 is 8-bit, so every frame goes
through ONE linear window per video, the video's own min and max, which clips
nothing. The window is recorded in the result and in the mp4's metadata
(`comment`), so the mapping is auditable from the file alone.

No Qt here: the CLI and the dialog both call `plan()` and `export_all()`.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import imageio_ffmpeg
import numpy as np

from record_zstack_viewer.model.experiment import ExperimentModel
from record_zstack_viewer.reader.zarr_v3 import RecordingFovSource

CRF = 12                        # near-lossless; still plays everywhere
CANCELLED = "cancelled"         # ExportResult.error of a job stopped by the caller


@dataclass(frozen=True)
class ExportJob:
    well: str
    fov: int
    acq_t: int
    z: int
    out: Path


@dataclass
class ExportResult:
    job: ExportJob
    frames: int = 0                           # frames written to the mp4
    held: int = 0                             # unwritten slots filled by repeating the previous frame
    window: tuple[float, float] | None = None  # the 16-to-8 window used
    error: str | None = None                  # set when no file was written

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> str:
        """One log line: `A1_fov0.mp4: 30 frames, window 4–65534`."""
        name = self.job.out.name
        if self.error == CANCELLED:
            return f"{name}: cancelled"
        if self.error is not None:
            return f"{name}: FAILED: {self.error}"
        held = f", {self.held} held frame(s)" if self.held else ""
        lo, hi = self.window
        return f"{name}: {self.frames} frames{held}, window {lo:.0f}–{hi:.0f}"


# ------------------------------------------------------------------ planning

def default_out_dir(experiment_root: Path) -> Path:
    """`~/Downloads/videos-<experiment name>/`: outside the data folder, one folder per run."""
    return Path.home() / "Downloads" / f"videos-{Path(experiment_root).name}"


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


def plan(model: ExperimentModel, out_dir: Path, wells: list[str] | None = None,
         overwrite: bool = False) -> list[ExportJob]:
    """Every recording store of `wells` (default: all) as an `ExportJob`.

    An mp4 already on disk and newer than its store is left out (`overwrite`
    re-plans it; it only ever replaces mp4 files, the zarr is read-only). A
    store still being written (not complete while the run has no `.done`
    marker yet) is left out; a later run picks it up.
    """
    out_dir = Path(out_dir)
    run_finished = (Path(model.root) / ".done").exists()
    jobs: list[ExportJob] = []
    for well in wells if wells is not None else model.wells():
        for node in model.fovs(well):
            if not node.has_recording:
                continue
            source = model.open_fov(well, node.fov).recording
            n_t, _frames, n_z = source.shape[:3]
            for acq_t in range(n_t):
                for z in range(n_z):
                    complete = source.is_complete(acq_t, z)
                    if not complete and not run_finished:
                        continue
                    out = out_dir / video_name(well, node.fov, acq_t, z, n_t, n_z, not complete)
                    if not overwrite and _up_to_date(out, source.store_path(acq_t, z)):
                        continue
                    jobs.append(ExportJob(well, node.fov, acq_t, z, out))
    return jobs


def _up_to_date(out: Path, store: Path) -> bool:
    try:
        return out.stat().st_mtime >= (store / "zarr.json").stat().st_mtime
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------ encoding

def default_fps(source: RecordingFovSource) -> float:
    """Real-time playback: the rate squid recorded at, else 1 / frame interval."""
    meta = source.metadata
    if meta.effective_fps:
        return float(meta.effective_fps)
    return round(1.0 / meta.time_increment_s, 3) if meta.time_increment_s > 0 else 10.0


def data_window(source: RecordingFovSource, acq_t: int, z: int, n_frames: int) -> tuple[float, float]:
    """The video's own `(min, max)` over every written frame: the linear 16-to-8
    window that clips nothing. One read pass; frames are not kept."""
    lo, hi = np.inf, -np.inf
    for f in range(n_frames):
        plane = source.read_plane({"acq_t": acq_t, "frame": f, "z": z})
        if plane.any():  # a never-written slot is all fill value, not data
            lo, hi = min(lo, float(plane.min())), max(hi, float(plane.max()))
    if not np.isfinite(lo):
        lo, hi = 0.0, 0.0
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _to_uint8(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (plane.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)


def _ffmpeg_command(width: int, height: int, fps: float, out: Path,
                    window: tuple[float, float], dtype: np.dtype) -> list[str]:
    comment = (f"squid record-zstack export: gray8 = ({dtype.name} - {window[0]:g}) * 255 / "
               f"{window[1] - window[0]:g}; window {window[0]:g}-{window[1]:g}")
    return [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
            "-r", f"{fps:.3f}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-metadata", f"comment={comment}", "-f", "mp4", str(out)]


def export_job(model: ExperimentModel, job: ExportJob, *, fps: float | None = None,
               on_progress: Callable[[int, int], None] | None = None,
               cancelled: Callable[[], bool] | None = None) -> ExportResult:
    """Encode one store to `job.out`. Atomic: a `.tmp` file is renamed on success
    and removed on any failure, cancel or interrupt.

    `fps` defaults to real time (`default_fps`). The 16-to-8 window is the
    video's own min..max (`data_window`). An odd side is cropped by one pixel
    (yuv420p needs even sides); nothing else is resampled. A frame slot
    that was never written (all fill value) repeats the previous frame rather
    than flashing black; the count is reported as `held` (a never-written slot
    before the first real frame stays black). `cancelled()` is checked before
    every frame: a cancelled job is discarded, `error == CANCELLED`.
    """
    result = ExportResult(job)
    source = model.open_fov(job.well, job.fov).recording
    n_frames = source.frames_written(job.acq_t, job.z)
    if n_frames == 0:
        result.error = "no frames written"
        return result

    height, width = source.shape[-2:]
    height -= height % 2
    width -= width % 2
    job.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = job.out.with_name(job.out.name + ".tmp")
    proc: subprocess.Popen | None = None
    previous: np.ndarray | None = None
    result.error = "interrupted"  # cleared only when every frame went through
    try:
        fps = fps or default_fps(source)
        lo, hi = result.window = data_window(source, job.acq_t, job.z, n_frames)
        proc = subprocess.Popen(_ffmpeg_command(width, height, fps, tmp, (lo, hi), source.dtype),
                                stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in range(n_frames):
            if cancelled is not None and cancelled():
                result.error = CANCELLED
                break
            plane = source.read_plane({"acq_t": job.acq_t, "frame": f, "z": job.z})[:height, :width]
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
            os.replace(tmp, job.out)
        else:
            tmp.unlink(missing_ok=True)
    return result


def export_all(model: ExperimentModel, jobs: list[ExportJob], **kwargs) -> Iterator[ExportResult]:
    """`export_job` over `jobs`, in order, yielding each result as it finishes.
    Stops once a job reports `CANCELLED`."""
    for job in jobs:
        result = export_job(model, job, **kwargs)
        yield result
        if result.error == CANCELLED:
            return
