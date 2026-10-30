"""Export recording stores to H.264 mp4 files.

One recording store, i.e. one `(well, fov, acq_t, z-plane)`, becomes one mp4.
Frames stream through the existing `FovSource.read_plane`, one at a time, into
an ffmpeg pipe (the `imageio-ffmpeg` wheel ships the binary), so memory does
not grow with the video length. Contrast is one fixed `(lo, hi)` for the whole
video, by default the same 1-99.9 percentile rule the viewer applies on first
display, taken over a few sampled frames.

No Qt here: the CLI and the dialog both call `plan()` and `export_job()`.
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

AUTO_PERCENTILES = (1.0, 99.9)  # same rule as the viewer's first-display contrast
MAX_SIDE_PX = 4096              # H.264 level 5.1 decoder limit; larger frames are downscaled
DEFAULT_FPS = 10.0              # last resort when the store carries no timing at all
CRF = 18                        # visually lossless


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
    frames: int = 0           # frames written to the mp4
    held: int = 0             # unwritten slots filled by repeating the previous frame
    partial: bool = False     # the store was not acquisition_complete
    error: str | None = None  # set when nothing usable was written

    @property
    def ok(self) -> bool:
        return self.error is None


# ------------------------------------------------------------------ planning

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

    A store whose mp4 is already on disk and newer than the store is left out
    unless `overwrite`. A store still being written (not complete while the run
    has no `.done` marker yet) is left out; a later run picks it up.
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
    if meta.time_increment_s and meta.time_increment_s > 0:
        return round(1.0 / meta.time_increment_s, 3)
    return DEFAULT_FPS


def auto_clims(source: RecordingFovSource, acq_t: int, z: int, n_frames: int) -> tuple[float, float]:
    """Percentile limits over the first, middle and last written frame."""
    picks = sorted({0, n_frames // 2, max(n_frames - 1, 0)})
    sample = np.stack([source.read_plane({"acq_t": acq_t, "frame": f, "z": z}) for f in picks])
    lo, hi = (float(v) for v in np.percentile(sample, AUTO_PERCENTILES))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _to_uint8(plane: np.ndarray, lo: float, hi: float) -> np.ndarray:
    scaled = (plane.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _ffmpeg_command(width: int, height: int, fps: float, out: Path) -> list[str]:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
           "-r", f"{fps:.3f}", "-i", "-", "-an"]
    if max(width, height) > MAX_SIDE_PX:
        # keep aspect; -2 lets ffmpeg pick the other side, rounded to even
        cmd += ["-vf", f"scale={MAX_SIDE_PX}:-2" if width >= height else f"scale=-2:{MAX_SIDE_PX}"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(out)]
    return cmd


def export_job(model: ExperimentModel, job: ExportJob, *, fps: float | None = None,
               clims: tuple[float, float] | None = None,
               on_progress: Callable[[int, int], None] | None = None,
               cancelled: Callable[[], bool] | None = None) -> ExportResult:
    """Encode one store to `job.out`. Atomic: a `.tmp` file is renamed on success.

    `fps` defaults to real time (`default_fps`); `clims` to `auto_clims`. A frame
    slot that was never written (all fill value) repeats the previous frame
    rather than flashing black; the count is reported as `held`.
    """
    result = ExportResult(job)
    source = model.open_fov(job.well, job.fov).recording
    n_frames = source.frames_written(job.acq_t, job.z)
    result.partial = not source.is_complete(job.acq_t, job.z)
    if n_frames == 0:
        result.error = "no frames written"
        return result

    fps = fps or default_fps(source)
    lo, hi = clims or auto_clims(source, job.acq_t, job.z, n_frames)
    height, width = source.shape[-2:]
    height -= height % 2  # yuv420p needs even sides
    width -= width % 2

    job.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = job.out.with_name(job.out.name + ".tmp")
    proc = subprocess.Popen(_ffmpeg_command(width, height, fps, tmp),
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    previous: np.ndarray | None = None
    try:
        for f in range(n_frames):
            if cancelled is not None and cancelled():
                result.error = "cancelled"
                break
            plane = source.read_plane({"acq_t": job.acq_t, "frame": f, "z": job.z})[:height, :width]
            if not plane.any() and previous is not None:
                plane, result.held = previous, result.held + 1
            previous = plane
            proc.stdin.write(_to_uint8(plane, lo, hi).tobytes())
            result.frames += 1
            if on_progress is not None:
                on_progress(f + 1, n_frames)
    except BrokenPipeError:
        result.error = "ffmpeg stopped accepting frames"
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read().decode(errors="replace").strip()
        code = proc.wait()
    if result.error is None and code != 0:
        result.error = stderr or f"ffmpeg exited with {code}"
    if result.error is None:
        os.replace(tmp, job.out)
    else:
        tmp.unlink(missing_ok=True)
    return result


def export_all(model: ExperimentModel, jobs: list[ExportJob], **kwargs) -> Iterator[ExportResult]:
    """`export_job` over `jobs`, in order, yielding each result as it finishes.
    Stops after the current file once `cancelled()` turns true."""
    for job in jobs:
        result = export_job(model, job, **kwargs)
        yield result
        if result.error == "cancelled":
            return
