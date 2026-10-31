"""Viewer-side export: which recording stores to convert, and in what order.

The conversion itself lives in `convert.py`, self-contained so Squid can run
the identical code at acquisition time. This module adds what only the viewer
knows: the experiment tree (`ExperimentModel`), the per-store naming, and the
skip-if-up-to-date rule. No Qt here: the CLI and the dialog both call `plan()`
and `export_all()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from record_zstack_viewer.export.convert import (CANCELLED, ConvertResult, convert_store,
                                                 default_out_dir, video_name)
from record_zstack_viewer.model.experiment import ExperimentModel

__all__ = ["CANCELLED", "ConvertResult", "ExportJob", "default_out_dir", "export_all",
           "export_job", "plan", "video_name"]


@dataclass(frozen=True)
class ExportJob:
    well: str
    fov: int
    acq_t: int
    z: int
    store: Path
    out: Path


def plan(model: ExperimentModel, out_dir: Path, wells: list[str] | None = None) -> list[ExportJob]:
    """Every recording store of `wells` (default: all) as an `ExportJob`.

    An mp4 already on disk is rewritten (only ever mp4 files; the zarr is
    read-only). A store still being written (not complete while the run has no
    `.done` marker yet) is left out; a later run picks it up.
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
                    store = source.store_path(acq_t, z)
                    if not source.is_complete(acq_t, z) and not run_finished:
                        continue
                    out = out_dir / video_name(well, node.fov, acq_t, z, n_t, n_z)
                    jobs.append(ExportJob(well, node.fov, acq_t, z, store, out))
    return jobs


def export_job(job: ExportJob, **kwargs) -> ConvertResult:
    """Convert one planned store (see `convert_store` for the kwargs)."""
    return convert_store(job.store, job.out, **kwargs)


def export_all(jobs: list[ExportJob], **kwargs) -> Iterator[ConvertResult]:
    """`export_job` over `jobs`, in order, yielding each result as it finishes.
    Stops once a job reports `CANCELLED`."""
    for job in jobs:
        result = export_job(job, **kwargs)
        yield result
        if result.error == CANCELLED:
            return
