"""CLI: ``python -m record_zstack_viewer.export EXPERIMENT [--well A1] [--out DIR]``.

Converts every recording store of an experiment (or of the given wells) to
mp4, one file per store, skipping files that are already up to date. Qt-free,
so it runs on a headless box.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.export.mp4 import default_out_dir, export_all, plan
from record_zstack_viewer.model.experiment import ExperimentModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="record_zstack_viewer.export",
        description="Export squid Record + Z-Stack recordings (OME-Zarr v3) to mp4.")
    parser.add_argument("experiment", type=Path, help="experiment root (holds recording/)")
    parser.add_argument("--well", action="append", metavar="WELL",
                        help="only this well (repeatable); default: all wells")
    parser.add_argument("--out", type=Path, default=None,
                        help="output folder (default: ~/Downloads/videos-<experiment name>)")
    parser.add_argument("--fps", type=float, default=None,
                        help="playback rate (default: real time, as recorded)")
    parser.add_argument("--overwrite", action="store_true", help="re-export existing mp4s")
    args = parser.parse_args(argv)

    root = args.experiment.expanduser()
    if not (root / "recording").is_dir():
        parser.error(f"{root} has no recording/ folder")
    out_dir = args.out.expanduser() if args.out is not None else default_out_dir(root)

    model = ExperimentModel(root, default_context())
    jobs = plan(model, out_dir, wells=args.well, overwrite=args.overwrite)
    if not jobs:
        print("nothing to export (all mp4s up to date)")
        return 0
    failures = 0
    for result in export_all(model, jobs, fps=args.fps):
        name = result.job.out.name
        if result.ok:
            note = f" ({result.held} held)" if result.held else ""
            print(f"{name}: {result.frames} frames{note}")
        else:
            failures += 1
            print(f"{name}: FAILED: {result.error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
