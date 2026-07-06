"""Task 1.6 — `FormatAdapter`: open the right source(s) for one FOV.

A squid experiment can hold two acquisition phases for the same FOV — a
multi-plane *recording* (`recording/t{t}/{well}/fov_{k}*.ome.zarr`) and a
*z-stack* (`zarr/{well}/fov_{k}.ome.zarr`). `open_fov` returns whichever phases
are present as a `FovDatasets`, each a uniform `FovSource`, so callers never
branch on the on-disk layout.

Format detection lives behind a small registry seam: each `FormatAdapter`
declares a `format` key, a `detect(root)` probe, and an `open_fov(...)`. v1 ships
only the Zarr-v3 adapter; the legacy layouts (INDIVIDUAL_IMAGES / OME-TIFF /
HCS-plate zarr) can register their own adapter later without touching callers.
Adapters reuse the Task 1.3/1.4 sources — no reading is reimplemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import tensorstore as ts

from record_zstack_viewer.reader.fov_source import FovSource
from record_zstack_viewer.reader.zarr_v3 import (
    ZStackFovSource, build_recording_source)


@dataclass
class FovDatasets:
    """The acquisition phases present for one FOV; either may be `None`.

    `recording` is the multi-plane recording (dims `(acq_t, frame, z, y, x)`),
    `zstack` the z-stack (dims `(acq_t, c, z, y, x)`). A FOV may carry one phase,
    the other, or both — an absent phase is `None`.
    """

    recording: FovSource | None = None
    zstack: FovSource | None = None


@runtime_checkable
class FormatAdapter(Protocol):
    """Opens a FOV for one on-disk storage format — the pluggable seam.

    Legacy layouts plug in later by registering an adapter whose `detect`
    recognizes the experiment's format and whose `open_fov` returns the uniform
    `FovDatasets`. Structural protocol: adapters satisfy it by shape, not
    inheritance.
    """

    format: str

    def detect(self, root: Path) -> bool:
        """True when `root` is an experiment of this adapter's format."""
        ...

    def open_fov(self, root: Path, well: str, fov: int,
                 context: ts.Context) -> FovDatasets:
        """Open the acquisition phases present for `(well, fov)` under `root`."""
        ...


_REGISTRY: dict[str, FormatAdapter] = {}


def register_adapter(adapter: FormatAdapter) -> None:
    """Register `adapter` under its `format` key (last registration wins)."""
    _REGISTRY[adapter.format] = adapter


def detect_adapter(root: str | Path) -> FormatAdapter:
    """The first registered adapter whose `detect` recognizes `root`.

    Raises `ValueError` when no adapter matches (e.g. a not-yet-supported legacy
    layout), so a format gap surfaces explicitly rather than as `None`.
    """
    root = Path(root)
    for adapter in _REGISTRY.values():
        if adapter.detect(root):
            return adapter
    raise ValueError(f"no registered FormatAdapter recognizes {root}")


def _is_fov_store(name: str, fov: int) -> bool:
    """Exact `fov` match for the loose `fov_{fov}*` glob (fov_1* also hits fov_10)."""
    return name == f"fov_{fov}.ome.zarr" or name.startswith(f"fov_{fov}_z")


class ZarrV3Adapter:
    """The v1 (Zarr-v3, `driver: zarr3`) layout: `recording/` + `zarr/` trees.

    `detect` recognizes an experiment by the presence of either subtree;
    `open_fov` returns whichever phase actually has a store for the FOV, reusing
    `build_recording_source` (Task 1.4) and `ZStackFovSource` (Task 1.3) — this
    adapter never touches pixels itself.
    """

    format = "ZARR_V3"

    def detect(self, root: Path) -> bool:
        return (root / "recording").is_dir() or (root / "zarr").is_dir()

    def _has_recording(self, root: Path, well: str, fov: int) -> bool:
        """True when an exact-match recording store exists for `(well, fov)`.

        Mirrors `build_recording_source`'s exact-match filter so detection and
        construction agree (the loose glob otherwise matches fov_10 for fov_1).
        """
        rec = root / "recording"
        return rec.is_dir() and any(
            _is_fov_store(p.name, fov)
            for p in rec.glob(f"t*/{well}/fov_{fov}*.ome.zarr"))

    def open_fov(self, root: Path, well: str, fov: int,
                 context: ts.Context) -> FovDatasets:
        recording: FovSource | None = None
        if self._has_recording(root, well, fov):
            recording = build_recording_source(root, well, fov, context)
        zstack: FovSource | None = None
        group = root / "zarr" / well / f"fov_{fov}.ome.zarr"
        if group.is_dir():
            zstack = ZStackFovSource(group, context)
        return FovDatasets(recording=recording, zstack=zstack)


register_adapter(ZarrV3Adapter())


def open_fov(root: str | Path, well: str, fov: int,
             context: ts.Context) -> FovDatasets:
    """Open the acquisition phases present for `(well, fov)` under `root`.

    Detects the experiment's storage format via the adapter registry, then
    delegates to that adapter. v1 has one registered format (Zarr-v3); the
    result is a `FovDatasets` whose `recording`/`zstack` are `FovSource`s, or
    `None` for a phase with no store for this FOV.
    """
    root = Path(root)
    return detect_adapter(root).open_fov(root, well, int(fov), context)
