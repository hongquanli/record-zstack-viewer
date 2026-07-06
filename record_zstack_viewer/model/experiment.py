"""`ExperimentModel` — the plate/scan tree the UI navigates.

Walks an experiment's `recording/` and `zarr/` subtrees, unions the `(well, fov)`
pairs found in either, and for each FOV records what phases exist, a coarse
acquisition `status`, and a stage position in mm. Reads are metadata-only here
(`zarr.json` + directory listings); pixels stay untouched until `open_fov` hands
a FOV to the reader — so building the model is O(number of stores), never
O(dataset size), honoring the memory constraint. No Qt, no ndv, no tensorstore
opened directly (the reader adapter owns that).

**Positioning is translation-first** (a per-FOV `xy_mm`, or None):

1. the per-store OME `translation` (`FovMetadata.position_um`, um -> mm) — the
   primary, mm-exact path for data written by current squid;
2. else `coordinates.csv` at the experiment root (already mm);
3. else reconstruct from `acquisition.yaml`'s `wellplate_scan` geometry — the
   region center plus the FOV-grid + scan-pattern math in `model.geometry`;
4. else (no `wellplate_scan`, e.g. a "Current Position" run) a flat FOV list with
   `xy_mm=None`.

Reconstruction (3) cross-checks the geometry grid against the FOVs actually
present: `fov_grid_shape` can be off by a tile at region edges, so the grid is
reconciled to hold exactly the FOVs found before the scan pattern maps each index
to a cell — otherwise FOVs would land in the wrong cells.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

import tensorstore as ts

from record_zstack_viewer.model.geometry import (
    fov_grid_shape, infer_plate_format, spattern_index_to_ij,
    unidirectional_index_to_ij, well_name_to_rc)
from record_zstack_viewer.reader.adapter import FovDatasets
from record_zstack_viewer.reader.adapter import open_fov as _reader_open_fov
from record_zstack_viewer.reader.metadata import (
    FovMetadata, fov_metadata_from_attrs, parse_acquisition_yaml,
    read_store_zarr_json)

_FOV_RE = re.compile(r"^fov_(\d+)(?:_z\d+)?\.ome\.zarr$")
_T_RE = re.compile(r"^t(\d+)$")


@dataclass
class FovNode:
    """One FOV's place in the plate tree: which phases exist, where it sits, and
    a coarse acquisition status.

    `status` is one of ``"empty"`` (a store exists but no pixels yet),
    ``"in_progress"`` (some data, not finalized), ``"complete"`` (finalized or the
    run's `.done` marker is present), or ``"aborted"``. `xy_mm` is the stage
    position in mm, or None when no source (translation / coordinates.csv /
    geometry) can supply one.
    """

    well: str
    fov: int
    has_recording: bool
    has_zstack: bool
    xy_mm: tuple[float, float] | None
    status: str


@dataclass
class _FovStores:
    """The on-disk stores backing one `(well, fov)`, gathered during a tree walk."""

    zstack_group: Path | None = None                    # zarr/{well}/fov_{k}.ome.zarr
    recording_by_t: dict[int, list[Path]] | None = None  # t -> plane store dirs

    def __post_init__(self) -> None:
        if self.recording_by_t is None:
            self.recording_by_t = {}

    @property
    def has_recording(self) -> bool:
        return bool(self.recording_by_t)

    @property
    def has_zstack(self) -> bool:
        return self.zstack_group is not None

    def recording_rep(self) -> Path | None:
        """A representative recording store: the latest timepoint's first plane.

        Latest `t` reflects the FOV's current progress (status); any plane carries
        the FOV's shared XY (position). None when there is no recording phase.
        """
        if not self.recording_by_t:
            return None
        latest = self.recording_by_t[max(self.recording_by_t)]
        return sorted(latest, key=lambda p: p.name)[0] if latest else None


class ExperimentModel:
    """The navigable view of one experiment: wells, FOVs, status, positions.

    Construct with the experiment `root` and a shared tensorstore `context` (used
    only by `open_fov`, when a FOV is actually viewed). `refresh()` re-walks the
    tree so a growing live acquisition can be re-read cheaply; the constructor
    calls it once.
    """

    def __init__(self, root: str | Path, context: ts.Context):
        self.root = Path(root)
        self._context = context
        self._acq: dict = {}
        self._coords: dict[tuple[str, int], tuple[float, float]] = {}
        self._expected: dict[str, int] = {}
        self._stores: dict[str, dict[int, _FovStores]] = {}
        self._nodes: dict[str, list[FovNode]] = {}
        self.refresh()

    # --- public API ----------------------------------------------------------

    def wells(self) -> list[str]:
        """Wells present in the tree, ordered by plate position (non-well region
        ids, e.g. a "Current Position" run, sort last by name)."""
        return sorted(self._nodes, key=_well_sort_key)

    def fovs(self, well: str) -> list[FovNode]:
        """The `FovNode`s for `well`, ordered by FOV index (empty for unknown wells)."""
        return list(self._nodes.get(well, []))

    def fov_count_expected(self, well: str) -> int | None:
        """Expected FOV count for `well` from `events.jsonl`'s `run_started`, or
        None when that sidecar is absent (the count is then simply unknown)."""
        return self._expected.get(well)

    def plate_format(self) -> str:
        """Best-guess plate-format key (`"96"`, ...) from region-center spacing, or
        `"unknown"` when there is no `wellplate_scan` or it is ambiguous."""
        centers = [tuple(r["center_mm"][:2]) for r in self._regions().values()
                   if isinstance(r.get("center_mm"), (list, tuple))
                   and len(r["center_mm"]) >= 2]
        return infer_plate_format([(float(x), float(y)) for x, y in centers])

    def open_fov(self, well: str, fov: int) -> FovDatasets:
        """Open the reader sources for `(well, fov)` (lazy; pixels read on demand)."""
        return _reader_open_fov(self.root, well, int(fov), self._context)

    def refresh(self) -> None:
        """Re-read sidecars and re-walk the tree, rebuilding every `FovNode`."""
        self._acq = self._load_acquisition()
        self._coords = self._load_coordinates()
        self._expected = self._load_expected_fovs()
        self._stores = self._walk_tree()
        done = (self.root / ".done").exists()
        self._nodes = {}
        for well in self._stores:
            self._nodes[well] = self._build_well_nodes(well, done)

    # --- sidecar loading -----------------------------------------------------

    def _load_acquisition(self) -> dict:
        path = self.root / "acquisition.yaml"
        try:
            return parse_acquisition_yaml(path) if path.exists() else {}
        except (OSError, ValueError):
            return {}

    def _regions(self) -> dict[str, dict]:
        """`{region_name: region_dict}` from `wellplate_scan.regions` (empty when
        there is no `wellplate_scan` block — a Current-Position run)."""
        scan = self._acq.get("wellplate_scan") or {}
        return {str(r.get("name")): r for r in (scan.get("regions") or [])
                if isinstance(r, dict) and r.get("name") is not None}

    def _load_coordinates(self) -> dict[tuple[str, int], tuple[float, float]]:
        """Parse `coordinates.csv` -> `{(region, fov): (x_mm, y_mm)}` (first row per
        FOV wins; empty when the sidecar is absent or unreadable)."""
        path = self.root / "coordinates.csv"
        if not path.exists():
            return {}
        out: dict[tuple[str, int], tuple[float, float]] = {}
        try:
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
                reg_c = cols.get("region")
                fov_c = cols.get("fov")
                x_c = next((orig for low, orig in cols.items() if low.startswith("x")), None)
                y_c = next((orig for low, orig in cols.items() if low.startswith("y")), None)
                if not all((reg_c, fov_c, x_c, y_c)):
                    return {}
                for row in reader:
                    try:
                        key = (str(row[reg_c]).strip(), int(float(row[fov_c])))
                        xy = (float(row[x_c]), float(row[y_c]))
                    except (KeyError, ValueError, TypeError):
                        continue
                    out.setdefault(key, xy)
        except (OSError, csv.Error):
            return {}
        return out

    def _load_expected_fovs(self) -> dict[str, int]:
        """`expected_fovs_per_well` from `events.jsonl`'s `run_started` line, if any."""
        path = self.root / "events.jsonl"
        if not path.exists():
            return {}
        try:
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "run_started":
                        raw = ev.get("expected_fovs_per_well") or {}
                        return {str(k): int(v) for k, v in raw.items()}
        except OSError:
            return {}
        return {}

    # --- tree walk -----------------------------------------------------------

    def _walk_tree(self) -> dict[str, dict[int, _FovStores]]:
        """Union the `(well, fov)` pairs across `recording/` and `zarr/`.

        `recording/t{t}/{well}/fov_{k}[_z{j}].ome.zarr` contributes a per-timepoint
        plane store; `zarr/{well}/fov_{k}.ome.zarr` contributes the z-stack group.
        """
        stores: dict[str, dict[int, _FovStores]] = {}

        def slot(well: str, fov: int) -> _FovStores:
            return stores.setdefault(well, {}).setdefault(fov, _FovStores())

        rec = self.root / "recording"
        if rec.is_dir():
            for t_dir in rec.iterdir():
                m = _T_RE.match(t_dir.name)
                if not (m and t_dir.is_dir()):
                    continue
                t = int(m.group(1))
                for well_dir in t_dir.iterdir():
                    if not well_dir.is_dir():
                        continue
                    for store in well_dir.iterdir():
                        fm = _FOV_RE.match(store.name)
                        if fm and store.is_dir():
                            slot(well_dir.name, int(fm.group(1))
                                 ).recording_by_t.setdefault(t, []).append(store)

        zroot = self.root / "zarr"
        if zroot.is_dir():
            for well_dir in zroot.iterdir():
                if not well_dir.is_dir():
                    continue
                for store in well_dir.iterdir():
                    fm = _FOV_RE.match(store.name)
                    if fm and store.is_dir():
                        slot(well_dir.name, int(fm.group(1))).zstack_group = store
        return stores

    # --- node construction ---------------------------------------------------

    def _build_well_nodes(self, well: str, done: bool) -> list[FovNode]:
        fov_slots = self._stores[well]
        # per-FOV representative metadata (translation, complete/aborted flags)
        metas: dict[int, list[FovMetadata]] = {}
        positions: dict[int, tuple[float, float] | None] = {}
        for fov in sorted(fov_slots):
            fs = fov_slots[fov]
            metas[fov] = self._read_metas(fs)
            positions[fov] = self._position_direct(well, fov, metas[fov])

        # step 3: reconstruct any FOV still lacking a position, per well, from geometry
        recon = self._reconstruct_well(well, fov_slots, metas,
                                       need_recon=any(positions[k] is None for k in positions))

        nodes = []
        for fov in sorted(fov_slots):
            fs = fov_slots[fov]
            xy = positions[fov] if positions[fov] is not None else recon.get(fov)
            nodes.append(FovNode(
                well=well, fov=fov,
                has_recording=fs.has_recording, has_zstack=fs.has_zstack,
                xy_mm=xy, status=self._status(fs, metas[fov], done)))
        return nodes

    def _read_metas(self, fs: _FovStores) -> list[FovMetadata]:
        """Representative `FovMetadata` for each present phase (z-stack group node,
        latest recording store). zarr.json only — no tensorstore open."""
        out: list[FovMetadata] = []
        for store in (fs.zstack_group, fs.recording_rep()):
            if store is None:
                continue
            try:
                attrs = read_store_zarr_json(store).get("attributes") or {}
            except (OSError, ValueError):
                continue
            out.append(fov_metadata_from_attrs(attrs))
        return out

    def _status(self, fs: _FovStores, metas: list[FovMetadata], done: bool) -> str:
        """Coarse per-FOV status from the phase flags, the `.done` marker, and
        on-disk chunk presence (aborted > complete > in_progress > empty)."""
        if any(m.aborted for m in metas):
            return "aborted"
        if done or (metas and all(m.acquisition_complete for m in metas)):
            return "complete"
        return "in_progress" if self._has_written_chunks(fs) else "empty"

    def _has_written_chunks(self, fs: _FovStores) -> bool:
        """True when any backing store has at least one written chunk on disk.

        zarr3 writes each chunk under the array node's `c/` dir; the z-stack array
        is at `<group>/0`, a recording store is an array at its own root.
        """
        array_nodes: list[Path] = []
        if fs.zstack_group is not None:
            array_nodes.append(fs.zstack_group / "0")
        rep = fs.recording_rep()
        if rep is not None:
            array_nodes.append(rep)
        for node in array_nodes:
            try:
                with os.scandir(node / "c") as it:
                    if any(True for _ in it):
                        return True
            except (FileNotFoundError, NotADirectoryError):
                continue
        return False

    # --- positioning ---------------------------------------------------------

    def _position_direct(self, well: str, fov: int,
                         metas: list[FovMetadata]) -> tuple[float, float] | None:
        """Steps 1-2: the OME translation (um -> mm), else `coordinates.csv` (mm)."""
        for m in metas:                                  # step 1: translation
            if m.position_um is not None:
                x_um, y_um, _z = m.position_um
                return (x_um / 1000.0, y_um / 1000.0)
        return self._coords.get((well, fov))             # step 2: coordinates.csv (or None)

    def _reconstruct_well(self, well: str, fov_slots: dict[int, _FovStores],
                          metas: dict[int, list[FovMetadata]],
                          need_recon: bool) -> dict[int, tuple[float, float]]:
        """Step 3: reconstruct `{fov: (x_mm, y_mm)}` from `wellplate_scan` geometry.

        Empty when there is nothing to reconstruct from (no region / no FOV size)
        or when no FOV needs it — that leaves those FOVs at `xy_mm=None` (step 4).
        """
        if not need_recon:
            return {}
        region = self._regions().get(well)
        if not region or not isinstance(region.get("center_mm"), (list, tuple)):
            return {}
        cx, cy = float(region["center_mm"][0]), float(region["center_mm"][1])

        fov_wh = self._fov_size_mm(fov_slots, metas)
        if fov_wh is None:
            return {}
        fov_w, fov_h = fov_wh

        scan = self._acq.get("wellplate_scan") or {}
        overlap = float(scan.get("overlap_percent", 0.0) or 0.0)
        scan_size = scan.get("scan_size_mm", None)
        scan_xy = _as_xy(scan_size) if scan_size is not None else (fov_w, fov_h)

        nx, ny = fov_grid_shape(scan_xy, overlap, (fov_w, fov_h))
        # geometry can miss the true tile count by one at region edges; size the grid
        # to the FOVs actually present so each index maps to a real cell.
        need = max(fov_slots) + 1                        # highest S-pattern index in use
        nx, ny = _reconcile_grid(nx, ny, need)

        step_x = fov_w * (1.0 - overlap / 100.0)
        step_y = fov_h * (1.0 - overlap / 100.0)
        x0 = cx - (nx - 1) / 2.0 * step_x                # grid centered on the region
        y0 = cy - (ny - 1) / 2.0 * step_y
        index_to_ij = (unidirectional_index_to_ij
                       if _is_unidirectional(scan) else spattern_index_to_ij)

        out: dict[int, tuple[float, float]] = {}
        for fov in fov_slots:
            i, j = index_to_ij(fov, nx, ny)              # i = row (y), j = col (x)
            out[fov] = (x0 + j * step_x, y0 + i * step_y)
        return out

    def _fov_size_mm(self, fov_slots: dict[int, _FovStores],
                     metas: dict[int, list[FovMetadata]]) -> tuple[float, float] | None:
        """A FOV's physical (width, height) in mm = pixel size x pixel extent.

        Pixel size comes from any FOV's metadata; the (Y, X) pixel extent from the
        array node's declared shape. None when neither is readable.
        """
        pixel_um = next((m.pixel_size_um for ms in metas.values() for m in ms
                         if m.pixel_size_um), None)
        if not pixel_um:
            return None
        for fov in sorted(fov_slots):
            fs = fov_slots[fov]
            node = (fs.zstack_group / "0") if fs.zstack_group is not None \
                else fs.recording_rep()
            if node is None:
                continue
            try:
                shape = read_store_zarr_json(node).get("shape")
            except (OSError, ValueError):
                continue
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                ny, nx = int(shape[-2]), int(shape[-1])
                return (pixel_um * nx / 1000.0, pixel_um * ny / 1000.0)
        return None


def _well_sort_key(name: str) -> tuple[int, int, int, str]:
    """Order wells by ``(row, col)``; region ids that are not well names sort last."""
    try:
        r, c = well_name_to_rc(name)
        return (0, r, c, name)
    except ValueError:
        return (1, 0, 0, name)


def _as_xy(v) -> tuple[float, float]:
    """Scalar -> ``(v, v)``; a 2-sequence -> ``(x, y)``."""
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return float(v[0]), float(v[1])
    if isinstance(v, (tuple, list)) and len(v) == 1:
        return float(v[0]), float(v[0])
    return float(v), float(v)


def _is_unidirectional(scan: dict) -> bool:
    """Whether the intra-region FOV order is raster (not squid's default serpentine).

    squid scans a region's FOVs serpentine (S-pattern) by default; treat an
    explicit unidirectional/raster hint in `wellplate_scan` as the exception.
    """
    hint = str(scan.get("fov_pattern") or scan.get("pattern") or scan.get("xy_mode") or "")
    return any(w in hint.lower() for w in ("unidirection", "raster"))


def _reconcile_grid(nx: int, ny: int, need: int) -> tuple[int, int]:
    """Adjust a geometry grid ``(nx, ny)`` to hold exactly ``need`` FOVs.

    `fov_grid_shape` can be off by a tile at region edges; a synthetic tiny FOV can
    make it wildly oversized. Match the FOV count so each acquisition index maps to
    a real cell: keep the column width and extend rows when the grid under-counts;
    shrink to just fit (preserving the column:row aspect) when it over-counts.
    """
    nx, ny = max(1, int(nx)), max(1, int(ny))
    if need <= 0 or nx * ny == need:
        return nx, ny
    if nx * ny < need:
        return nx, math.ceil(need / nx)
    aspect = nx / ny
    nx2 = max(1, min(nx, math.ceil(math.sqrt(need * aspect))))
    return nx2, max(1, math.ceil(need / nx2))
