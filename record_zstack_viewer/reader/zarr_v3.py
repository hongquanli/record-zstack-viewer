"""Concrete `FovSource` implementations backed by TensorStore zarr3 stores.

Each acquisition mode is one class here; both present the uniform 5-D FovSource
surface so the model/UI never branch on mode:
  - `ZStackFovSource` — dims `(acq_t, c, z, y, x)`  (this task, 1.3)
  - `RecordingFovSource` — dims `(acq_t, frame, z, y, x)`  (Task 1.4)

All pixel reads go through the shared `cache.memory` context and pull only the
one visible plane / tile — never a whole array (Global Constraints).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tensorstore as ts

from record_zstack_viewer.cache.memory import open_array
from record_zstack_viewer.reader.metadata import (
    FovMetadata, fov_metadata_from_attrs, read_store_zarr_json)


class ZStackFovSource:
    """A z-stack recording's pixels: group at `group_dir`, array at `group_dir/0`.

    The store's T axis is the acquisition-timepoint axis, so the FovSource dims
    are `(acq_t, c, z, y, x)`. The pixel array lives at `/0`; the FOV metadata
    (channels, `_squid` scalars/flags, placement) lives in the *group* node's
    `zarr.json`, one level up — read from there, not from the array node.
    """

    dims: tuple[str, ...] = ("acq_t", "c", "z", "y", "x")

    def __init__(self, group_dir: str | Path, context: ts.Context):
        self._group_dir = Path(group_dir)
        self._arr = open_array(self._group_dir / "0", context)
        self.shape: tuple[int, ...] = tuple(self._arr.shape)
        # ts.dtype -> np.dtype: the FovSource protocol and all downstream code
        # expect a numpy dtype, and `self._arr.dtype` is a tensorstore dtype.
        self.dtype: np.dtype = np.dtype(self._arr.dtype.numpy_dtype)
        attrs = read_store_zarr_json(self._group_dir).get("attributes") or {}
        self.metadata: FovMetadata = fov_metadata_from_attrs(attrs)

    def _plane_view(self, index: dict[str, int]) -> ts.TensorStore:
        """Lazy (y, x) view for `index`, selecting non-spatial axes by NAME."""
        return self._arr[index["acq_t"], index["c"], index["z"]]

    def read_plane(self, index: dict[str, int]) -> np.ndarray:
        """The full (y, x) plane at `(acq_t, c, z)` as a 2-D numpy array."""
        return self._plane_view(index).read().result()

    def read_tile(self, index: dict[str, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """The `(y0:y1, x0:x1)` sub-region of the plane at `(acq_t, c, z)`."""
        return self._plane_view(index)[y0:y1, x0:x1].read().result()

    def filled_extent(self, axis: str) -> int:
        """Leading count of real (non-fill) data along `axis`.

        Minimal Task-1.3 stub: report the declared size — correct for a complete
        store. The FAST chunk-presence probing for live/partial stores lands in
        Task 1.5.
        """
        return self.shape[self.dims.index(axis)]


def _squid_of(store_dir: Path) -> dict:
    """The `_squid` attribute block of a recording store's root `zarr.json`."""
    return (read_store_zarr_json(store_dir).get("attributes") or {}).get("_squid") or {}


def _z_from_name(name: str) -> int:
    """The `j` in a `fov_{k}_z{j}.ome.zarr` name (fallback plane order when
    `_squid.plane_index` is absent); 0 if the name carries no `_z` suffix."""
    m = re.search(r"_z(\d+)\.ome\.zarr$", name)
    return int(m.group(1)) if m else 0


def _plane_stores(well_dir: Path, fov: int) -> list[tuple[int, Path]]:
    """Ordered `(plane_index, store)` planes of `fov` under one `t{t}/{well}` dir.

    Multi-plane: the `fov_{fov}_z*.ome.zarr` siblings, ordered by
    `_squid.plane_index` (filename `_z{j}` as fallback). Single-plane: a lone
    `fov_{fov}.ome.zarr` → one plane at index 0. Empty when neither exists.
    The `_z*` glob is exact — `fov_1_z*` never matches `fov_10_z*`.
    """
    zstores = list(well_dir.glob(f"fov_{fov}_z*.ome.zarr"))
    if zstores:
        indexed = []
        for p in zstores:
            pidx = _squid_of(p).get("plane_index")
            indexed.append((int(pidx) if pidx is not None else _z_from_name(p.name), p))
        indexed.sort(key=lambda pi_p: (pi_p[0], pi_p[1].name))
        return indexed
    single = well_dir / f"fov_{fov}.ome.zarr"
    return [(0, single)] if single.is_dir() else []


class RecordingFovSource:
    """A multi-plane recording's pixels, composited into the uniform 5-D surface.

    A recording is one store PER `(acq_t, plane)`:
    `recording/t{acq_t}/{well}/fov_{fov}[_z{z}].ome.zarr`, each an array at the
    STORE ROOT with shape `(T,1,1,Y,X)` whose T axis is the `frame` axis. This
    source stacks the `recording/t{t}` dirs into `acq_t` and the sibling planes
    (ordered by `_squid.plane_index`) into `z`, so dims are
    `(acq_t, frame, z, y, x)`. `read_plane({"acq_t":a,"frame":f,"z":z})` opens
    the one store for `(a, z)` and reads frame `f` as a 2-D `(y, x)` plane —
    never a whole array. A single-plane FOV (lone `fov_{k}.ome.zarr`) is z-size 1.

    `well_dir_by_t` maps each acquisition timepoint to its `t{t}/{well}` dir; the
    `acq_t` axis is dense (0..n-1) over those timepoints in sorted order, so a
    missing `t{t}` dir simply shifts the axis rather than leaving a hole.
    """

    dims: tuple[str, ...] = ("acq_t", "frame", "z", "y", "x")

    def __init__(self, well_dir_by_t: dict[int, Path], fov: int, context: ts.Context):
        self._context = context
        self._fov = int(fov)
        acq_ts = sorted(well_dir_by_t)  # dense acq_t axis over sorted timepoints
        if not acq_ts:
            raise FileNotFoundError(f"no recording timepoints for fov {fov}")
        # (acq_t index, z index) -> the store holding that plane's frames
        self._store: dict[tuple[int, int], Path] = {}
        planes_by_acq: list[list[tuple[int, Path]]] = []
        for ai, t in enumerate(acq_ts):
            planes = _plane_stores(Path(well_dir_by_t[t]), self._fov)
            if not planes:
                raise FileNotFoundError(f"no fov_{fov} store under {well_dir_by_t[t]}")
            planes_by_acq.append(planes)
            for zi, (_pidx, store) in enumerate(planes):
                self._store[(ai, zi)] = store

        ref_planes = planes_by_acq[0]          # reference timepoint defines the axes
        ref_store = ref_planes[0][1]
        ref_arr = open_array(ref_store, context)
        t_frames, _c, _z, ny, nx = ref_arr.shape
        self.shape: tuple[int, ...] = (
            len(acq_ts), int(t_frames), len(ref_planes), int(ny), int(nx))
        # ts.dtype -> np.dtype: the FovSource protocol expects a numpy dtype.
        self.dtype: np.dtype = np.dtype(ref_arr.dtype.numpy_dtype)

        attrs = read_store_zarr_json(ref_store).get("attributes") or {}
        meta = fov_metadata_from_attrs(attrs)  # XY placement, channels, pixel size, ...
        meta.n_planes = len(ref_planes)
        # per-plane z offsets collected across sibling stores, ordered by plane_index
        meta.plane_z_offsets = [
            float(_squid_of(store).get("plane_z_offset_um", 0.0)) for _pidx, store in ref_planes]
        self.metadata: FovMetadata = meta

    def _plane_view(self, index: dict[str, int]) -> ts.TensorStore:
        """Lazy (y, x) view of `frame` in the store for `(acq_t, z)`.

        Opens the one target store per call (bound to the shared cache_pool);
        the store-handle LRU that avoids re-opening lands in Task 3.1.
        """
        arr = open_array(self._store[(index["acq_t"], index["z"])], self._context)
        return arr[index["frame"], 0, 0]

    def read_plane(self, index: dict[str, int]) -> np.ndarray:
        """The full (y, x) plane at `(acq_t, frame, z)` as a 2-D numpy array."""
        return self._plane_view(index).read().result()

    def read_tile(self, index: dict[str, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """The `(y0:y1, x0:x1)` sub-region of the plane at `(acq_t, frame, z)`."""
        return self._plane_view(index)[y0:y1, x0:x1].read().result()

    def filled_extent(self, axis: str) -> int:
        """Leading count of real (non-fill) data along `axis`.

        Minimal Task-1.4 stub: report the declared size. FAST chunk-presence
        probing for live/partial recordings lands in Task 1.5.
        """
        return self.shape[self.dims.index(axis)]


def build_recording_source(root: str | Path, well: str, fov: int,
                           context: ts.Context) -> RecordingFovSource:
    """Assemble the `RecordingFovSource` for `(well, fov)` under `root`.

    Globs `recording/t*/{well}/fov_{fov}*.ome.zarr`, groups the matches by their
    `t{t}` dir into `well_dir_by_t`, and hands off to `RecordingFovSource` (which
    orders each timepoint's sibling planes by `plane_index`). Raises
    `FileNotFoundError` when no matching store exists.
    """
    root = Path(root)
    rec = root / "recording"
    if not rec.is_dir():
        raise FileNotFoundError(f"no recording/ under {root}")
    well_dir_by_t: dict[int, Path] = {}
    for store in rec.glob(f"t*/{well}/fov_{fov}*.ome.zarr"):
        # the glob's fov_{fov}* is loose (fov_1* also hits fov_10); keep exact matches only
        if store.name != f"fov_{fov}.ome.zarr" and not store.name.startswith(f"fov_{fov}_z"):
            continue
        m = re.fullmatch(r"t(\d+)", store.parent.parent.name)  # recording/t{t}/{well}/store
        if m:
            well_dir_by_t[int(m.group(1))] = store.parent
    if not well_dir_by_t:
        raise FileNotFoundError(f"no recording store for well {well!r} fov {fov} under {rec}")
    return RecordingFovSource(well_dir_by_t, fov, context)
