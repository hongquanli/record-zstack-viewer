"""Task 3.2 — `ThumbnailCache`: per-FOV preview thumbnails, three-tier lookup.

`get(well, fov)` returns a small (~128-256 px) 2-D uint8 array for one FOV, drawn
from the cheapest available source, in order:

  1. **Squid-written thumbnail** `<root>/thumbnails/{well}/fov_{fov}.png` — the
     producer already rendered it; read it (downsizing if larger) rather than
     recompute.
  2. **Persistent disk cache** under `platformdirs.user_cache_dir` — a thumbnail
     this viewer computed on a previous run, keyed by (well, fov) plus a token of
     the FOV's on-disk store state (mtimes) so a grown/finalized store bypasses a
     stale entry.
  3. **Compute** a representative plane through the reader (`open_fov`; z-stack ->
     mid-Z of a channel, recording -> mid frame), contrast-scale to 8-bit,
     downsample, and PERSIST it to the step-2 location for next time.

A bounded in-RAM LRU (`ram_items`) short-circuits all of the above for
recently-requested FOVs.

**Global Constraint — never write into the experiment folder.** Squid experiments
routinely live on Dropbox / a NAS / a read-only mount, so the computed-thumbnail
disk cache lives ONLY in the per-user cache dir (`platformdirs.user_cache_dir`),
namespaced by a hash of the experiment path. No cache file is ever created under
`experiment_root`.

No Qt/ndv here; decoding is reused from the reader (`open_fov`), never
reimplemented. Memory stays bounded: the RAM LRU caps the thumbnail count and the
reader's shared `cache_pool` caps decoded bytes during a compute.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import platformdirs
import tensorstore as ts

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.reader.adapter import open_fov
from record_zstack_viewer.reader.fov_source import FovSource

_APP_NAME = "record-zstack-viewer"
_TARGET_PX = 192          # ~128-256 px target for the thumbnail's long edge
_DEFAULT_RAM_ITEMS = 512


# --- pixel helpers -----------------------------------------------------------

def _read_png(path: Path) -> np.ndarray:
    """Decode a PNG to an array (via imageio, matching the fixture writer)."""
    import imageio.v3 as iio
    return np.asarray(iio.imread(path))


def _write_png(path: Path, arr: np.ndarray) -> None:
    """Atomically write a uint8 array as a PNG (temp file in-dir + os.replace),
    so a concurrent reader never sees a half-written thumbnail."""
    import imageio.v3 as iio
    tmp = path.parent / f"{path.stem}.tmp{os.getpid()}.png"     # keep .png so imageio infers PNG
    iio.imwrite(tmp, arr)
    os.replace(tmp, path)


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse an RGB(A) image to a single 2-D luminance channel; pass 2-D through."""
    a = np.asarray(arr)
    if a.ndim >= 3:                       # (H, W, C) -> drop alpha, average RGB
        a = a[..., :3].mean(axis=2)
    return a


def _downsample(arr: np.ndarray, target: int = _TARGET_PX) -> np.ndarray:
    """Decimate a 2-D array so its long edge is ~`target` px; never upsize.

    Strided decimation (dependency-free) is coarse but ample for a thumbnail: a
    real ~2000-px plane drops to ~200 px in a single step.
    """
    h, w = arr.shape[:2]
    if max(h, w) <= target:
        return arr
    step = int(np.ceil(max(h, w) / target))
    return arr[::step, ::step]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Contrast-scale a raw plane to 8-bit via its 1-99.9 percentile window.

    Mirrors the viewer's auto-contrast ethos so the thumbnail resembles the
    on-screen render; a flat/blank plane collapses to zeros instead of dividing
    by zero.
    """
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        return a
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1.0, 99.9))
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255.0 + 0.5).astype(np.uint8)


def _finalize_png(arr: np.ndarray) -> np.ndarray:
    """Normalize a *finished* 8-bit thumbnail PNG (squid- or self-written) to the
    contract shape — 2-D, uint8, long edge <= target — with NO re-contrast (the
    source is already a rendered thumbnail)."""
    a = _downsample(_to_2d(arr))
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return np.ascontiguousarray(a)


def _finalize_computed(plane: np.ndarray) -> np.ndarray:
    """Turn a raw decoded plane into the contract thumbnail: 2-D, downsampled,
    contrast-scaled to uint8."""
    return np.ascontiguousarray(_to_uint8(_downsample(_to_2d(plane))))


def _representative_plane(source: FovSource) -> np.ndarray:
    """Read one representative 2-D plane: the middle index of every non-spatial
    axis, clamped into the filled (non-fill) extent of a live-growing axis so a
    partially-written store still yields real pixels.

    This reads exactly one (y, x) plane (never a whole array), honoring the
    Global Constraint even for a huge live store.
    """
    index: dict[str, int] = {}
    for name, size in zip(source.dims, source.shape):
        if name in ("y", "x"):
            continue
        filled = source.filled_extent(name)      # == size for complete / non-live axes
        usable = filled if filled > 0 else size
        index[name] = min(usable // 2, size - 1)
    return source.read_plane(index)


class ThumbnailCache:
    """Three-tier per-FOV thumbnail cache: RAM LRU -> squid PNG -> user-cache disk
    -> compute+persist.

    See the module docstring for the lookup order and the never-write-into-the-
    experiment Global Constraint. The RAM LRU is keyed by `(well, fov)` and bounds
    the number of held thumbnails at `ram_items`; the disk cache additionally
    keys on a store-state token so a finalized/grown store invalidates a stale
    on-disk thumbnail across sessions.
    """

    def __init__(self, experiment_root: str | Path, ram_items: int = _DEFAULT_RAM_ITEMS):
        self._root = Path(experiment_root)
        self._ram_items = int(ram_items)
        # insertion-ordered (well, fov) -> thumbnail; front = least-recently-used.
        self._ram: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()
        self._context: ts.Context | None = None   # lazily created on the first compute

    def get(self, well: str, fov: int) -> np.ndarray:
        """The ~128-256 px, 2-D uint8 thumbnail for `(well, fov)`.

        A repeat request is served from the RAM LRU (no I/O); otherwise the squid
        PNG / disk cache / compute tiers are tried in order and the result is
        cached in RAM (evicting the least-recently-used entry past `ram_items`).
        """
        key = (str(well), int(fov))
        cached = self._ram.get(key)
        if cached is not None:
            self._ram.move_to_end(key)             # mark most-recently-used
            return cached
        thumb = self._load_or_compute(*key)
        self._ram[key] = thumb
        self._ram.move_to_end(key)
        while len(self._ram) > self._ram_items:
            self._ram.popitem(last=False)          # evict least-recently-used
        return thumb

    def _load_or_compute(self, well: str, fov: int) -> np.ndarray:
        squid = self._root / "thumbnails" / well / f"fov_{fov}.png"
        if squid.is_file():                        # 1. producer-rendered thumbnail
            return _finalize_png(_read_png(squid))

        disk = self._disk_path(well, fov)          # 2. our persistent user-cache copy
        if disk.is_file():
            return _finalize_png(_read_png(disk))

        thumb = self._compute(well, fov)           # 3. compute from a decoded plane ...
        self._persist(disk, thumb)                 #    ... and persist for next time
        return thumb

    def _compute(self, well: str, fov: int) -> np.ndarray:
        if self._context is None:                  # one bounded cache_pool for all computes
            self._context = default_context()
        datasets = open_fov(self._root, well, fov, self._context)
        source = datasets.zstack or datasets.recording
        if source is None:
            raise FileNotFoundError(
                f"no FOV store for well {well!r} fov {fov} under {self._root}")
        return _finalize_computed(_representative_plane(source))

    # -- disk cache location: ALWAYS the user-cache dir, NEVER the experiment ----

    def _cache_dir(self) -> Path:
        """`<user_cache_dir>/<hash(experiment_root)>/` — resolved at call time so a
        test can monkeypatch `platformdirs.user_cache_dir`."""
        digest = hashlib.sha256(str(self._root.resolve()).encode("utf-8")).hexdigest()[:16]
        return Path(platformdirs.user_cache_dir(_APP_NAME)) / digest

    def _disk_path(self, well: str, fov: int) -> Path:
        return self._cache_dir() / f"{well}_fov{fov}_{self._store_token(well, fov)}.png"

    def _store_token(self, well: str, fov: int) -> str:
        """A short cache-busting token from the FOV's on-disk store mtimes: a grown
        or finalized store (which adds chunks / rewrites zarr.json) yields a new
        token, so a stale disk thumbnail is bypassed rather than served."""
        stamps: list[int] = []
        candidates = [self._root / "zarr" / well / f"fov_{fov}.ome.zarr"]
        rec = self._root / "recording"
        if rec.is_dir():
            candidates += rec.glob(f"t*/{well}/fov_{fov}.ome.zarr")
            candidates += rec.glob(f"t*/{well}/fov_{fov}_z*.ome.zarr")
        for store in candidates:
            for p in (store, store / "zarr.json"):
                try:
                    stamps.append(p.stat().st_mtime_ns)
                except OSError:
                    pass
        return hashlib.sha256(repr(sorted(stamps)).encode("utf-8")).hexdigest()[:12]

    def _persist(self, disk: Path, thumb: np.ndarray) -> None:
        """Best-effort write of a computed thumbnail into the user-cache dir.

        Failures (e.g. a read-only cache dir) are swallowed — the thumbnail is
        still returned, just not cached. Stale-token siblings for the same
        (well, fov) are pruned first so a live-growing store's cache stays bounded.
        """
        try:
            disk.parent.mkdir(parents=True, exist_ok=True)
            stem = disk.name.rsplit("_", 1)[0]     # "{well}_fov{fov}" (drops the "_{token}.png")
            for old in disk.parent.glob(f"{stem}_*.png"):
                if old != disk:
                    old.unlink(missing_ok=True)
            _write_png(disk, thumb)
        except OSError:
            pass
