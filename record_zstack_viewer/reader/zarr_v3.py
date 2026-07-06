"""Concrete `FovSource` implementations backed by TensorStore zarr3 stores.

Each acquisition mode is one class here; both present the uniform 5-D FovSource
surface so the model/UI never branch on mode:
  - `ZStackFovSource` — dims `(acq_t, c, z, y, x)`  (this task, 1.3)
  - `RecordingFovSource` — dims `(acq_t, frame, z, y, x)`  (Task 1.4)

All pixel reads go through the shared `cache.memory` context and pull only the
one visible plane / tile — never a whole array (Global Constraints).
"""
from __future__ import annotations

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
