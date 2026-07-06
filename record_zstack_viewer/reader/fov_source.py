"""`FovSource` — the uniform read interface over a single FOV's pixels.

Both acquisition modes present the same 5-D shape through this protocol, so the
UI/model layers never branch on mode:
  - Z-stack recording  → dims `(acq_t, c, z, y, x)`   (Task 1.3 ZStackFovSource)
  - Multi-plane record  → dims `(acq_t, frame, z, y, x)` (Task 1.4 RecordingFovSource)

`dims` names each axis; `read_plane`/`read_tile` select the non-spatial axes by
name (never by position) and return only the visible (y, x) pixels — honoring
the Global Constraint that no read pulls a whole array. `filled_extent` reports
how far real (non-fill) data reaches along an axis, for live/partial stores.

This is a structural `Protocol` (runtime-checkable): concrete sources satisfy it
by shape, not by inheritance.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from record_zstack_viewer.reader.metadata import FovMetadata


@runtime_checkable
class FovSource(Protocol):
    """Per-FOV read interface backed by a lazy TensorStore array."""

    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype
    metadata: FovMetadata

    def read_plane(self, index: dict[str, int]) -> np.ndarray:
        """Return the full (y, x) plane selected by the non-spatial axes in `index`."""
        ...

    def read_tile(self, index: dict[str, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Return the (y0:y1, x0:x1) sub-region of the plane selected by `index`."""
        ...

    def filled_extent(self, axis: str) -> int:
        """Count of leading indices along `axis` that hold real (non-fill) data."""
        ...
