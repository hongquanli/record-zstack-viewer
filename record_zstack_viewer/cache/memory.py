"""The single decoded-pixel cache: a shared TensorStore `cache_pool`.

Global Constraints require memory to be O(1) in dataset size. That budget lives
in *one* byte-capped `cache_pool` (default 192 MB) held by a `ts.Context` and
shared by every store the reader/ndv/prefetch open. `open_array` opens a zarr3
store lazily against that context, so all reads draw from — and are bounded by —
the same pool. There is no home-grown tile LRU.

Verified against tensorstore 0.1.77: `ts.Context({"cache_pool": {...}})` and
`ts.open({driver: zarr3, kvstore: {driver: file}, ...}, context=ctx)` both work;
`spec(retain_context=True)` shows two opens binding the identical pool resource.
"""
from __future__ import annotations

from pathlib import Path

import tensorstore as ts

_DEFAULT_BYTE_LIMIT = 192 * 1024 * 1024  # 192 MB


def default_context(byte_limit: int = _DEFAULT_BYTE_LIMIT) -> ts.Context:
    """A `ts.Context` whose `cache_pool` caps decoded-pixel memory at `byte_limit`.

    Reuse one instance across every `open_array` call so the stores share the
    single bounded pool (see module docstring).
    """
    return ts.Context({"cache_pool": {"total_bytes_limit": int(byte_limit)}})


def open_array(path: str | Path, context: ts.Context) -> ts.TensorStore:
    """Open the zarr3 array at `path` lazily, bound to `context`'s shared pool.

    Returns a lazy `TensorStore` (no pixels are read until an explicit
    `.read()`); reads are meant to be per-plane / per-chunk-tile, never whole.
    `recheck_cached_data="open"` revalidates cached chunks once at open time,
    which suits growing live stores without re-stat'ing on every read.
    """
    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
            "recheck_cached_data": "open",
        },
        context=context,
    ).result()
