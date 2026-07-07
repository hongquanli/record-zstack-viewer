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

import threading
from collections import OrderedDict
from pathlib import Path

import tensorstore as ts

_DEFAULT_BYTE_LIMIT = 192 * 1024 * 1024  # 192 MB
_DEFAULT_MAX_OPEN = 32


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


class HandleCache:
    """A bounded LRU of opened TensorStore handles, all bound to one `context`.

    The reader re-opens the same store constantly: every visible plane of a
    recording re-selects one per-`(acq_t, plane)` store, and a fresh `open_array`
    per read is wasteful (Tasks 1.4/1.5 flagged it). `get(path)` opens a store once
    and returns that *same* handle on repeat, so reads reuse the handle.

    Memory stays O(1) in dataset size on two independent axes: the shared
    `cache_pool` (held by `context`) caps decoded *bytes*, and this LRU caps the
    *number* of concurrently-open handles at `max_open`. Opening one more drops the
    least-recently-used handle — a plain reference drop, which is how TensorStore
    releases a handle (there is no explicit close) — so idle handles are freed.

    **Thread-safe.** The `LiveEngine` runs its poll/tick (which drives
    `ExperimentModel.refresh()` → `clear()`) on a background thread while the UI
    reads through the same cache, so a `threading.Lock` makes the check-then-act
    over the `OrderedDict` atomic. Without it, concurrent callers could double-open
    a path, lose an insert, exceed `max_open`, or raise a `KeyError` from
    `move_to_end` on a handle another thread just evicted. The lock is held across
    `open_array` too so two threads racing on the same missing path open it once;
    opens are fast local-file opens and contention is low (one background tick vs.
    the UI), so serializing them is a safe, simple choice.
    """

    def __init__(self, context: ts.Context, max_open: int = _DEFAULT_MAX_OPEN):
        self._context = context
        self._max_open = int(max_open)
        # insertion-ordered path -> handle; front = least-recently-used.
        self._handles: "OrderedDict[str, ts.TensorStore]" = OrderedDict()
        # guards every read/mutation of `_handles` (see class docstring).
        self._lock = threading.Lock()

    def get(self, path: str | Path) -> ts.TensorStore:
        """The opened handle for `path`, opening and caching it on first use.

        A repeated `get` of the same path returns the identical handle (no re-open)
        and marks it most-recently-used; opening beyond `max_open` distinct paths
        evicts (drops the reference to) the least-recently-used handle. Thread-safe:
        the whole check-open-insert-evict sequence runs under the lock.
        """
        key = str(path)
        with self._lock:
            handle = self._handles.get(key)
            if handle is not None:
                self._handles.move_to_end(key)      # mark most-recently-used
                return handle
            handle = open_array(path, self._context)
            self._handles[key] = handle             # new key lands at the MRU end
            while len(self._handles) > self._max_open:
                self._handles.popitem(last=False)   # drop the LRU handle's reference
            return handle

    def clear(self) -> None:
        """Drop every cached handle so the next `get` re-opens the store fresh.

        `ExperimentModel.refresh()` calls this so a growing live store's
        newly-written planes are re-opened (each `open_array` uses
        `recheck_cached_data="open"`, which revalidates at open time) rather than
        served stale from a long-lived handle. Thread-safe.
        """
        with self._lock:
            self._handles.clear()

    def __len__(self) -> int:
        """The number of handles currently held (always <= `max_open`)."""
        with self._lock:
            return len(self._handles)
