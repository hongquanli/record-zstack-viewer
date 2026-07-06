"""Task 1.2 — FovSource protocol + TensorStore cache-pool context.

`open_array` opens a zarr3 store lazily through a shared `cache_pool` Context
(Global Constraints: memory is O(1) in dataset size via a single bounded
cache_pool shared by every read). This file starts the recording-reader tests;
the concrete RecordingFovSource lands in Task 1.4.

Sharing is asserted via `spec(retain_context=True)`, which surfaces the pool
resource the array is bound to — two opens with the same Context bind the
identical configured pool, a context-less open binds the empty default.
"""
import numpy as np
import tensorstore as ts

from record_zstack_viewer.cache.memory import default_context, open_array
from record_zstack_viewer.reader.fov_source import FovSource
from record_zstack_viewer.reader.metadata import FovMetadata

_DEFAULT_LIMIT = 192 * 1024 * 1024


def _bound_pool(arr) -> dict:
    """The cache_pool resource `arr` is actually bound to."""
    return arr.spec(retain_context=True).to_json()["context"]["cache_pool"]


def test_open_array_shares_cache_context(tiny_experiment):
    ctx = default_context()
    st = tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr"
    a = open_array(st, ctx)
    assert a.ndim == 5 and a.shape[1:3] == (1, 1)


def test_default_context_is_a_context():
    assert isinstance(default_context(), ts.Context)


def test_default_context_configures_byte_limit(tiny_experiment):
    limit = 64 * 1024 * 1024
    a = open_array(tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr",
                   default_context(byte_limit=limit))
    assert _bound_pool(a)["total_bytes_limit"] == limit


def test_default_context_default_limit_is_192mb(tiny_experiment):
    a = open_array(tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr",
                   default_context())
    assert _bound_pool(a)["total_bytes_limit"] == _DEFAULT_LIMIT


def test_two_opens_share_the_same_pool(tiny_experiment):
    ctx = default_context()
    st = tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr"
    a, b = open_array(st, ctx), open_array(st, ctx)
    # both bind the one configured pool; a context-less open would bind {}
    assert _bound_pool(a) == _bound_pool(b) == {"total_bytes_limit": _DEFAULT_LIMIT}


def test_open_array_is_lazy_and_reads_per_plane(tiny_experiment):
    ctx = default_context()
    a = open_array(tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr", ctx)
    # opening yields a lazy TensorStore array, not a materialized ndarray
    assert not isinstance(a, np.ndarray)
    plane = a[0, 0, 0].read().result()  # explicit per-plane read
    assert plane.shape == (64, 64) and plane.dtype == np.uint16


class _Stub:
    """Minimal structurally-conforming FovSource."""
    dims = ("acq_t", "frame", "z", "y", "x")
    shape = (2, 1, 1, 64, 64)
    dtype = np.dtype("uint16")
    metadata = FovMetadata(0.5, 2.0, 1.0, [], True, False, 1)

    def read_plane(self, index):
        return np.zeros((64, 64), "uint16")

    def read_tile(self, index, y0, y1, x0, x1):
        return np.zeros((y1 - y0, x1 - x0), "uint16")

    def filled_extent(self, axis):
        return 2


def test_fov_source_protocol_accepts_conforming_object():
    assert isinstance(_Stub(), FovSource)


def test_fov_source_protocol_rejects_incomplete_object():
    class Missing:  # only one attr, no methods
        dims = ()

    assert not isinstance(Missing(), FovSource)
