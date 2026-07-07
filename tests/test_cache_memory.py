"""Task 3.1 — `HandleCache`: a bounded LRU of opened TensorStore handles.

Global Constraints keep memory O(1) in dataset size on two independent axes: the
shared `cache_pool` caps decoded *bytes*, and this LRU caps the *number* of
concurrently-open store handles (default 32). A repeated `get` of the same path
returns the one cached handle (no re-open); opening past `max_open` drops the
least-recently-used handle so idle handles are released.
"""
import numpy as np

from record_zstack_viewer.cache.memory import (
    HandleCache, default_context, open_array)


def _recording_stores(exp):
    """Distinct recording plane-store paths under `exp` (tiny_experiment has 12)."""
    return sorted((exp / "recording").glob("t*/*/fov_0_z*.ome.zarr"))


def test_get_returns_same_handle_on_repeat(tiny_experiment):
    cache = HandleCache(default_context())
    st = _recording_stores(tiny_experiment)[0]
    # repeated get of the same path returns the identical handle, not a fresh open
    assert cache.get(st) is cache.get(st)
    assert len(cache) == 1


def test_get_reads_correct_data_through_cache(tiny_experiment):
    cache = HandleCache(default_context())
    st = _recording_stores(tiny_experiment)[0]
    arr = cache.get(st)
    assert not isinstance(arr, np.ndarray)             # a lazy ts array, not pixels
    plane = arr[0, 0, 0].read().result()               # explicit per-plane read
    assert plane.shape == (64, 64) and plane.dtype == np.uint16
    # the cached handle reads exactly what a direct open would
    direct = open_array(st, default_context())[0, 0, 0].read().result()
    np.testing.assert_array_equal(plane, direct)


def test_exceeding_max_open_bounds_handle_count(tiny_experiment):
    stores = _recording_stores(tiny_experiment)
    assert len(stores) >= 5                             # fixture sanity
    cache = HandleCache(default_context(), max_open=3)
    for st in stores[:5]:                               # open 5 distinct > max_open
        cache.get(st)
    assert len(cache) <= 3                              # LRU keeps handles bounded


def test_lru_evicts_least_recently_used(tiny_experiment):
    s0, s1, s2, s3 = _recording_stores(tiny_experiment)[:4]
    cache = HandleCache(default_context(), max_open=3)
    h0, _h1, h2 = cache.get(s0), cache.get(s1), cache.get(s2)
    cache.get(s3)                                       # s0 is the LRU -> evicted
    assert len(cache) == 3
    assert cache.get(s0) is not h0                      # s0 was dropped -> fresh open
    assert cache.get(s2) is h2                          # s2 still cached -> same handle


def test_recent_use_protects_a_handle_from_eviction(tiny_experiment):
    s0, s1, s2, s3 = _recording_stores(tiny_experiment)[:4]
    cache = HandleCache(default_context(), max_open=3)
    h0 = cache.get(s0)
    cache.get(s1)
    cache.get(s2)
    assert cache.get(s0) is h0                          # touch s0 -> now most-recent
    cache.get(s3)                                       # LRU is now s1 -> evicted
    assert cache.get(s0) is h0                          # s0 survived (recently used)
