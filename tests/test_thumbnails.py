"""Task 3.2 — `ThumbnailCache`: three-tier lookup + the never-write-into-the-
experiment Global Constraint.

`_isolated_cache` (autouse) redirects `platformdirs.user_cache_dir` to a per-test
tmp dir, so no test pollutes the real user cache and every computed thumbnail is
observable at a known location.
"""
import numpy as np
import platformdirs
import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Redirect the user-cache dir to a tmp path for every thumbnail test."""
    cache = tmp_path / "user_cache"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **k: str(cache))
    return cache


def test_thumbnail_prefers_squid_png(tiny_experiment):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    tc = ThumbnailCache(tiny_experiment)
    t = tc.get("A1", 0)
    assert t.ndim == 2 and t.dtype.name == "uint8"


def test_thumbnail_computes_when_absent(tiny_experiment_poor):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    tc = ThumbnailCache(tiny_experiment_poor)
    t = tc.get("A1", 0)                              # no thumbnails/ -> computed
    assert t.ndim == 2 and t.dtype.name == "uint8"
    assert not (tiny_experiment_poor / ".ndviewer_cache").exists()


def test_no_cache_written_into_experiment_folder(tiny_experiment_poor, _isolated_cache):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    exp = tiny_experiment_poor
    before = set(exp.rglob("*"))                     # snapshot the experiment tree
    tc = ThumbnailCache(exp)
    t = tc.get("A1", 0)
    assert t.ndim == 2 and t.dtype.name == "uint8"
    # the computed thumbnail persisted into the redirected user-cache dir ...
    assert list(_isolated_cache.rglob("*.png")), "computed thumbnail must persist to user-cache"
    # ... and NOT anywhere inside the experiment folder.
    assert not (exp / ".ndviewer_cache").exists()
    assert not list(exp.rglob("*.png"))             # poor fixture has no PNGs; we add none
    assert set(exp.rglob("*")) == before            # experiment tree left untouched


def test_disk_cache_reused_across_instances(tiny_experiment_poor, _isolated_cache, monkeypatch):
    from record_zstack_viewer.cache import thumbnails as thumbs_mod
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    first = ThumbnailCache(tiny_experiment_poor).get("A1", 0)   # computes + persists
    assert list(_isolated_cache.rglob("*.png"))

    def _boom(*a, **k):
        raise AssertionError("must serve from disk cache, not recompute")

    monkeypatch.setattr(thumbs_mod, "open_fov", _boom)          # a fresh instance ...
    second = ThumbnailCache(tiny_experiment_poor).get("A1", 0)  # ... reads the disk copy
    np.testing.assert_array_equal(first, second)


def test_ram_cache_returns_same_object(tiny_experiment):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    tc = ThumbnailCache(tiny_experiment)
    assert tc.get("A1", 0) is tc.get("A1", 0)       # second get is a RAM hit


def test_ram_lru_is_bounded(tiny_experiment_poor):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    tc = ThumbnailCache(tiny_experiment_poor, ram_items=2)
    for k in range(4):                              # poor fixture has 6 fovs in A1
        tc.get("A1", k)
    assert len(tc._ram) <= 2                        # LRU keeps the RAM cache bounded
