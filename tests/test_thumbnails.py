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
    from record_zstack_viewer.reader import adapter as adapter_mod
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    first = ThumbnailCache(tiny_experiment_poor).get("A1", 0)   # computes + persists
    assert list(_isolated_cache.rglob("*.png"))

    def _boom(*a, **k):
        raise AssertionError("must serve from disk cache, not recompute")

    # `_compute` imports `open_fov` from `reader.adapter` lazily, so patch it at the
    # source: a fresh instance must serve the disk copy and never reach open_fov.
    monkeypatch.setattr(adapter_mod, "open_fov", _boom)         # a fresh instance ...
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


def test_invalidate_drops_ram_entry_so_next_get_reflects_change(tiny_experiment):
    """invalidate(well, fov) drops the RAM-LRU entry so the next get re-reads.

    The RAM cache is keyed by (well, fov) alone (no token), so within a session it
    would keep serving a stale thumbnail after the underlying data changes. The live
    engine calls invalidate on a changed FOV; the next get must reflect the change.
    """
    import imageio.v3 as iio

    from record_zstack_viewer.cache.thumbnails import ThumbnailCache

    tc = ThumbnailCache(tiny_experiment)
    first = tc.get("A1", 0)                          # reads the squid PNG, caches in RAM
    assert first.ndim == 2 and first.dtype.name == "uint8"
    assert tc.get("A1", 0) is first                  # repeat is a RAM hit (same object)

    # the underlying source (here the squid-written thumbnail) changes on disk
    png = tiny_experiment / "thumbnails" / "A1" / "fov_0.png"
    iio.imwrite(png, np.full((32, 32), 200, dtype="uint8"))
    assert tc.get("A1", 0) is first                  # still stale until invalidated

    tc.invalidate("A1", 0)                           # drop the RAM entry
    second = tc.get("A1", 0)                          # RAM miss -> re-read the PNG
    assert second is not first                        # not the stale RAM object
    assert not np.array_equal(second, first)          # reflects the changed data
    assert int(second.max()) == 200                   # the new PNG's value


def test_invalidate_is_a_noop_when_uncached(tiny_experiment):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    tc = ThumbnailCache(tiny_experiment)
    tc.invalidate("A1", 0)                            # nothing cached yet -> no error
    assert ("A1", 0) not in tc._ram


def test_thumbnailcache_importable_from_package_no_cycle():
    """Regression guard for the cache<->reader import cycle.

    `cache/__init__` re-exports `ThumbnailCache` and `reader.adapter` imports
    `cache.memory`; a top-level `open_fov` import in `thumbnails` re-forms a cycle
    that fails when `reader.adapter` is imported first (the normal app path). A warm
    in-process import would hide it, so import that order in a FRESH interpreter and
    require it to succeed.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    code = ("import record_zstack_viewer.reader.adapter\n"
            "from record_zstack_viewer.cache import ThumbnailCache\n"
            "print(ThumbnailCache.__name__)\n")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )
    assert result.returncode == 0, result.stderr
    assert "ThumbnailCache" in result.stdout


def test_passed_context_is_used(tiny_experiment_poor, monkeypatch):
    """A context passed to ThumbnailCache is threaded into the compute (so a caller
    can share the app's single bounded cache_pool), and no fresh default_context is
    created when one is supplied."""
    from record_zstack_viewer.reader import adapter as adapter_mod
    from record_zstack_viewer.cache import thumbnails as thumbs_mod
    from record_zstack_viewer.cache.memory import default_context
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache

    shared = default_context()
    seen = {}
    real_open_fov = adapter_mod.open_fov

    def _spy(root, well, fov, context):
        seen["context"] = context
        return real_open_fov(root, well, fov, context)

    def _no_default(*a, **k):
        raise AssertionError("must reuse the passed context, not create a default one")

    monkeypatch.setattr(adapter_mod, "open_fov", _spy)
    monkeypatch.setattr(thumbs_mod, "default_context", _no_default)

    ThumbnailCache(tiny_experiment_poor, context=shared).get("A1", 0)
    assert seen["context"] is shared               # the exact object we passed, not a new pool
