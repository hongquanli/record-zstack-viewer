"""Task 5.1 — `FovSourceDataWrapper` + `FovViewer` (offscreen Qt/ndv).

Env guards run before any Qt/ndv import: offscreen QPA for headless CI, and
NDV_SYNCHRONOUS so ndv's data requests resolve inline (deterministic asserts).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NDV_SYNCHRONOUS", "1")

import re
from pathlib import Path

import numpy as np
import pytest

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.reader.adapter import FovDatasets, open_fov
from record_zstack_viewer.reader.metadata import FovMetadata
from record_zstack_viewer.ui.fov_viewer import (
    FovSourceDataWrapper, FovViewer, _auto_clims)

FULL_UINT16 = (0.0, 65535.0)  # the fixture's omero.window — clamp bounds, not a display default


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def ds(tiny_experiment):
    return open_fov(tiny_experiment, "A1", 0, default_context())


class FakeFovSource:
    """Minimal in-memory FovSource that counts `read_plane` calls.

    It has NO bulk-read path at all, so anything the wrapper materializes can
    only have come through per-plane reads.
    """

    dims = ("acq_t", "c", "z", "y", "x")

    def __init__(self, shape=(2, 2, 4, 8, 8)):
        self.shape = shape
        self.dtype = np.dtype("uint16")
        self.metadata = FovMetadata(
            pixel_size_um=1.0, z_step_um=1.0, time_increment_s=1.0, channels=[],
            acquisition_complete=True, aborted=False, n_planes=1)
        self.calls: list[dict] = []

    def read_plane(self, index):
        self.calls.append(dict(index))
        code = index["acq_t"] * 100 + index["c"] * 10 + index["z"]
        return np.full(self.shape[-2:], code, dtype=self.dtype)

    def read_tile(self, index, y0, y1, x0, x1):
        return self.read_plane(index)[y0:y1, x0:x1]

    def filled_extent(self, axis):
        return self.shape[self.dims.index(axis)]

    def is_extent_exact(self):
        return True


# --------------------------- FovSourceDataWrapper ---------------------------

def test_sizes_named_dims_zstack(ds):
    w = FovSourceDataWrapper(ds.zstack)
    assert dict(w.sizes()) == {"acq_t": 2, "c": 2, "z": 4, "y": 64, "x": 64}
    assert tuple(w.sizes().values()) == (2, 2, 4, 64, 64)


def test_sizes_named_dims_recording(ds):
    w = FovSourceDataWrapper(ds.recording)
    assert dict(w.sizes()) == {"acq_t": 2, "frame": 5, "z": 3, "y": 64, "x": 64}


def test_isel_named_keys_matches_read_plane(ds):
    w = FovSourceDataWrapper(ds.zstack)
    plane = w.isel({"acq_t": 0, "c": 0, "z": 0})
    assert plane.shape == (64, 64)
    np.testing.assert_array_equal(
        plane, ds.zstack.read_plane({"acq_t": 0, "c": 0, "z": 0}))
    # discriminates by (c, z): a different channel/plane is different pixels
    other = ds.zstack.read_plane({"acq_t": 0, "c": 1, "z": 1})
    assert not np.array_equal(plane, other)
    np.testing.assert_array_equal(w.isel({"acq_t": 0, "c": 1, "z": 1}), other)


def test_isel_ndv_style_int_keys_and_slices(ds):
    # exactly what ndv's current_slice_requests sends in composite mode:
    # int axis keys, width-1 slices for stepped axes, full slices for
    # the channel axis and the visible (y, x) axes — singleton dims kept.
    w = FovSourceDataWrapper(ds.zstack)
    out = w.isel({0: slice(0, 1), 1: slice(None), 2: slice(1, 2),
                  3: slice(None), 4: slice(None)})
    assert out.shape == (1, 2, 1, 64, 64)
    for c in range(2):
        np.testing.assert_array_equal(
            out[0, c, 0], ds.zstack.read_plane({"acq_t": 0, "c": c, "z": 1}))


def test_isel_reads_single_planes_only():
    src = FakeFovSource()
    w = FovSourceDataWrapper(src)
    w.isel({"acq_t": 1, "c": 1, "z": 2})
    assert src.calls == [{"acq_t": 1, "c": 1, "z": 2}]  # exactly ONE plane read
    src.calls.clear()
    out = w.isel({0: slice(1, 2), 1: slice(1, 2), 2: slice(2, 3),
                  3: slice(None), 4: slice(None)})      # mono-style ndv request
    assert out.shape == (1, 1, 1, 8, 8)
    assert src.calls == [{"acq_t": 1, "c": 1, "z": 2}]
    src.calls.clear()
    w.isel({0: slice(0, 1), 1: slice(None), 2: slice(0, 1),
            3: slice(None), 4: slice(None)})            # composite: C visible planes
    assert len(src.calls) == 2                          # one read per channel — no more


def test_isel_plane_budget_guard_raises_on_wide_requests(ds):
    """Partial/empty indexes must RAISE, not bulk-read (O(1)-memory constraint).

    An empty index would full-slice every non-spatial axis (16 planes on the
    tiny zstack); only the composite channel axis may be wide.
    """
    w = FovSourceDataWrapper(ds.zstack)
    with pytest.raises(ValueError, match=r"acq_t.*width > 1"):
        w.isel({})                       # would be 2*2*4 = 16 plane reads
    with pytest.raises(ValueError, match=r"'z'"):
        w.isel({"acq_t": 0})             # c may be wide; z (width 4) may not
    # the legitimate ndv shapes still succeed: a single plane...
    assert w.isel({"acq_t": 0, "c": 0, "z": 0}).shape == (64, 64)
    # ...and the composite request (channel axis wide, all others width-1)
    out = w.isel({0: slice(0, 1), 1: slice(None), 2: slice(1, 2),
                  3: slice(None), 4: slice(None)})
    assert out.shape == (1, 2, 1, 64, 64)
    # recording layout has no channel axis at all -> nothing may be wide
    wr = FovSourceDataWrapper(ds.recording)
    with pytest.raises(ValueError, match=r"frame"):
        wr.isel({})
    assert wr.isel({"acq_t": 0, "frame": 0, "z": 0}).shape == (64, 64)
    # the guard fires BEFORE any read: zero read_plane calls on the fake
    src = FakeFovSource()
    with pytest.raises(ValueError):
        FovSourceDataWrapper(src).isel({})
    assert src.calls == []


def test_supports_and_create(ds):
    from ndv.models._data_wrapper import DataWrapper
    assert FovSourceDataWrapper.supports(ds.zstack)
    assert FovSourceDataWrapper.supports(ds.recording)
    assert not FovSourceDataWrapper.supports(np.zeros((2, 2)))
    # registered subclass: ndv's own factory picks our wrapper for a FovSource
    assert isinstance(DataWrapper.create(ds.zstack), FovSourceDataWrapper)


# ------------------------------- auto-contrast ------------------------------

def test_auto_clims_percentile_clamped_nondegenerate():
    plane = np.linspace(0, 4000, 64 * 64).astype(np.uint16).reshape(64, 64)
    lo, hi = _auto_clims(plane, FULL_UINT16)
    exp = np.percentile(plane, (1.0, 99.9))
    assert (lo, hi) == pytest.approx((float(exp[0]), float(exp[1])))
    assert lo < hi and (lo, hi) != FULL_UINT16
    # the omero window is only a CLAMP: limits are pulled inside it
    assert _auto_clims(plane, (500.0, 1000.0)) == (500.0, 1000.0)
    # constant plane still yields a non-degenerate range
    lo, hi = _auto_clims(np.full((8, 8), 7, dtype=np.uint16), FULL_UINT16)
    assert hi > lo


# --------------------------------- FovViewer --------------------------------

def test_viewer_zstack_composite_auto_contrast(qapp, ds):
    v = FovViewer()
    v.set_datasets(ds, "zstack")
    assert v.mode == "zstack"
    dm = v.ndv_viewer.display_model
    assert dm.channel_mode.value == "composite"
    nz = ds.zstack.shape[ds.zstack.dims.index("z")]
    for ci in range(2):
        lut = dm.luts[ci]
        lo, hi = float(lut.clims.min), float(lut.clims.max)
        exp = np.percentile(
            ds.zstack.read_plane({"acq_t": 0, "c": ci, "z": nz // 2}), (1.0, 99.9))
        assert (lo, hi) == pytest.approx((float(exp[0]), float(exp[1])))
        assert lo < hi
        assert (lo, hi) != FULL_UINT16          # NOT the full dtype/window range
        assert lut.clim_bounds == FULL_UINT16   # window used only as clamp bounds


def test_viewer_recording_grayscale_auto_contrast(qapp, ds):
    v = FovViewer()
    v.set_datasets(ds, "recording")
    dm = v.ndv_viewer.display_model
    assert dm.channel_mode.value == "grayscale"
    assert dict(v.ndv_viewer.data_wrapper.sizes())["frame"] == 5
    assert "frame" in dm.current_index          # frame scrub axis is stepped
    nz = ds.recording.shape[ds.recording.dims.index("z")]
    lo, hi = float(dm.default_lut.clims.min), float(dm.default_lut.clims.max)
    exp = np.percentile(
        ds.recording.read_plane({"acq_t": 0, "frame": 0, "z": nz // 2}), (1.0, 99.9))
    assert (lo, hi) == pytest.approx((float(exp[0]), float(exp[1])))
    assert lo < hi and (lo, hi) != FULL_UINT16


def test_viewer_set_timepoint_clamped(qapp, ds):
    v = FovViewer()
    v.set_datasets(ds, "zstack")
    v.set_timepoint(1)
    assert v.ndv_viewer.display_model.current_index["acq_t"] == 1
    v.set_timepoint(99)                          # clamped to the acq_t extent
    assert v.ndv_viewer.display_model.current_index["acq_t"] == 1
    v.set_timepoint(-5)
    assert v.ndv_viewer.display_model.current_index["acq_t"] == 0


def test_viewer_mode_switch_rebuilds_and_validates(qapp, ds):
    v = FovViewer()
    v.set_datasets(ds, "zstack")
    first = v.ndv_viewer
    v.set_datasets(ds, "recording")              # dims signature changed -> rebuild
    assert v.ndv_viewer is not first
    assert v.mode == "recording"
    assert tuple(v.ndv_viewer.data_wrapper.dims) == ("acq_t", "frame", "z", "y", "x")
    with pytest.raises(ValueError):
        v.set_datasets(FovDatasets(recording=None, zstack=ds.zstack), "recording")
    with pytest.raises(ValueError):
        v.set_datasets(ds, "bogus")


def test_viewer_same_dims_swaps_in_place(qapp, tiny_experiment):
    ctx = default_context()
    a = open_fov(tiny_experiment, "A1", 0, ctx)
    b = open_fov(tiny_experiment, "B2", 0, ctx)
    v = FovViewer()
    v.set_datasets(a, "zstack")
    first = v.ndv_viewer
    v.set_datasets(b, "zstack")                  # same dims -> public in-place swap
    assert v.ndv_viewer is first
    assert v.ndv_viewer.data is b.zstack


def test_viewer_auto_contrast_recomputes(qapp, ds):
    from ndv.models import ClimsManual
    v = FovViewer()
    v.set_datasets(ds, "zstack")
    dm = v.ndv_viewer.display_model
    orig = (float(dm.luts[0].clims.min), float(dm.luts[0].clims.max))
    dm.luts[0].clims = ClimsManual(min=0.0, max=65535.0)   # user blows the range
    v.auto_contrast()
    now = (float(dm.luts[0].clims.min), float(dm.luts[0].clims.max))
    assert now == pytest.approx(orig)


def test_ndv_confined_to_fov_viewer():
    """Global Constraint: ndv is imported ONLY in ui/fov_viewer.py."""
    import record_zstack_viewer
    pkg = Path(record_zstack_viewer.__file__).parent
    offenders = [
        str(p) for p in pkg.rglob("*.py") if p.name != "fov_viewer.py"
        and re.search(r"^\s*(import ndv\b|from ndv\b)", p.read_text(), re.M)]
    assert offenders == []
