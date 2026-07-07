"""Task 5.2 — `PlateNavigator`: hue-coded plate map + adaptive FOV strip.

Offscreen Qt (env guard before any Qt import). Pure Qt raster — the navigator
must not import ndv (enforced package-wide by
`test_fov_viewer.test_ndv_confined_to_fov_viewer`); thumbnails come from
`ThumbnailCache` only, so `_isolated_cache` (autouse) redirects
`platformdirs.user_cache_dir` to a tmp dir exactly like the thumbnail tests.

Interaction contract under test:
- single-FOV well click -> `fov_selected(well, its_one_fov)` and NO strip;
- multi-FOV well click -> strip shown (one thumbnail per FOV), no direct emit;
  clicking thumbnail k -> `fov_selected(well, k)`;
- `set_status` restyles ONE existing cell (no scene rebuild) with the locked
  hue coding (blue/amber/grey) + the red-outline-and-cross aborted cue;
- `update_coverage` drives the k/N header + the "current" well marker.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import platformdirs
import pytest

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.model.experiment import ExperimentModel
from record_zstack_viewer.ui.plate_navigator import (
    ABORTED_OUTLINE, STATUS_FILL, PlateNavigator)


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Computed thumbnails go to a per-test tmp dir, never the real user cache."""
    cache = tmp_path / "user_cache"
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *a, **k: str(cache))
    return cache


@pytest.fixture
def tiny_model(tiny_experiment):
    """Two single-FOV wells (A1, B2), all complete, 96-format region spacing."""
    return ExperimentModel(tiny_experiment, default_context())


@pytest.fixture
def poor_model(tiny_experiment_poor):
    """One well (A1) with 6 FOVs; no sidecars (rich=False) -> computed thumbs."""
    return ExperimentModel(tiny_experiment_poor, default_context())


def _nav(model):
    nav = PlateNavigator()
    nav.set_model(model)
    return nav


def _spy(nav):
    got = []
    nav.fov_selected.connect(lambda w, k: got.append((w, int(k))))
    return got


# ------------------------------- tier 1: plate map ---------------------------

def test_wells_render_full_plate_grid(qapp, tiny_model):
    nav = _nav(tiny_model)
    assert tiny_model.plate_format() == "96"          # fixture region spacing
    assert len(nav._cells) == 96                      # full 8x12 grid drawn
    assert {"A1", "B2"} <= set(nav._cells)
    # present wells are hue-coded by status (fixture is finalized -> blue)
    for well in ("A1", "B2"):
        assert nav._cells[well].body.brush().color().name() == STATUS_FILL["complete"]
    # an absent grid position carries no status fill
    from qtpy.QtCore import Qt
    assert nav._cells["A2"].body.brush().style() == Qt.NoBrush


def test_unknown_plate_format_grid_covers_present_wells(qapp, poor_model):
    nav = _nav(poor_model)
    assert poor_model.plate_format() == "unknown"     # single region -> no inference
    assert set(nav._cells) == {"A1"}                  # grid sized to present wells
    assert nav._cells["A1"].body.brush().color().name() == STATUS_FILL["complete"]


def test_single_fov_well_click_emits_directly_no_strip(qapp, tiny_model):
    nav = _nav(tiny_model)
    got = _spy(nav)
    nav.select_well("A1")                             # the click handler's path
    assert got == [("A1", 0)]
    assert not nav._strip.isVisibleTo(nav)            # strip never shown
    assert nav._strip.count() == 0


def test_mouse_click_on_well_cell_selects(qapp, tiny_model):
    from qtpy.QtCore import Qt
    from qtpy.QtTest import QTest
    nav = _nav(tiny_model)
    nav.resize(640, 480)
    nav.show()
    qapp.processEvents()
    got = _spy(nav)
    center = nav._cells["B2"].body.sceneBoundingRect().center()
    QTest.mouseClick(nav._view.viewport(), Qt.LeftButton, Qt.NoModifier,
                     nav._view.mapFromScene(center))
    assert got == [("B2", 0)]
    nav.close()


def test_selecting_absent_or_unknown_well_is_noop(qapp, tiny_model):
    nav = _nav(tiny_model)
    got = _spy(nav)
    nav.select_well("A2")                             # grid cell with no data
    nav.select_well("H12")
    nav.select_well("not-a-well")
    assert got == []
    assert not nav._strip.isVisibleTo(nav)


def test_zoom_scales_the_view(qapp, tiny_model):
    nav = _nav(tiny_model)
    s0 = nav._view.transform().m11()
    nav._view.zoom(2.0)
    assert nav._view.transform().m11() == pytest.approx(s0 * 2.0)
    nav._view.zoom(0.25)
    assert nav._view.transform().m11() == pytest.approx(s0 * 0.5)


def test_tooltip_carries_id_status_coverage(qapp, tiny_model):
    nav = _nav(tiny_model)
    tip = nav._cells["A1"].body.toolTip()
    assert "A1" in tip and "complete" in tip and "1/1" in tip
    nav.set_status("A1", 0, "in_progress")            # live update refreshes it
    tip = nav._cells["A1"].body.toolTip()
    assert "in_progress" in tip and "0/1" in tip


# --------------------------- tier 2: adaptive FOV strip ----------------------

def test_multi_fov_well_shows_strip_thumb_click_emits(qapp, poor_model):
    nav = _nav(poor_model)
    got = _spy(nav)
    nav.select_well("A1")
    assert got == []                                  # no direct emit for multi-FOV
    assert nav._strip.isVisibleTo(nav)
    assert nav._strip.count() == 6                    # one thumbnail per FOV
    assert all(not nav._strip.item(i).icon().isNull() for i in range(6))
    nav._strip.itemClicked.emit(nav._strip.item(3))   # user clicks the 4th thumb
    assert got == [("A1", 3)]


# ----------------------------------- live ------------------------------------

def test_set_status_restyles_cell_incrementally(qapp, tiny_model):
    nav = _nav(tiny_model)
    body = nav._cells["A1"].body
    nav.set_status("A1", 0, "in_progress")
    assert nav._cells["A1"].body is body              # same item: no rebuild
    assert body.brush().color().name() == STATUS_FILL["in_progress"]
    assert not nav._cells["A1"].cross.isVisible()
    nav.set_status("A1", 0, "aborted")
    assert body.brush().color().name() == STATUS_FILL["aborted"]
    assert body.pen().color().name() == ABORTED_OUTLINE
    assert nav._cells["A1"].cross.isVisible()         # colorblind-safe shape cue
    nav.set_status("A1", 0, "empty")
    assert body.brush().color().name() == STATUS_FILL["empty"]
    assert not nav._cells["A1"].cross.isVisible()
    nav.set_status("A1", 0, "complete")
    assert body.brush().color().name() == STATUS_FILL["complete"]
    with pytest.raises(ValueError):
        nav.set_status("A1", 0, "bogus")


def test_well_hue_aggregates_over_its_fovs(qapp, poor_model):
    nav = _nav(poor_model)
    body = nav._cells["A1"].body
    assert body.brush().color().name() == STATUS_FILL["complete"]
    nav.set_status("A1", 2, "in_progress")            # 5 complete + 1 running
    assert body.brush().color().name() == STATUS_FILL["in_progress"]
    nav.set_status("A1", 4, "aborted")                # any abort trumps
    assert body.brush().color().name() == STATUS_FILL["aborted"]
    assert nav._cells["A1"].cross.isVisible()
    nav.set_status("A1", 4, "complete")
    nav.set_status("A1", 2, "complete")               # back to all-complete
    assert body.brush().color().name() == STATUS_FILL["complete"]
    assert not nav._cells["A1"].cross.isVisible()


def test_update_coverage_header_and_current_marker(qapp, tiny_model):
    nav = _nav(tiny_model)
    assert "2/2" in nav._coverage_label.text()        # seeded from the model
    nav.update_coverage(3, 12, ("B2", 0))
    text = nav._coverage_label.text()
    assert "3/12" in text and "B2" in text
    assert nav._current_ring.isVisible()              # marks the acquiring well
    nav.update_coverage(12, 12, None)
    assert "12/12" in nav._coverage_label.text()
    assert not nav._current_ring.isVisible()
