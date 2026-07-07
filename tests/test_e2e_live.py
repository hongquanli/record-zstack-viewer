"""Task 6.2 — end-to-end live-growth test.

Build a `MainWindow` over a partial experiment, then grow that experiment on
disk one pending plane per step (via `generate.grow_experiment`) while driving
the window's own `LiveEngine.tick()` (NOT `start()`) after each step — exactly
the UI-thread wiring the shell uses. This exercises the whole live pipeline end
to end and asserts:

- coverage rises (1/2 -> 2/2);
- a previously `in_progress` FOV becomes `complete` (navigator hue amber -> blue);
- the DISPLAYED FOV refreshes via the data-only path — the SAME embedded
  `ArrayViewer` and display model are kept (`set_datasets` is never called on the
  live path), so the user's scrub index and contrast survive;
- the freshly-written planes surface as real pixels (not fill).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NDV_SYNCHRONOUS", "1")   # ndv resolves data requests inline

import threading

import platformdirs
import pytest

from record_zstack_viewer.live.engine import LiveEngine
from record_zstack_viewer.ui.app import MainWindow
from record_zstack_viewer.ui.plate_navigator import STATUS_FILL
from tests.fixtures.generate import generate_experiment, grow_experiment


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


@pytest.fixture(autouse=True)
def no_live_start(monkeypatch):
    """The app must NEVER call `LiveEngine.start()`; record any call to assert on."""
    calls: list = []
    monkeypatch.setattr(LiveEngine, "start", lambda self: calls.append(self))
    return calls


@pytest.fixture
def partial_experiment(tmp_path):
    """A partial run with NO events.jsonl (-> the LiveEngine poll path): A1/0 is
    complete, B2/0 is half-written (3/6 recording frames; only z0 of a 4-plane
    z-stack) so `grow_experiment` can fill it in over a few steps."""
    root = generate_experiment(
        tmp_path / "exp", wells=["A1", "B2"], fovs_per_well=1, nt=1,
        recording_frames=6, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=4,
        rich=False, complete=False)
    assert not (root / "events.jsonl").exists()          # precondition: poll path
    return root


def test_e2e_live_growth_refreshes_data_only(
        qapp, partial_experiment, monkeypatch, no_live_start):
    root = partial_experiment
    win = MainWindow(root)
    win.navigator.fov_selected.emit("B2", 0)             # display the half-written FOV
    assert win.displayed == ("B2", 0) and win.viewer.mode == "recording"

    win._live.tick()                                     # poll path: baseline snapshot

    # baseline: B2 in_progress (amber), coverage 1/2, the last frame reads as fill (0)
    assert (win.navigator._cells["B2"].body.brush().color().name()
            == STATUS_FILL["in_progress"])
    assert "1/2" in win.coverage_label.text()
    assert int(win.viewer.ndv_viewer.data.read_plane(
        {"acq_t": 0, "frame": 5, "z": 0}).max()) == 0

    # remember the embedded viewer + display model, place a user scrub, capture contrast
    viewer_before = win.viewer.ndv_viewer
    dm_before = viewer_before.display_model
    dm_before.current_index.update({"frame": 2})         # the user's scrub position
    clims_before = (float(dm_before.default_lut.clims.min),
                    float(dm_before.default_lut.clims.max))

    # spy the two swap paths AFTER the initial open
    sd_calls, ue_calls = [], []
    orig_sd, orig_ue = win.viewer.set_datasets, win.viewer.update_extent
    monkeypatch.setattr(win.viewer, "set_datasets",
                        lambda *a, **k: (sd_calls.append(a), orig_sd(*a, **k))[1])
    monkeypatch.setattr(win.viewer, "update_extent",
                        lambda s: (ue_calls.append(s), orig_ue(s))[1])

    # grow one pending plane per step; the UI-thread tick follows each step
    grow_frames = [3, 4, 5]                              # B2's pending recording frames
    for step, frame in enumerate(grow_frames):
        assert grow_experiment(root, step) == {("B2", 0)}   # only the partial FOV grows
        win._live.tick()

        # the freshly-written frame is now visible through the re-opened source ...
        assert int(win.viewer.ndv_viewer.data.read_plane(
            {"acq_t": 0, "frame": frame, "z": 0}).max()) == 1000 * (step + 1)
        # ... via the DATA-ONLY path: same embedded viewer + display model, no rebuild
        assert win.viewer.ndv_viewer is viewer_before
        assert win.viewer.ndv_viewer.display_model is dm_before

    # coverage rose and B2 went in_progress -> complete, all without a full rebuild
    assert sd_calls == [] and len(ue_calls) == len(grow_frames)
    assert "2/2" in win.coverage_label.text()
    assert (win.navigator._cells["B2"].body.brush().color().name()
            == STATUS_FILL["complete"])

    # the user's scrub index and contrast survived every data-only swap
    assert dm_before.current_index["frame"] == 2
    assert (float(dm_before.default_lut.clims.min),
            float(dm_before.default_lut.clims.max)) == clims_before

    # and the growth was followed entirely on the UI thread — never start()
    assert no_live_start == []
    assert not any(t.name == "LiveEngine" for t in threading.enumerate())
    win.close()
