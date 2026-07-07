"""Task 5.3 — `MainWindow` app shell + `__main__` entry (offscreen Qt/ndv).

Locks the shell wiring, not the parts (5.1/5.2/4.2 own those):

- **Layout A** — docked `PlateNavigator` left, `FovViewer` central, live coverage
  header — built over ONE shared `default_context()` and ONE `ThumbnailCache`
  (navigator + model + live engine all draw from the same bounded caches).
- `navigator.fov_selected` -> remember `(well, fov)`, `open_fov` -> viewer.
- The Recording | Z-stack toggle lives in the shell and switches the viewer;
  the shared `acq_t` timepoint survives the mode switch.
- **Live is a UI-thread QTimer driving `LiveEngine.tick()`** — the app must
  NEVER call `LiveEngine.start()` (its background thread races the
  non-thread-safe `model.refresh()` against UI-thread reads). Tests drive
  `tick()` directly. A change to the DISPLAYED FOV takes the data-only path
  (`viewer.update_extent`: display model, index, contrast all preserved),
  never a full `set_datasets` rebuild.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NDV_SYNCHRONOUS", "1")   # ndv resolves data requests inline

import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import platformdirs
import pytest
import tensorstore as ts

from record_zstack_viewer.live.engine import LiveEngine
from record_zstack_viewer.ui.app import MainWindow
from record_zstack_viewer.ui.plate_navigator import STATUS_FILL, PlateNavigator
from tests.fixtures.generate import generate_experiment


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
    """Record any `LiveEngine.start()` call — the app must never make one."""
    calls: list = []
    monkeypatch.setattr(LiveEngine, "start", lambda self: calls.append(self))
    return calls


@pytest.fixture
def live_experiment(tmp_path):
    """A partial run with NO events.jsonl -> the LiveEngine poll path; the last
    FOV (B2/0 at t0) is half-written so a test can grow it on disk."""
    root = generate_experiment(
        tmp_path / "exp", wells=["A1", "B2"], fovs_per_well=1, nt=1,
        recording_frames=6, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=4,
        rich=False, complete=False)
    assert not (root / "events.jsonl").exists()          # precondition: poll path
    return root


# --- on-disk growth helpers (what a live acquisition does) --------------------

def _finalize(store: Path, **squid_updates) -> None:
    zj = store / "zarr.json"
    meta = json.loads(zj.read_text())
    meta["attributes"]["_squid"].update(squid_updates)
    zj.write_text(json.dumps(meta, indent=2))


def _grow_recording(root: Path, well: str, fov: int, *, value: int) -> None:
    """Write the pending frames of (well, fov)'s t0 recording store and finalize."""
    store = root / "recording" / "t0" / well / f"fov_{fov}.ome.zarr"
    arr = ts.open({"driver": "zarr3",
                   "kvstore": {"driver": "file", "path": str(store)}}).result()
    n, (h, w) = arr.shape[0], arr.shape[-2:]
    for f in range(n // 2, n):                           # generator wrote [0, n//2)
        arr[f, 0, 0].write(np.full((h, w), value, dtype="uint16")).result()
    _finalize(store, captured_frames=n, acquisition_complete=True)


def _grow_zstack(root: Path, well: str, fov: int, *, value: int) -> None:
    """Fill (well, fov)'s pending z planes at t0 and finalize the z-stack group."""
    grp = root / "zarr" / well / f"fov_{fov}.ome.zarr"
    a = ts.open({"driver": "zarr3",
                 "kvstore": {"driver": "file", "path": str(grp / "0")}}).result()
    _nt, nc, nz, h, w = a.shape
    for c in range(nc):
        for z in range(1, nz):                           # z=0 was already written
            a[0, c, z].write(np.full((h, w), value, dtype="uint16")).result()
    _finalize(grp, acquisition_complete=True)


# ------------------------------ shell construction ----------------------------

def test_mainwindow_constructs_layout_a_and_selects(qapp, tiny_experiment, no_live_start):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget

    win = MainWindow(tiny_experiment)

    # Layout A: navigator docked LEFT, viewer inside the central widget
    assert isinstance(win.navigator, PlateNavigator)
    docks = win.findChildren(QDockWidget)
    assert len(docks) == 1 and docks[0].widget() is win.navigator
    assert win.dockWidgetArea(docks[0]) == Qt.LeftDockWidgetArea
    assert win.centralWidget().isAncestorOf(win.viewer)

    # ONE context / ONE thumbnail cache shared by navigator + model + live engine
    assert win.navigator._thumbs is win._thumbnails
    assert win._live._thumbnail_cache is win._thumbnails
    assert win._thumbnails._context is win._context
    assert win._model._context is win._context

    # live coverage header seeded from the model (2 wells x 1 FOV, all complete)
    assert "2/2" in win.coverage_label.text()

    # live drive: an ACTIVE ~1s UI-thread QTimer -> tick(); NEVER start()
    assert win._live_timer.isActive()
    assert 200 <= win._live_timer.interval() <= 5000
    assert no_live_start == []
    assert not any(t.name == "LiveEngine" for t in threading.enumerate())

    # selecting a FOV sets the viewer's datasets (default mode: recording)
    assert win.displayed is None and win.viewer.ndv_viewer is None
    win.navigator.fov_selected.emit("A1", 0)
    assert win.displayed == ("A1", 0)
    assert win.viewer.mode == "recording"
    assert win.viewer.ndv_viewer.data is win._datasets.recording

    win.navigator.fov_selected.emit("B2", 0)             # a different FOV re-opens
    assert win.displayed == ("B2", 0)
    assert win.viewer.ndv_viewer.data is win._datasets.recording

    win.close()
    assert not win._live_timer.isActive()                # close stops the live timer


# --------------------------------- mode toggle --------------------------------

def test_mode_toggle_switches_viewer_and_shares_timepoint(qapp, tiny_experiment):
    win = MainWindow(tiny_experiment)
    win.set_mode("zstack")                               # pre-selection: just remembered
    assert win.viewer.mode is None
    win.navigator.fov_selected.emit("A1", 0)
    assert win.viewer.mode == "zstack"                   # opened in the chosen mode
    assert win.mode_buttons["zstack"].isChecked()
    assert not win.mode_buttons["recording"].isChecked()

    win.viewer.set_timepoint(1)                          # user steps the shared acq_t
    win.mode_buttons["recording"].click()                # the UI toggle path
    assert win.viewer.mode == "recording"
    assert win.mode_buttons["recording"].isChecked()
    assert not win.mode_buttons["zstack"].isChecked()
    assert win.viewer.timepoint == 1                     # acq_t shared across modes

    win.mode_buttons["zstack"].click()
    assert win.viewer.mode == "zstack"
    assert win.viewer.timepoint == 1

    with pytest.raises(ValueError):
        win.set_mode("bogus")
    win.close()


# ------------------------------------ live ------------------------------------

def test_live_tick_updates_navigator_and_refreshes_displayed_data_only(
        qapp, live_experiment, monkeypatch, no_live_start):
    win = MainWindow(live_experiment)
    win.navigator.fov_selected.emit("B2", 0)             # display the half-written FOV
    assert win.displayed == ("B2", 0) and win.viewer.mode == "recording"

    win._live.tick()                                     # poll path: baseline snapshot

    # pre-grow: B2 amber; the pending frame still reads as fill (0)
    assert (win.navigator._cells["B2"].body.brush().color().name()
            == STATUS_FILL["in_progress"])
    assert "1/2" in win.coverage_label.text()
    pending = {"acq_t": 0, "frame": 5, "z": 0}
    assert int(win.viewer.ndv_viewer.data.read_plane(pending).max()) == 0

    viewer_before = win.viewer.ndv_viewer
    dm_before = viewer_before.display_model
    dm_before.current_index.update({"frame": 2})         # the user's scrub position
    clims_before = (float(dm_before.default_lut.clims.min),
                    float(dm_before.default_lut.clims.max))

    sd_calls, ue_calls = [], []                          # spy AFTER the initial open
    orig_sd, orig_ue = win.viewer.set_datasets, win.viewer.update_extent
    monkeypatch.setattr(win.viewer, "set_datasets",
                        lambda *a, **k: (sd_calls.append(a), orig_sd(*a, **k))[1])
    monkeypatch.setattr(win.viewer, "update_extent",
                        lambda s: (ue_calls.append(s), orig_ue(s))[1])

    _grow_recording(live_experiment, "B2", 0, value=1234)
    _grow_zstack(live_experiment, "B2", 0, value=4242)
    win._live.tick()                                     # UI-thread tick sees the growth

    # navigator: the changed cell restyled + both coverage headers advanced
    assert (win.navigator._cells["B2"].body.brush().color().name()
            == STATUS_FILL["complete"])
    assert "2/2" in win.navigator._coverage_label.text()
    assert "2/2" in win.coverage_label.text()

    # displayed FOV refreshed via the DATA-ONLY path — never a full rebuild
    assert len(ue_calls) == 1 and sd_calls == []
    assert win.viewer.ndv_viewer is viewer_before        # same embedded viewer
    assert win.viewer.ndv_viewer.display_model is dm_before   # display model kept
    assert dm_before.current_index["frame"] == 2         # user's index preserved
    assert (float(dm_before.default_lut.clims.min),
            float(dm_before.default_lut.clims.max)) == clims_before  # contrast kept

    # ... and the grown pixels are really visible through the re-opened source
    assert int(win.viewer.ndv_viewer.data.read_plane(pending).max()) == 1234

    # the app never used the background-thread driver
    assert no_live_start == []
    assert not any(t.name == "LiveEngine" for t in threading.enumerate())
    win.close()


def test_live_tick_full_rebuild_fallback_when_data_swap_fails(
        qapp, live_experiment, monkeypatch):
    """If the data-only swap raises, the shell falls back to set_datasets."""
    win = MainWindow(live_experiment)
    win.navigator.fov_selected.emit("B2", 0)
    win._live.tick()                                     # baseline

    def boom(source):
        raise RuntimeError("simulated swap failure")

    monkeypatch.setattr(win.viewer, "update_extent", boom)
    sd_calls = []
    orig_sd = win.viewer.set_datasets
    monkeypatch.setattr(win.viewer, "set_datasets",
                        lambda *a, **k: (sd_calls.append(a), orig_sd(*a, **k))[1])

    _grow_recording(live_experiment, "B2", 0, value=1234)
    win._live.tick()

    assert len(sd_calls) == 1                            # rebuilt instead of crashed
    assert win.viewer.mode == "recording"
    pending = {"acq_t": 0, "frame": 5, "z": 0}
    assert int(win.viewer.ndv_viewer.data.read_plane(pending).max()) == 1234
    win.close()


# --------------------------------- CLI entry ----------------------------------

def test_cli_help_smoke():
    """`python -m record_zstack_viewer --help` exits 0 before any Qt import."""
    out = subprocess.run(
        [sys.executable, "-m", "record_zstack_viewer", "--help"],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert out.returncode == 0
    assert "experiment root" in out.stdout


def test_cli_rejects_missing_root(tmp_path):
    from record_zstack_viewer.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "does_not_exist")])
    assert exc.value.code == 2
