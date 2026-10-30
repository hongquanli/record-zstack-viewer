"""zarr -> mp4 export: planning, encoding, partial stores, holes, failures, cancel, CLI, dialog."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from qtpy.QtCore import Qt

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.export import mp4
from record_zstack_viewer.export.__main__ import main as cli_main
from record_zstack_viewer.model.experiment import ExperimentModel
from tests.fixtures.generate import generate_experiment


def _decode(path: Path, h: int, w: int) -> np.ndarray:
    """All frames of `path` as a (n, h, w) uint8 array, via ffmpeg rawvideo."""
    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(-1, h, w)


def _tmp_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.tmp")) if folder.exists() else []


@pytest.fixture
def single_plane(tmp_path):
    """2 wells x 1 FOV, Nt=1, one plane: 2 recording stores of 6 frames (64x64)."""
    return generate_experiment(tmp_path / "exp", wells=["A1", "B2"], fovs_per_well=1, nt=1,
                               recording_frames=6, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2)


@pytest.fixture
def model(single_plane):
    return ExperimentModel(single_plane, default_context())


@pytest.fixture
def out(tmp_path):
    return tmp_path / "videos"


# ------------------------------------------------------------------ planning

def test_video_name_adds_axes_only_when_needed():
    assert mp4.video_name("A1", 0, 0, 0, 1, 1, False) == "A1_fov0.mp4"
    assert mp4.video_name("A1", 2, 1, 0, 3, 1, False) == "A1_fov2_t1.mp4"
    assert mp4.video_name("A1", 0, 0, 2, 1, 3, False) == "A1_fov0_z2.mp4"
    assert mp4.video_name("A1", 0, 1, 2, 2, 3, True) == "A1_fov0_t1_z2_partial.mp4"


def test_plan_one_job_per_store_and_skips_up_to_date(model, out):
    jobs = mp4.plan(model, out)
    assert [(j.well, j.out.name) for j in jobs] == [("A1", "A1_fov0.mp4"), ("B2", "B2_fov0.mp4")]
    assert mp4.plan(model, out, wells=["B2"]) == jobs[1:]

    assert mp4.export_job(model, jobs[0]).ok
    assert [j.well for j in mp4.plan(model, out)] == ["B2"]           # A1 is up to date
    assert len(mp4.plan(model, out, overwrite=True)) == 2


def test_plan_multi_plane_time_lapse(tiny_experiment, out):
    model = ExperimentModel(tiny_experiment, default_context())  # 2 wells, nt=2, 3 planes
    names = sorted(j.out.name for j in mp4.plan(model, out))
    assert len(names) == 12
    assert names[0] == "A1_fov0_t0_z0.mp4" and names[-1] == "B2_fov0_t1_z2.mp4"


# ------------------------------------------------------------------ encoding

def test_export_writes_every_frame(model, out):
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(model, job)
    assert result.ok and result.frames == 6 and result.held == 0
    assert job.out.exists() and _tmp_files(out) == []
    assert _decode(job.out, 64, 64).shape == (6, 64, 64)


def test_contrast_is_fixed_for_the_whole_video(model, out):
    """With explicit clims the blob (2000 counts) saturates and the noise (<300)
    stays dark in every frame: no per-frame renormalisation."""
    job = mp4.plan(model, out)[0]
    assert mp4.export_job(model, job, clims=(0.0, 1000.0)).ok
    frames = _decode(job.out, 64, 64)
    blob, corner = frames[:, 26, 26], frames[:, 0, 0]       # blob centre is at 40 % of 64
    assert (blob > 245).all() and (corner < 90).all()


def test_real_time_fps_reaches_ffmpeg(model, out, monkeypatch):
    seen = []
    real = mp4._ffmpeg_command
    monkeypatch.setattr(mp4, "_ffmpeg_command",
                        lambda w, h, fps, o, *a: seen.append(fps) or real(w, h, fps, o, *a))
    job = mp4.plan(model, out)[0]
    assert mp4.export_job(model, job).ok and seen == [15.0]          # fixture: 1/15 s per frame
    assert mp4.export_job(model, job, fps=2.5).ok and seen[-1] == 2.5


def test_default_fps_prefers_effective_fps(single_plane):
    store = single_plane / "recording" / "t0" / "A1" / "fov_0.ome.zarr"
    meta = json.loads((store / "zarr.json").read_text())
    meta["attributes"]["_squid"]["effective_fps"] = 9.5
    (store / "zarr.json").write_text(json.dumps(meta))
    source = ExperimentModel(single_plane, default_context()).open_fov("A1", 0).recording
    assert source.metadata.effective_fps == 9.5 and mp4.default_fps(source) == 9.5


def test_partial_store_exports_captured_frames(tmp_path, out):
    root = generate_experiment(tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
                               recording_frames=6, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2, complete=False)
    model = ExperimentModel(root, default_context())
    assert mp4.plan(model, out) == []                  # run still going: left alone
    (root / ".done").write_text("")                    # aborted run: .done without completion
    (job,) = mp4.plan(model, out)
    assert job.out.name == "A1_fov0_partial.mp4"
    result = mp4.export_job(model, job)
    assert result.ok and result.frames == 3


def test_unwritten_slot_repeats_previous_frame(model, out):
    store = model.open_fov("A1", 0).recording.store_path(0, 0)
    shutil.rmtree(store / "c" / "3")                   # frame 3 was never written
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(model, job)
    assert result.ok and result.held == 1 and result.frames == 6
    frames = _decode(job.out, 64, 64)
    assert np.abs(frames[3].astype(int) - frames[2].astype(int)).mean() < 1.0


def test_odd_frame_size_is_cropped_even(tmp_path, out):
    root = generate_experiment(tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
                               recording_frames=2, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=1, hw=(33, 65))
    model = ExperimentModel(root, default_context())
    job = mp4.plan(model, out)[0]
    assert mp4.export_job(model, job).ok
    assert _decode(job.out, 32, 64).shape == (2, 32, 64)


def test_output_size_even_and_capped():
    assert mp4.output_size(3112, 2084) == (3112, 2084)             # under the cap: untouched
    assert mp4.output_size(6224, 4168) == (4096, 2742)             # 1x1-binned ToupTek, aspect kept
    assert mp4.output_size(6224, 4168, max_side=None) == (6224, 4168)
    assert mp4.output_size(33, 65, max_side=None) == (32, 64)      # odd sides cropped
    assert "-vf" not in mp4._ffmpeg_command(32, 20, 10.0, Path("x.mp4"), max_side=32)
    cmd = mp4._ffmpeg_command(64, 40, 10.0, Path("x.mp4"), max_side=32)
    assert cmd[cmd.index("-vf") + 1] == "scale=32:20"


def test_export_downscales_to_max_side_unless_full_res(model, out):
    job = mp4.plan(model, out)[0]
    assert mp4.export_job(model, job, max_side=32).ok
    assert _decode(job.out, 32, 32).shape == (6, 32, 32)
    assert mp4.export_job(model, job, max_side=None).ok
    assert _decode(job.out, 64, 64).shape == (6, 64, 64)


# ------------------------------------------------------------- failure paths

def test_ffmpeg_failure_is_reported_and_leaves_no_tmp(model, out, monkeypatch):
    monkeypatch.setattr(mp4, "_ffmpeg_command", lambda *a: ["sh", "-c", "echo boom >&2; exit 3"])
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(model, job)
    assert not result.ok and "boom" in result.error
    assert not job.out.exists() and _tmp_files(out) == []


def test_read_error_is_reported_and_leaves_no_tmp(model, out, monkeypatch):
    from record_zstack_viewer.reader.zarr_v3 import RecordingFovSource

    def broken(self, index):
        raise OSError("torn zarr.json")
    monkeypatch.setattr(RecordingFovSource, "read_plane", broken)
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(model, job)
    assert result.error == "torn zarr.json" and _tmp_files(out) == []


def test_cancel_discards_the_file_in_progress_and_stops(model, out):
    jobs = mp4.plan(model, out)
    results = list(mp4.export_all(model, jobs, cancelled=lambda: True))
    assert len(results) == 1 and results[0].error == mp4.CANCELLED
    assert not out.exists() or list(out.iterdir()) == []


# ----------------------------------------------------------------------- CLI

def test_cli_exports_all_then_reports_nothing_to_do(single_plane, out, capsys):
    assert cli_main([str(single_plane), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["A1_fov0.mp4", "B2_fov0.mp4"]
    assert cli_main([str(single_plane), "--out", str(out)]) == 0
    assert "nothing to export" in capsys.readouterr().out


def test_cli_returns_1_when_a_file_fails(single_plane, out, monkeypatch, capsys):
    monkeypatch.setattr(mp4, "_ffmpeg_command", lambda *a: ["sh", "-c", "exit 1"])
    assert cli_main([str(single_plane), "--out", str(out)]) == 1
    assert "FAILED" in capsys.readouterr().err


# -------------------------------------------------------------------- dialog

@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_default_out_dir_is_downloads_per_experiment(tmp_path):
    assert mp4.default_out_dir(tmp_path / "2026-10-30_run") == Path.home() / "Downloads" / "videos-2026-10-30_run"


def _wait(dialog, qapp, timeout_ms=30_000) -> None:
    """Run the event loop until the worker has finished and its signals landed."""
    from qtpy.QtCore import QDeadlineTimer
    deadline = QDeadlineTimer(timeout_ms)
    while dialog.worker is not None and not deadline.hasExpired():
        qapp.processEvents()
    assert dialog.worker is None, "export did not finish"


def test_dialog_exports_current_well_in_worker(qapp, model, out):
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model, current_well="B2")
    assert dialog.wells.selected() == ["B2"]                       # displayed well pre-selected
    assert dialog.out_dir.text() == "~/Downloads/videos-exp"      # shown with ~, never a full home path
    dialog.out_dir.setText(str(out))
    assert dialog.options().out_dir == out
    dialog.start()
    _wait(dialog, qapp)
    assert sorted(p.name for p in out.iterdir()) == ["B2_fov0.mp4"]
    assert "B2_fov0.mp4: 6 frames" in dialog.log.toPlainText()
    assert dialog.status.text().startswith("Done: 1 file")
    assert dialog.start_button.isEnabled() and not dialog.cancel_button.isEnabled()

    dialog.start()                                     # second run: nothing left to do
    _wait(dialog, qapp)
    assert dialog.log.toPlainText().count("frames") == 1
    assert dialog.status.text().startswith("Done: 0 file")


def test_grid_shape_uses_the_format_else_the_smallest_plate_that_fits():
    from record_zstack_viewer.ui.export_dialog import grid_shape
    assert grid_shape(["A1", "B2"], "unknown") == (2, 3, [])            # fits a 6-well
    assert grid_shape(["A1", "D6"], "unknown") == (4, 6, [])            # 24-well
    assert grid_shape(["A1", "H12"], "unknown") == (8, 12, [])          # 96-well
    assert grid_shape(["A1"], "384") == (16, 24, [])                    # known format wins
    assert grid_shape(["A1", "current"], "unknown") == (2, 3, ["current"])  # non-well id listed apart


def test_well_selector_is_a_plate_matrix(qapp, model):
    """Present wells are selectable, absent wells are not; a rectangle (what a
    shift-click does) only picks up present wells; All/None work."""
    from qtpy.QtWidgets import QTableWidgetSelectionRange
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model)                       # A1 and B2; fixture spacing 9 mm -> 96-well
    sel = dialog.wells
    assert (sel.rowCount(), sel.columnCount()) == (8, 12)
    assert sel.selected() == ["A1", "B2"]                          # nothing displayed: all present wells
    assert sel.item(0, 1).flags() == Qt.NoItemFlags                # A2 absent: greyed, unselectable
    sel.clearSelection()
    assert sel.selected() == []
    sel.setRangeSelected(QTableWidgetSelectionRange(0, 0, 7, 11), True)  # whole plate rectangle
    assert sel.selected() == ["A1", "B2"]
    sel.clearSelection()
    sel.item(1, 1).setSelected(True)
    assert sel.selected() == ["B2"]
    sel.selectAll()
    assert sel.selected() == ["A1", "B2"]


def test_empty_selection_is_refused_and_subset_exports(qapp, model, out):
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model)
    dialog.wells.clearSelection()
    dialog.out_dir.setText(str(out))
    dialog.start()
    assert dialog.worker is None and dialog.status.text() == "Select at least one well."
    dialog.wells.item(1, 1).setSelected(True)                      # B2 only
    dialog.start()
    _wait(dialog, qapp)
    assert sorted(p.name for p in out.iterdir()) == ["B2_fov0.mp4"]


def test_dialog_options_contrast_resolution_and_fovs(qapp, model, out, tmp_path, monkeypatch):
    from record_zstack_viewer.ui import export_dialog
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    # on-screen contrast is the default when given; auto otherwise
    d = ExportDialog(model, current_well="A1", screen_clims=(100.0, 2000.0))
    assert d.contrast.currentData() == (100.0, 2000.0) and d.contrast.count() == 2
    assert ExportDialog(model).contrast.currentData() is None
    assert d.fovs is None                                          # one FOV per well: no FOV row
    assert d.resolution.text() == "64 × 64" and d.full_res.isHidden()

    # frames over the cap: shown as downscaled, full_res restores them
    monkeypatch.setattr(export_dialog, "MAX_SIDE_PX", 32)
    d = ExportDialog(model)
    assert d.resolution.text().startswith("64 × 64 → 32 × 32") and not d.full_res.isHidden()
    assert d.options().max_side == 32
    d.full_res.setChecked(True)
    assert d.resolution.text() == "64 × 64" and d.options().max_side is None

    # several FOVs per well: a FOV matrix, same indices in every well
    root = generate_experiment(tmp_path / "multi", wells=["A1", "B2"], fovs_per_well=3, nt=1,
                               recording_frames=2, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=1)
    d = ExportDialog(ExperimentModel(root, default_context()))
    assert d.fovs is not None and d.fovs.selected() == ["0", "1", "2"]
    d.fovs.clearSelection()
    d.fovs.item(0, 1).setSelected(True)
    d.out_dir.setText(str(out))
    assert d.options().fovs == {1}
    d.start()
    _wait(d, qapp)
    assert sorted(p.name for p in out.iterdir()) == ["A1_fov1.mp4", "B2_fov1.mp4"]


def test_fov_viewer_exposes_current_clims(qapp, model):
    from record_zstack_viewer.ui.fov_viewer import FovViewer
    viewer = FovViewer()
    assert viewer.current_clims() is None
    datasets = model.open_fov("A1", 0)
    viewer.set_datasets(datasets, "recording")
    lo, hi = viewer.current_clims()
    assert 0 <= lo < hi <= 65535
    viewer.set_datasets(datasets, "zstack")
    assert viewer.current_clims() is None                          # only recordings export


def test_dialog_reports_a_planning_error_and_recovers(qapp, model, out, monkeypatch):
    from record_zstack_viewer.ui import export_dialog

    def broken(*a, **k):
        raise OSError("unreadable store")
    monkeypatch.setattr(export_dialog, "plan", broken)
    dialog = export_dialog.ExportDialog(model)
    dialog.out_dir.setText(str(out))
    dialog.start()
    _wait(dialog, qapp)
    assert "unreadable store" in dialog.log.toPlainText()
    assert dialog.start_button.isEnabled()


def test_main_window_reuses_one_export_dialog(qapp, single_plane):
    from record_zstack_viewer.ui.app import MainWindow
    win = MainWindow(single_plane)
    first = win.open_export_dialog()
    assert first.wells.selected() == ["A1", "B2"]     # nothing displayed yet: all wells
    second = win.open_export_dialog()                 # idle dialog is replaced, not stacked
    assert second is not first and win._export_dialog is second
    win.close()
