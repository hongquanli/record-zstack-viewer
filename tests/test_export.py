"""zarr -> mp4 export: planning, encoding, fidelity, incomplete stores, failures,
cancel, CLI, dialog (plate matrix, worker lifecycle)."""
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
from record_zstack_viewer.export import convert, mp4
from record_zstack_viewer.export.__main__ import main as cli_main
from record_zstack_viewer.model.experiment import ExperimentModel
from tests.fixtures.generate import generate_experiment


def _decode(path: Path, h: int, w: int, pix_fmt: str = "gray") -> np.ndarray:
    """All frames of `path` as a (n, h, w[, 3]) uint8 array, via ffmpeg rawvideo."""
    out = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"], capture_output=True, check=True).stdout
    shape = (-1, h, w) if pix_fmt == "gray" else (-1, h, w, 3)
    return np.frombuffer(out, dtype=np.uint8).reshape(shape)


def _metadata_comment(path: Path) -> str:
    """The container's `comment` tag, from `ffmpeg -i`'s stderr listing."""
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
                          capture_output=True, text=True).stderr
    return next((line.split(":", 1)[1].strip() for line in info.splitlines()
                 if line.strip().startswith("comment")), "")


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
    assert mp4.video_name("A1", 0, 0, 0, 1, 1) == "A1_fov_0.mp4"          # mirrors fov_0.ome.zarr
    assert mp4.video_name("A1", 2, 1, 0, 3, 1) == "A1_t1_fov_2.mp4"
    assert mp4.video_name("A1", 0, 0, 2, 1, 3) == "A1_fov_0_z2.mp4"       # mirrors fov_0_z2.ome.zarr
    assert mp4.video_name("A1", 0, 1, 2, 2, 3) == "A1_t1_fov_0_z2.mp4"


def test_plan_one_job_per_store_and_skips_up_to_date(model, out):
    jobs = mp4.plan(model, out)
    assert [(j.well, j.out.name) for j in jobs] == [("A1", "A1_fov_0.mp4"), ("B2", "B2_fov_0.mp4")]
    assert mp4.plan(model, out, wells=["B2"]) == jobs[1:]

    assert mp4.export_job(jobs[0]).ok
    assert [j.well for j in mp4.plan(model, out)] == ["B2"]           # A1 is up to date
    assert len(mp4.plan(model, out, overwrite=True)) == 2


def test_plan_multi_plane_time_lapse(tiny_experiment, out):
    model = ExperimentModel(tiny_experiment, default_context())  # 2 wells, nt=2, 3 planes
    names = sorted(j.out.name for j in mp4.plan(model, out))
    assert len(names) == 12
    assert names[0] == "A1_t0_fov_0_z0.mp4" and names[-1] == "B2_t1_fov_0_z2.mp4"


def test_default_out_dir_is_downloads_per_experiment(tmp_path):
    assert mp4.default_out_dir(tmp_path / "2026-10-30_run") == Path.home() / "Downloads" / "mp4-2026-10-30_run"


# ------------------------------------------------------------------ encoding

def test_export_writes_every_frame_at_full_size(model, out):
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(job)
    assert result.ok and result.frames == 6
    assert job.out.exists() and _tmp_files(out) == []
    assert _decode(job.out, 64, 64).shape == (6, 64, 64)


def test_window_is_the_videos_min_max_and_clips_nothing(model, out):
    source = model.open_fov("A1", 0).recording
    frames = np.stack([source.read_plane({"acq_t": 0, "frame": f, "z": 0}) for f in range(6)])
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(job)
    assert result.window == (float(frames.min()), float(frames.max()))
    assert result.summary() == f"A1_fov_0.mp4: 6 frames, window {frames.min()}–{frames.max()}"

    # the same linear map, applied by us, reproduces the decoded video closely (near-lossless)
    lo, hi = result.window
    expected = np.clip(np.rint((frames.astype(np.float32) - lo) * 255.0 / (hi - lo)), 0, 255)
    decoded = _decode(job.out, 64, 64).astype(np.float32)
    assert np.abs(decoded - expected).mean() < 2.0      # 64x64 noise fixture; smooth frames do better
    assert f"window {lo:g}-{hi:g}" in _metadata_comment(job.out)       # mapping travels with the file


def test_channel_color_is_the_viewers_ramp_applied_after_the_window(model, out):
    """The viewer paints a recording black -> channel colour, the colour being
    `omero.channels[0].color` in the store (Squid writes it without '#'). The
    mp4 applies the same ramp AFTER the 16-to-8 window, so contrast is exactly
    the gray video's and R:G:B follow the colour. White stays gray."""
    source = model.open_fov("A1", 0).recording
    frames = np.stack([source.read_plane({"acq_t": 0, "frame": f, "z": 0}) for f in range(6)])
    zarr_json = source.store_path(0, 0) / "zarr.json"
    meta = json.loads(zarr_json.read_text())
    assert convert.channel_color(source.store_path(0, 0)) is None            # fixture channel is white
    meta["attributes"]["ome"]["omero"]["channels"][0]["color"] = "1FFF00"
    zarr_json.write_text(json.dumps(meta))
    assert convert.channel_color(source.store_path(0, 0)) == "#1FFF00"

    job = mp4.plan(model, out)[0]
    result = mp4.export_job(job)
    assert result.ok and result.color == "#1FFF00"
    lo, hi = result.window
    assert (lo, hi) == (float(frames.min()), float(frames.max()))            # window unchanged by colour
    gray = np.clip(np.rint((frames.astype(np.float32) - lo) * 255.0 / (hi - lo)), 0, 255)
    expected = np.rint(gray[..., None] * np.array([0x1F, 0xFF, 0x00], dtype=np.float32) / 255.0)
    decoded = _decode(job.out, 64, 64, "rgb24").astype(np.float32)
    assert np.abs(decoded - expected).mean() < 4.0   # yuv420p stores colour at half resolution: ~3 levels
    assert result.summary().endswith(f"window {lo:g}–{hi:g}, color #1FFF00")
    assert "color #1FFF00" in _metadata_comment(job.out)


def test_real_time_fps_reaches_ffmpeg(model, out, monkeypatch):
    seen = []
    real = convert._ffmpeg_command
    monkeypatch.setattr(convert, "_ffmpeg_command",
                        lambda w, h, fps, o, *a: seen.append(fps) or real(w, h, fps, o, *a))
    job = mp4.plan(model, out)[0]
    assert mp4.export_job(job).ok and seen == [15.0]          # fixture: 1/15 s per frame
    assert mp4.export_job(job, fps=2.5).ok and seen[-1] == 2.5
    cmd = real(64, 64, 15.0, out / "x.mp4", (0.0, 1.0), "uint16", None)
    assert cmd[cmd.index("-preset") + 1] == "veryfast"        # measured: 3.6x medium, same PSNR


def test_window_and_color_from_the_viewer_are_copied(model, out):
    """From the viewer the mp4 is a copy of the screen: the displayed contrast
    limits and colour apply, values outside the limits clip as on screen, and
    both are written into the file's comment. Without them (CLI, Squid) the
    store's own min/max and channel colour apply, so the two paths differ only
    in the window, and the file says which it used."""
    source = model.open_fov("A1", 0).recording
    frames = np.stack([source.read_plane({"acq_t": 0, "frame": f, "z": 0}) for f in range(6)])
    lo, hi = float(np.percentile(frames, 30)), float(np.percentile(frames, 70))
    store = source.store_path(0, 0)

    result = convert.convert_store(store, out / "seen.mp4", window=(lo, hi), color="#FF0000")
    assert result.ok and result.window == (lo, hi) and result.color == "#FF0000"
    assert result.summary() == f"seen.mp4: 6 frames, window {lo:g}–{hi:g}, color #FF0000"
    assert f"window {lo:g}-{hi:g}" in _metadata_comment(result.out)
    assert "color #FF0000" in _metadata_comment(result.out)
    red = _decode(result.out, 64, 64, "rgb24").astype(float)
    # yuv420p stores colour at half resolution, so on this noise fixture judge regions, not pixels
    assert red[..., 0][frames >= hi].mean() > 200 and red[..., 0][frames <= lo].mean() < 50  # clipped as on screen
    assert red[..., 1].mean() < 10 and red[..., 2].mean() < 10                                 # a red ramp

    plain = convert.convert_store(store, out / "plain.mp4", window=(lo, hi), color="#FFFFFF")
    assert plain.ok and plain.color is None                 # white = gray, as the viewer shows it
    default = convert.convert_store(store, out / "default.mp4")
    assert default.window == (float(frames.min()), float(frames.max())) and default.color is None


def test_default_fps_prefers_effective_fps(single_plane):
    store = single_plane / "recording" / "t0" / "A1" / "fov_0.ome.zarr"
    meta = json.loads((store / "zarr.json").read_text())
    meta["attributes"]["_squid"]["effective_fps"] = 9.5
    (store / "zarr.json").write_text(json.dumps(meta))
    source = ExperimentModel(single_plane, default_context()).open_fov("A1", 0).recording
    assert source.metadata.effective_fps == 9.5 and convert.default_fps(store) == 9.5


def test_incomplete_store_exports_its_captured_frames_once_the_run_is_done(tmp_path, out):
    root = generate_experiment(tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
                               recording_frames=6, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2, complete=False)
    model = ExperimentModel(root, default_context())
    assert mp4.plan(model, out) == []                  # run still going: left alone
    (root / ".done").write_text("")                    # aborted run: .done without completion
    (job,) = mp4.plan(model, out)
    assert job.out.name == "A1_fov_0.mp4"             # the name mirrors the store, complete or not
    result = mp4.export_job(job)
    assert result.ok and result.frames == 3


def test_unwritten_slot_is_a_black_frame(model, out):
    """A translation, not a repair: a slot the recorder never wrote reads as the
    fill value and is written as it reads, a black frame."""
    source = model.open_fov("A1", 0).recording
    written = np.stack([source.read_plane({"acq_t": 0, "frame": f, "z": 0}) for f in (0, 1, 2, 4, 5)])
    shutil.rmtree(source.store_path(0, 0) / "c" / "3")  # frame 3 was never written
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(job)
    assert result.ok and result.frames == 6
    assert result.window == (0.0, float(written.max()))              # the black frame is data too
    assert result.summary() == f"A1_fov_0.mp4: 6 frames, window 0–{written.max()}"
    frames = _decode(job.out, 64, 64)
    assert frames[3].max() <= 1 and frames[2].mean() > 1


def test_odd_frame_size_is_cropped_even(tmp_path, out):
    root = generate_experiment(tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
                               recording_frames=2, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=1, hw=(33, 65))
    model = ExperimentModel(root, default_context())
    job = mp4.plan(model, out)[0]
    assert mp4.export_job(job).ok
    assert _decode(job.out, 32, 64).shape == (2, 32, 64)


# ------------------------------------------------------------- failure paths

def test_ffmpeg_failure_is_reported_and_leaves_no_tmp(model, out, monkeypatch):
    monkeypatch.setattr(convert, "_ffmpeg_command", lambda *a: ["sh", "-c", "echo boom >&2; exit 3"])
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(job)
    assert not result.ok and "boom" in result.error
    assert not job.out.exists() and _tmp_files(out) == []


def test_read_error_is_reported_and_leaves_no_tmp(model, out, monkeypatch):
    def broken(arr, n_frames):
        raise OSError("torn zarr.json")
    monkeypatch.setattr(convert, "data_window", broken)
    job = mp4.plan(model, out)[0]
    result = mp4.export_job(job)
    assert result.error == "torn zarr.json" and _tmp_files(out) == []


def test_cancel_discards_the_file_in_progress_and_stops(model, out):
    jobs = mp4.plan(model, out)
    results = list(mp4.export_all(jobs, cancelled=lambda: True))
    assert len(results) == 1 and results[0].error == mp4.CANCELLED
    assert not out.exists() or list(out.iterdir()) == []


# ----------------------------------------------------------------------- CLI

def test_cli_exports_all_then_reports_nothing_to_do(single_plane, out, capsys):
    assert cli_main([str(single_plane), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["A1_fov_0.mp4", "B2_fov_0.mp4"]
    assert cli_main([str(single_plane), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "A1_fov_0.mp4: already exists, skipped" in text and "nothing to export" in text


def test_cli_returns_1_when_a_file_fails(single_plane, out, monkeypatch, capsys):
    monkeypatch.setattr(convert, "_ffmpeg_command", lambda *a: ["sh", "-c", "exit 1"])
    assert cli_main([str(single_plane), "--out", str(out)]) == 1
    assert "FAILED" in capsys.readouterr().err


# -------------------------------------------------------------------- dialog

@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _wait(dialog, qapp, timeout_ms=30_000) -> None:
    """Run the event loop until the worker thread has exited and its signals landed."""
    from qtpy.QtCore import QDeadlineTimer
    deadline = QDeadlineTimer(timeout_ms)
    while dialog.worker is not None and not deadline.hasExpired():
        qapp.processEvents()
    assert dialog.worker is None, "export did not finish"


def test_grid_shape_uses_the_format_else_the_smallest_plate_that_fits():
    from record_zstack_viewer.ui.export_dialog import grid_shape
    assert grid_shape(["A1", "B2"], "unknown") == (2, 3, [])            # fits a 6-well
    assert grid_shape(["A1", "D6"], "unknown") == (4, 6, [])            # 24-well
    assert grid_shape(["A1", "H12"], "unknown") == (8, 12, [])          # 96-well
    assert grid_shape(["A1"], "384") == (16, 24, [])                    # known format wins
    assert grid_shape(["A1", "current"], "unknown") == (2, 3, ["current"])  # non-well id listed apart


def test_well_selector_is_a_plate_matrix(qapp, model):
    """Present wells are selectable, absent wells are not; a rectangle (what a
    shift-click does) only picks up present wells; All works; no None button."""
    from qtpy.QtWidgets import QPushButton, QTableWidgetSelectionRange
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model)                       # A1 and B2; fixture spacing 9 mm -> 96-well
    sel = dialog.wells
    assert (sel.rowCount(), sel.columnCount()) == (8, 12)
    assert sel.selected() == ["A1", "B2"]                          # nothing displayed: all present wells
    assert sel.item(0, 1).flags() == Qt.NoItemFlags                # A2 absent: greyed, unselectable
    sel.clearSelection()
    sel.setRangeSelected(QTableWidgetSelectionRange(0, 0, 7, 11), True)  # whole plate rectangle
    assert sel.selected() == ["A1", "B2"]
    sel.clearSelection()
    sel.item(1, 1).setSelected(True)
    assert sel.selected() == ["B2"]
    sel.selectAll()
    assert sel.selected() == ["A1", "B2"]
    assert [b.text() for b in dialog.findChildren(QPushButton)] == ["All", "Browse…", "Export", "Cancel", "Close"]


def test_selection_stays_blue_when_matrix_loses_focus(qapp, model):
    """Regression (Windows): clicking All moved focus to the button and Qt drew
    the selection in the inactive highlight, near white. The button must not
    take focus, and the matrix paints inactive selection like active."""
    from qtpy.QtGui import QPalette
    from qtpy.QtWidgets import QPushButton
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model)
    all_btn = next(b for b in dialog.findChildren(QPushButton) if b.text() == "All")
    assert all_btn.focusPolicy() == Qt.NoFocus
    pal = dialog.wells.palette()
    for role in (QPalette.Highlight, QPalette.HighlightedText):
        assert pal.color(QPalette.Inactive, role) == pal.color(QPalette.Active, role)


def test_dialog_exports_current_well_in_worker(qapp, model, out):
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model, current_well="B2")
    assert dialog.wells.selected() == ["B2"]                       # displayed well pre-selected
    assert dialog.out_dir.text() == "~/Downloads/mp4-exp"      # shown with ~, never a full home path
    dialog.out_dir.setText(str(out))
    assert dialog.output_path() == out
    dialog.start()
    _wait(dialog, qapp)
    assert sorted(p.name for p in out.iterdir()) == ["B2_fov_0.mp4"]
    assert "B2_fov_0.mp4: 6 frames, window" in dialog.log.toPlainText()
    assert dialog.status.text().startswith("Done: 1 file")
    assert dialog.start_button.isEnabled() and not dialog.cancel_button.isEnabled()

    dialog.start()                                     # second run: nothing left to do, and it says so
    _wait(dialog, qapp)
    assert dialog.log.toPlainText().count("frames") == 1
    assert "B2_fov_0.mp4: already exists, skipped" in dialog.log.toPlainText()
    assert dialog.status.text().startswith("Done: 0 file")


def test_dialog_passes_the_displayed_lut_to_every_file(qapp, model, out):
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model, window=(100.0, 200.0), color="#00FF00")
    assert dialog.contrast.text() == "100 to 200 as displayed, colour #00FF00"
    dialog.out_dir.setText(str(out))
    dialog.start()
    _wait(dialog, qapp)
    log = dialog.log.toPlainText()
    assert "A1_fov_0.mp4: 6 frames, window 100–200, color #00FF00" in log
    assert "B2_fov_0.mp4: 6 frames, window 100–200, color #00FF00" in log
    assert ExportDialog(model).contrast.text() == "each video's own min and max, channel colour as recorded"


def test_close_button_closes_idle_and_after_export(qapp, model, out):
    """Regression: reject() called close(), which Qt ignores while already
    closing, so the Close button and Esc never hid the dialog."""
    from qtpy.QtWidgets import QPushButton
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model)
    dialog.show()
    close = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Close")
    close.click()
    assert not dialog.isVisible()

    dialog = ExportDialog(model)
    dialog.show()
    dialog.out_dir.setText(str(out))
    dialog.start()
    _wait(dialog, qapp)
    close = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Close")
    close.click()
    assert not dialog.isVisible()

    dialog = ExportDialog(model)                   # Esc path, with a running export
    dialog.show()
    dialog.out_dir.setText(str(out / "again"))
    dialog.start()
    dialog.reject()
    assert not dialog.isVisible() and not dialog.is_running


def test_worker_reference_survives_until_the_thread_exits(qapp, model, out):
    """Regression: `finished_all` is emitted from inside run(); releasing the
    QThread there destroyed it while still running and aborted the app."""
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model)
    dialog.out_dir.setText(str(out))
    dialog.start()
    worker = dialog.worker
    dialog._on_finished(2, "")                         # what finished_all triggers
    assert dialog.worker is worker                     # still held: the thread may be running
    _wait(dialog, qapp)                                # released only by `finished`
    assert dialog.worker is None and not worker.isRunning()


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
    assert sorted(p.name for p in out.iterdir()) == ["B2_fov_0.mp4"]


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


# ------------------------------------------------------------ format guard

def test_converter_refuses_a_store_it_does_not_understand(model, out):
    """A format change must fail loudly, never convert the wrong thing."""
    job = mp4.plan(model, out)[0]
    assert convert.check_store(job.store)["shape"] == [6, 1, 1, 64, 64]

    meta_path = job.store / "zarr.json"
    meta = json.loads(meta_path.read_text())
    meta["shape"] = [6, 2, 1, 64, 64]                                  # a channel axis appeared
    meta_path.write_text(json.dumps(meta))
    result = mp4.export_job(job)
    assert not result.ok and "store format not supported" in result.error and "expected (T, 1, 1, Y, X)" in result.error
    assert not job.out.exists() and _tmp_files(out) == []

    meta["shape"] = [6, 1, 1, 64, 64]
    meta["attributes"]["_squid"]["structure"] = "4D-TZYX"               # producer renamed its layout
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(convert.UnsupportedStore, match="_squid.structure"):
        convert.check_store(job.store)

    with pytest.raises(convert.UnsupportedStore, match="no zarr.json"):
        convert.check_store(out / "nowhere")
