"""zarr -> mp4 export: planning, encoding, partial stores, holes, cancel, CLI, dialog."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

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


@pytest.fixture
def single_plane(tmp_path):
    """2 wells x 1 FOV, Nt=1, one plane: 2 recording stores of 6 frames (64x64)."""
    return generate_experiment(tmp_path / "exp", wells=["A1", "B2"], fovs_per_well=1, nt=1,
                               recording_frames=6, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2)


@pytest.fixture
def model(single_plane):
    return ExperimentModel(single_plane, default_context())


# ------------------------------------------------------------------ planning

def test_video_name_adds_axes_only_when_needed():
    assert mp4.video_name("A1", 0, 0, 0, 1, 1, False) == "A1_fov0.mp4"
    assert mp4.video_name("A1", 2, 1, 0, 3, 1, False) == "A1_fov2_t1.mp4"
    assert mp4.video_name("A1", 0, 0, 2, 1, 3, False) == "A1_fov0_z2.mp4"
    assert mp4.video_name("A1", 0, 1, 2, 2, 3, True) == "A1_fov0_t1_z2_partial.mp4"


def test_plan_one_job_per_store_and_skips_up_to_date(model, tmp_path):
    out = tmp_path / "videos"
    jobs = mp4.plan(model, out)
    assert [(j.well, j.out.name) for j in jobs] == [("A1", "A1_fov0.mp4"), ("B2", "B2_fov0.mp4")]
    assert mp4.plan(model, out, wells=["B2"]) == jobs[1:]

    assert mp4.export_job(model, jobs[0]).ok
    assert [j.well for j in mp4.plan(model, out)] == ["B2"]           # A1 is up to date
    assert len(mp4.plan(model, out, overwrite=True)) == 2


def test_plan_multi_plane_time_lapse(tiny_experiment, tmp_path):
    model = ExperimentModel(tiny_experiment, default_context())  # 2 wells, nt=2, 3 planes
    names = sorted(j.out.name for j in mp4.plan(model, tmp_path))
    assert len(names) == 12
    assert names[0] == "A1_fov0_t0_z0.mp4" and names[-1] == "B2_fov0_t1_z2.mp4"


# ------------------------------------------------------------------ encoding

def test_export_writes_every_frame_at_real_time_fps(model, tmp_path):
    job = mp4.plan(model, tmp_path)[0]
    result = mp4.export_job(model, job)
    assert result.ok and result.frames == 6 and result.held == 0 and not result.partial
    assert job.out.exists() and not job.out.with_name(job.out.name + ".tmp").exists()

    frames = _decode(job.out, 64, 64)
    assert frames.shape == (6, 64, 64)
    # fixed contrast: the bright blob stays bright in every frame, frames differ in the noise
    assert frames.max() > 200 and (frames[0] != frames[1]).any()

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate", "-of", "json", str(job.out)],
                           capture_output=True, text=True)
    if probe.returncode == 0:  # ffprobe is not shipped by imageio-ffmpeg; check when present
        assert json.loads(probe.stdout)["streams"][0]["r_frame_rate"] == "15/1"


def test_default_fps_prefers_effective_fps(model):
    source = model.open_fov("A1", 0).recording
    assert mp4.default_fps(source) == 15.0            # from time_increment_s = 1/15
    source.metadata.effective_fps = 9.5
    assert mp4.default_fps(source) == 9.5


def test_partial_store_exports_captured_frames(tmp_path):
    root = generate_experiment(tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
                               recording_frames=6, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2, complete=False)
    model = ExperimentModel(root, default_context())
    assert mp4.plan(model, tmp_path / "v") == []       # run still going: left alone
    (root / ".done").write_text("")                    # aborted run: .done without completion
    (job,) = mp4.plan(model, tmp_path / "v")
    assert job.out.name == "A1_fov0_partial.mp4"
    result = mp4.export_job(model, job)
    assert result.ok and result.partial and result.frames == 3


def test_unwritten_slot_repeats_previous_frame(model, tmp_path):
    store = model.open_fov("A1", 0).recording.store_path(0, 0)
    shutil.rmtree(store / "c" / "3")                   # frame 3 was never written
    job = mp4.plan(model, tmp_path)[0]
    result = mp4.export_job(model, job)
    assert result.ok and result.held == 1 and result.frames == 6
    frames = _decode(job.out, 64, 64)
    assert np.abs(frames[3].astype(int) - frames[2].astype(int)).mean() < 1.0


def test_cancel_stops_after_current_file_and_leaves_no_tmp(model, tmp_path):
    jobs = mp4.plan(model, tmp_path / "v")
    results = list(mp4.export_all(model, jobs, cancelled=lambda: True))
    assert len(results) == 1 and results[0].error == "cancelled"
    assert list((tmp_path / "v").glob("*")) == []


def test_odd_frame_size_is_cropped_even(tmp_path):
    root = generate_experiment(tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
                               recording_frames=2, recording_planes=1,
                               zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=1, hw=(33, 65))
    model = ExperimentModel(root, default_context())
    job = mp4.plan(model, tmp_path / "v")[0]
    assert mp4.export_job(model, job).ok
    assert _decode(job.out, 32, 64).shape == (2, 32, 64)


def test_oversized_frames_are_downscaled(monkeypatch):
    monkeypatch.setattr(mp4, "MAX_SIDE_PX", 32)
    assert "-vf" not in mp4._ffmpeg_command(32, 20, 10.0, Path("x.mp4"))
    cmd = mp4._ffmpeg_command(64, 40, 10.0, Path("x.mp4"))
    assert cmd[cmd.index("-vf") + 1] == "scale=32:-2"
    cmd = mp4._ffmpeg_command(40, 64, 10.0, Path("x.mp4"))
    assert cmd[cmd.index("-vf") + 1] == "scale=-2:32"


# ----------------------------------------------------------------------- CLI

def test_cli_exports_all_then_reports_nothing_to_do(single_plane, capsys):
    assert cli_main([str(single_plane)]) == 0
    assert sorted(p.name for p in (single_plane / "videos").iterdir()) == ["A1_fov0.mp4", "B2_fov0.mp4"]
    assert cli_main([str(single_plane)]) == 0
    assert "nothing to export" in capsys.readouterr().out


# -------------------------------------------------------------------- dialog

@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def test_dialog_exports_current_well_in_worker(qapp, model, tmp_path, monkeypatch):
    from qtpy.QtCore import QSettings
    monkeypatch.setattr(QSettings, "setValue", lambda self, *a: None)   # keep the test's folder out of real settings
    from record_zstack_viewer.ui.export_dialog import ExportDialog

    dialog = ExportDialog(model, current_well="B2")
    assert dialog.scope.currentData() == "B2"
    dialog.out_dir.setText(str(tmp_path / "v"))
    dialog.start()
    assert dialog.worker is not None and dialog.worker.wait(30_000)
    qapp.processEvents()
    assert sorted(p.name for p in (tmp_path / "v").iterdir()) == ["B2_fov0.mp4"]
    assert "B2_fov0.mp4: 6 frames" in dialog.log.toPlainText()
    assert dialog.status.text().startswith("Done: 1 file")
    assert dialog.start_button.isEnabled() and not dialog.cancel_button.isEnabled()


def test_main_window_has_export_button(qapp, single_plane):
    from record_zstack_viewer.ui.app import MainWindow
    win = MainWindow(single_plane)
    dialog = win.open_export_dialog()
    assert dialog.scope.count() == 1                   # nothing displayed yet: all wells only
    dialog.close()
    win.close()
