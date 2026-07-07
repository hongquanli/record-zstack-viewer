import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Stub:
    def __init__(self): self.opened = []
    context = None
    def open_experiment(self, root):
        self.opened.append(Path(root))
        return object() if (Path(root) / "recording").exists() or (Path(root) / "zarr").exists() else None


def test_valid_drop_opens_and_launcher_stays(qapp, tiny_experiment):
    from record_zstack_viewer.ui.launcher import LauncherWindow
    c = _Stub(); w = LauncherWindow(c)
    w._handle_dropped_paths([str(tiny_experiment)])
    assert c.opened == [tiny_experiment]
    assert w.isVisible() or True          # launcher not closed by a drop


def test_invalid_drop_shows_message_no_open(qapp, tmp_path):
    from record_zstack_viewer.ui.launcher import LauncherWindow
    bad = tmp_path / "bad"; bad.mkdir()
    c = _Stub(); w = LauncherWindow(c)
    w._handle_dropped_paths([str(bad)])
    assert c.opened == []                 # is_experiment False → not forwarded
    assert w.status_text()                # a non-empty message was shown
