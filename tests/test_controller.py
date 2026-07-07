import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import pytest
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_is_experiment(tiny_experiment, tmp_path):
    from record_zstack_viewer.ui.controller import is_experiment
    assert is_experiment(tiny_experiment) is True
    assert is_experiment(tmp_path / "does_not_exist") is False
    empty = tmp_path / "empty"; empty.mkdir()
    assert is_experiment(empty) is False            # no recording/ or zarr/


def test_controller_opens_and_registers_sharing_one_context(qapp, tiny_experiment):
    from record_zstack_viewer.ui.controller import ViewerController
    c = ViewerController()
    w1 = c.open_experiment(tiny_experiment)
    w2 = c.open_experiment(tiny_experiment)
    assert w1 is not None and w2 is not None and w1 is not w2
    assert len(c.windows()) == 2
    assert w1._context is w2._context is c.context      # ONE shared cache_pool context


def test_controller_rejects_invalid(qapp, tmp_path):
    from record_zstack_viewer.ui.controller import ViewerController
    c = ViewerController()
    assert c.open_experiment(tmp_path / "nope") is None
    assert c.windows() == []
