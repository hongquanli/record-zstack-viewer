"""App-level owner for the multi-window model: one shared TensorStore context
(a single cache_pool for ALL windows) + a registry + open_experiment()."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from record_zstack_viewer.cache.memory import default_context

if TYPE_CHECKING:
    from record_zstack_viewer.ui.app import MainWindow


def is_experiment(root: Path) -> bool:
    """True if `root` looks like a record+zstack experiment (cheap structural check)."""
    root = Path(root)
    return root.is_dir() and ((root / "recording").is_dir() or (root / "zarr").is_dir())


class ViewerController:
    def __init__(self, context=None) -> None:
        self.context = context if context is not None else default_context()
        self._windows: list["MainWindow"] = []
        self.launcher = None  # set by __main__ when a LauncherWindow exists

    def open_experiment(self, root) -> "MainWindow | None":
        root = Path(root).expanduser()
        if not is_experiment(root):
            return None
        from record_zstack_viewer.ui.app import MainWindow  # lazy: keep ndv out of import
        win = MainWindow(root, context=self.context, controller=self)
        self._windows.append(win)
        win.destroyed.connect(lambda *_: self._drop(win))
        win.show()
        return win

    def _drop(self, win) -> None:
        if win in self._windows:
            self._windows.remove(win)

    def windows(self) -> list:
        return list(self._windows)
