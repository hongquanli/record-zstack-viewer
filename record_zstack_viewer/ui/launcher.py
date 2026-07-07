"""Persistent drag-and-drop launcher: drop (or Browse to) an experiment folder
to open a viewer window; stays open so several experiments can be opened."""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFileDialog, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget)

from record_zstack_viewer.ui.controller import is_experiment


class LauncherWindow(QMainWindow):
    def __init__(self, controller, parent=None) -> None:
        super().__init__(parent)
        self._controller = controller
        self.setWindowTitle("Record + Z-Stack Viewer")
        self.setAcceptDrops(True)
        central = QWidget(self); lay = QVBoxLayout(central)
        self._drop = QLabel("Drop a Squid experiment folder here", central)
        self._drop.setAlignment(Qt.AlignCenter)
        self._drop.setStyleSheet(
            "border:2px dashed palette(mid); padding:48px; font-size:15px;")
        browse = QPushButton("Browse…", central)
        browse.clicked.connect(self._browse)
        self._status = QLabel("", central); self._status.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._drop); lay.addWidget(browse); lay.addWidget(self._status)
        self.setCentralWidget(central)
        self.resize(420, 300)

    # -- API used by tests / events --
    def status_text(self) -> str:
        return self._status.text()

    def _handle_dropped_paths(self, paths: list[str]) -> None:
        opened = 0
        for p in paths:
            root = Path(p)
            if is_experiment(root) and self._controller.open_experiment(root) is not None:
                opened += 1
            else:
                self._status.setText(f"Not a record+zstack experiment: {root.name}")
        if opened:
            self._status.setText(f"Opened {opened} experiment(s).")

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose an experiment folder")
        if d:
            self._handle_dropped_paths([d])

    def dragEnterEvent(self, event):  # noqa: N802
        md = event.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        md = event.mimeData()
        self._handle_dropped_paths(
            [u.toLocalFile() for u in md.urls() if u.isLocalFile()])
        event.acceptProposedAction()
