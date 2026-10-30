"""The "Export mp4…" dialog: scope, output folder, fps, overwrite, progress.

Planning and encoding both run on a `QThread` so the window and its live
following stay responsive; one file at a time (`export_all`). Cancel stops at
the next frame and discards the file in progress. The output folder is
remembered across sessions (QSettings).
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSettings, QThread, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                            QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
                            QWidget)

from record_zstack_viewer.export.mp4 import CANCELLED, ExportResult, export_all, plan
from record_zstack_viewer.model.experiment import ExperimentModel

_SETTINGS = ("cephla", "record-zstack-viewer")
_OUT_DIR_KEY = "export/out_dir"


class ExportWorker(QThread):
    """Plans and runs an export off the UI thread; `cancel()` stops at the next frame."""

    planned = Signal(int)                   # number of files to write
    file_progress = Signal(int, int)        # frames done, frames total (current file)
    file_done = Signal(object)              # ExportResult
    finished_all = Signal(int, str)         # files written, error text ("" when none)

    def __init__(self, model: ExperimentModel, out_dir: Path, wells: list[str] | None,
                 overwrite: bool, fps: float | None) -> None:
        super().__init__()
        self._args = (model, out_dir, wells, overwrite, fps)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401 (QThread entry point)
        model, out_dir, wells, overwrite, fps = self._args
        written = 0
        try:
            jobs = plan(model, out_dir, wells=wells, overwrite=overwrite)
            self.planned.emit(len(jobs))
            for result in export_all(model, jobs, fps=fps,
                                     on_progress=self.file_progress.emit,
                                     cancelled=lambda: self._cancel):
                written += int(result.ok)
                self.file_done.emit(result)
        except Exception as exc:  # a torn zarr.json or an unreadable store: report, never hang
            self.finished_all.emit(written, str(exc) or type(exc).__name__)
            return
        self.finished_all.emit(written, "")


class ExportDialog(QDialog):
    """Public surface for tests: `.scope`, `.out_dir`, `.fps`, `.overwrite`,
    `.start_button`, `.cancel_button`, `.log`, `.status`, `.start()`, `.worker`."""

    def __init__(self, model: ExperimentModel, current_well: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self.worker: ExportWorker | None = None
        self.setWindowTitle("Export recordings to mp4")
        self.setMinimumWidth(520)

        self.scope = QComboBox(self)
        self.scope.addItem("All wells", None)
        if current_well is not None:
            self.scope.addItem(f"Well {current_well}", current_well)
            self.scope.setCurrentIndex(1)

        default_dir = Path(model.root) / "videos"
        self.out_dir = QLineEdit(str(QSettings(*_SETTINGS).value(_OUT_DIR_KEY, str(default_dir))), self)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(browse)

        self.fps = QDoubleSpinBox(self)
        self.fps.setRange(0.0, 1000.0)
        self.fps.setDecimals(2)
        self.fps.setSpecialValueText("real time (as recorded)")
        self.fps.setValue(0.0)
        self.overwrite = QCheckBox("Re-export files that already exist", self)

        self.progress = QProgressBar(self)
        self.progress.setFormat("%v / %m files")
        self.status = QLabel("", self)
        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000)

        self.start_button = QPushButton("Export", self)
        self.start_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        close = QPushButton("Close", self)
        close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(close)

        form = QFormLayout(self)
        form.addRow("Scope", self.scope)
        form.addRow("Output folder", out_row)
        form.addRow("Playback fps", self.fps)
        form.addRow("", self.overwrite)
        form.addRow(self.progress)
        form.addRow(self.status)
        form.addRow(self.log)
        form.addRow(buttons)

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # ------------------------------------------------------------ actions

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if chosen:
            self.out_dir.setText(chosen)

    def start(self) -> None:
        """Plan and encode the chosen scope in the worker thread."""
        if self.is_running:
            return
        out_dir = Path(self.out_dir.text()).expanduser()
        QSettings(*_SETTINGS).setValue(_OUT_DIR_KEY, str(out_dir))
        well = self.scope.currentData()
        self.progress.setRange(0, 0)  # busy until planned
        self.status.setText(f"Planning export to {out_dir}…")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = ExportWorker(self._model, out_dir, [well] if well else None,
                                   self.overwrite.isChecked(), self.fps.value() or None)
        self.worker.planned.connect(self._on_planned)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Cancelling…")

    def _on_planned(self, n_jobs: int) -> None:
        self.progress.setRange(0, max(n_jobs, 1))
        self.progress.setValue(0)
        if n_jobs == 0:
            self.status.setText("Nothing to export: every mp4 is up to date.")

    def _on_file_progress(self, done: int, total: int) -> None:
        self.status.setText(f"File {self.progress.value() + 1} / {self.progress.maximum()}: "
                            f"frame {done} / {total}")

    def _on_file_done(self, result: ExportResult) -> None:
        self.progress.setValue(self.progress.value() + 1)
        name = result.job.out.name
        if result.ok:
            held = f", {result.held} held frame(s)" if result.held else ""
            self.log.appendPlainText(f"{name}: {result.frames} frames{held}")
        elif result.error == CANCELLED:
            self.log.appendPlainText(f"{name}: cancelled")
        else:
            self.log.appendPlainText(f"{name}: FAILED: {result.error}")

    def _on_finished(self, written: int, error: str) -> None:
        if error:
            self.log.appendPlainText(f"Export stopped: {error}")
        self.status.setText(f"Done: {written} file(s) written." if not error else "Export failed.")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.worker = None

    # ----------------------------------------------------------- lifecycle

    def reject(self) -> None:  # Esc: close (and so cancel), never just hide a running export
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self.is_running:
            self.worker.cancel()
            self.worker.wait()
        super().closeEvent(event)
