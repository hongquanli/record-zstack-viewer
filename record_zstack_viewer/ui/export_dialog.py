"""The "Export mp4…" dialog: scope, output folder, fps, overwrite, progress.

Encoding runs on a `QThread` so the window and its live following stay
responsive; one file at a time (`export_all`), cancel stops after the current
file. The output folder is remembered across sessions (QSettings).
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QSettings, QThread, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                            QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
                            QWidget)

from record_zstack_viewer.export.mp4 import ExportResult, export_all, plan
from record_zstack_viewer.model.experiment import ExperimentModel

_SETTINGS = ("cephla", "record-zstack-viewer")
_OUT_DIR_KEY = "export/out_dir"


class ExportWorker(QThread):
    """Runs `export_all` off the UI thread; `cancel()` stops after the current file."""

    file_progress = Signal(int, int)        # frames done, frames total (current file)
    file_done = Signal(object)              # ExportResult
    finished_all = Signal(int)              # number of files written

    def __init__(self, model: ExperimentModel, jobs: list, fps: float | None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model, self._jobs, self._fps = model, jobs, fps
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401 (QThread entry point)
        written = 0
        for result in export_all(self._model, self._jobs, fps=self._fps,
                                 on_progress=self.file_progress.emit,
                                 cancelled=lambda: self._cancel):
            written += int(result.ok)
            self.file_done.emit(result)
        self.finished_all.emit(written)


class ExportDialog(QDialog):
    """Public surface for tests: `.scope`, `.out_dir`, `.fps`, `.overwrite`,
    `.start_button`, `.cancel_button`, `.log`, `.status`, `.start()`, `.worker`."""

    def __init__(self, model: ExperimentModel, current_well: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self.worker: ExportWorker | None = None
        self._done = 0
        self.setWindowTitle("Export recordings to mp4")
        self.setMinimumWidth(520)

        self.scope = QComboBox(self)
        self.scope.addItem("All wells", None)
        if current_well is not None:
            self.scope.addItem(f"Well {current_well}", current_well)
            self.scope.setCurrentIndex(1)

        settings = QSettings(*_SETTINGS)
        default_dir = Path(model.root) / "videos"
        self.out_dir = QLineEdit(str(settings.value(_OUT_DIR_KEY, str(default_dir))), self)
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

    # ------------------------------------------------------------ actions

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if chosen:
            self.out_dir.setText(chosen)

    def start(self) -> None:
        """Plan the jobs for the chosen scope and start encoding in the worker."""
        out_dir = Path(self.out_dir.text()).expanduser()
        QSettings(*_SETTINGS).setValue(_OUT_DIR_KEY, str(out_dir))
        well = self.scope.currentData()
        jobs = plan(self._model, out_dir, wells=[well] if well else None,
                    overwrite=self.overwrite.isChecked())
        if not jobs:
            self.status.setText("Nothing to export: every mp4 is up to date.")
            return
        self._done = 0
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.status.setText(f"Exporting {len(jobs)} file(s) to {out_dir}")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = ExportWorker(self._model, jobs, self.fps.value() or None, parent=self)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Cancelling after the current file…")

    def _on_file_progress(self, done: int, total: int) -> None:
        self.status.setText(f"File {self._done + 1} / {self.progress.maximum()}: "
                            f"frame {done} / {total}")

    def _on_file_done(self, result: ExportResult) -> None:
        self._done += 1
        self.progress.setValue(self._done)
        name = result.job.out.name
        if result.ok:
            extra = f", {result.held} held frame(s)" if result.held else ""
            extra += ", partial store" if result.partial else ""
            self.log.appendPlainText(f"{name}: {result.frames} frames{extra}")
        else:
            self.log.appendPlainText(f"{name}: FAILED: {result.error}")

    def _on_finished(self, written: int) -> None:
        self.status.setText(f"Done: {written} file(s) written.")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().closeEvent(event)
