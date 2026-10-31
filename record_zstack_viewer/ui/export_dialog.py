"""The "Export mp4…" dialog: plate matrix, output folder, fps, overwrite, progress.

Planning and encoding both run on a `QThread` so the window and its live
following stay responsive; one file at a time (`export_all`). Cancel stops at
the next frame and discards the file in progress. Output defaults to
`~/Downloads/mp4-<experiment name>/`; paths are shown with `~` for home.
"""
from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (QAbstractItemView, QDialog, QDoubleSpinBox,
                            QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                            QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
                            QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from record_zstack_viewer.export.mp4 import ConvertResult, default_out_dir, export_all, plan
from record_zstack_viewer.model.experiment import ExperimentModel
from record_zstack_viewer.model.geometry import PLATE_FORMATS, well_name_to_rc

_CELL_PX = (10, 26)        # min / max side of a plate cell
_GRID_WIDTH_PX = 480       # the matrix fits this width for every plate format
_ABSENT = QColor(235, 235, 235)


class ExportWorker(QThread):
    """Plans and runs an export off the UI thread; `cancel()` stops at the next frame."""

    planned = Signal(int, object)           # files to write, list[Path] already up to date
    file_progress = Signal(int, int)        # frames done, frames total (current file)
    file_done = Signal(object)              # ConvertResult
    finished_all = Signal(int, str)         # files written, error text ("" when none)

    def __init__(self, model: ExperimentModel, out_dir: Path, wells: list[str],
                 fps: float | None) -> None:
        super().__init__()
        self._args = (model, out_dir, wells, fps)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401 (QThread entry point)
        model, out_dir, wells, fps = self._args
        written = 0
        try:
            skipped: list[Path] = []
            jobs = plan(model, out_dir, wells=wells, skipped=skipped)
            self.planned.emit(len(jobs), skipped)
            for result in export_all(jobs, fps=fps, on_progress=self.file_progress.emit,
                                     cancelled=lambda: self._cancel):
                written += int(result.ok)
                self.file_done.emit(result)
        except Exception as exc:  # a torn zarr.json or an unreadable store: report, never hang
            self.finished_all.emit(written, str(exc) or type(exc).__name__)
            return
        self.finished_all.emit(written, "")


# ------------------------------------------------------------- plate matrix

def grid_shape(wells: list[str], plate_format: str) -> tuple[int, int, list[str]]:
    """`(rows, cols, other)` for the plate matrix.

    Rows/cols come from the known plate format, else from the smallest standard
    plate (6 … 1536) that holds every well present, else from the wells alone.
    `other` lists region ids that are not well names (a "Current Position" run).
    """
    rc, other = [], []
    for name in wells:
        try:
            rc.append(well_name_to_rc(name))
        except ValueError:
            other.append(name)
    if plate_format in PLATE_FORMATS:
        fmt = PLATE_FORMATS[plate_format]
        return fmt.rows, fmt.cols, other
    need_r = 1 + max((r for r, _ in rc), default=0)
    need_c = 1 + max((c for _, c in rc), default=0)
    for fmt in PLATE_FORMATS.values():  # ordered small to large
        if fmt.rows >= need_r and fmt.cols >= need_c:
            return fmt.rows, fmt.cols, other
    return need_r, need_c, other


def _row_label(r: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (the inverse of `well_name_to_rc`)."""
    label, r = "", r + 1
    while r:
        r, rem = divmod(r - 1, 26)
        label = chr(ord("A") + rem) + label
    return label


class WellSelector(QTableWidget):
    """The plate as a clickable matrix: click, shift-click a rectangle, drag,
    ctrl/cmd-click to add. Wells absent from the experiment are greyed and
    cannot be selected; non-well region ids get an extra bottom row.
    `selected()` returns the selected names in plate order."""

    def __init__(self, wells: list[str], plate_format: str, selected: set[str],
                 parent: QWidget | None = None) -> None:
        rows, cols, other = grid_shape(wells, plate_format)
        super().__init__(rows + bool(other), max(cols, len(other)), parent)
        self._name = {(r, c): f"{_row_label(r)}{c + 1}" for r in range(rows) for c in range(cols)}
        self._name.update({(rows, i): name for i, name in enumerate(other)})
        present = set(wells)
        for r in range(self.rowCount()):
            for c in range(self.columnCount()):
                item = QTableWidgetItem()
                name = self._name.get((r, c))
                if name in present:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setToolTip(name)
                    if r == rows:                      # 'other' row: show the region id
                        item.setText(name)
                else:
                    item.setFlags(Qt.NoItemFlags)
                    item.setBackground(_ABSENT)
                self.setItem(r, c, item)
        self.setHorizontalHeaderLabels([str(c + 1) for c in range(self.columnCount())])
        self.setVerticalHeaderLabels([_row_label(r) for r in range(rows)] + (["other"] if other else []))
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        # Keep selected wells blue when focus is elsewhere (the All button, the
        # fps box): Qt otherwise paints them in the inactive highlight, near
        # white on Windows.
        pal = self.palette()
        for role in (QPalette.Highlight, QPalette.HighlightedText):
            pal.setColor(QPalette.Inactive, role, pal.color(QPalette.Active, role))
        self.setPalette(pal)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side = max(_CELL_PX[0], min(_CELL_PX[1], _GRID_WIDTH_PX // self.columnCount()))
        for header in (self.horizontalHeader(), self.verticalHeader()):
            header.setDefaultSectionSize(side)
            header.setMinimumSectionSize(side)
            font = header.font()
            font.setPointSize(7 if self.columnCount() > 24 else 9)
            header.setFont(font)
        self.setFixedSize(self.verticalHeader().width() + side * self.columnCount() + 2,
                          self.horizontalHeader().height() + side * self.rowCount() + 2)
        for rc, name in self._name.items():
            if name in selected:
                self.item(*rc).setSelected(True)

    def selected(self) -> list[str]:
        cells = sorted((i.row(), i.column()) for i in self.selectedItems())
        return [self._name[rc] for rc in cells]


# -------------------------------------------------------------------- dialog

class ExportDialog(QDialog):
    """Public surface for tests: `.wells`, `.out_dir`, `.fps`, `.start_button`,
    `.cancel_button`, `.log`, `.status`, `.start()`, `.worker`, `.output_path()`."""

    def __init__(self, model: ExperimentModel, current_well: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self.worker: ExportWorker | None = None
        self.setWindowTitle("Export recordings to mp4")
        self.setMinimumWidth(600)

        wells = model.wells()
        self.wells = WellSelector(wells, model.plate_format(),
                                  {current_well} if current_well in wells else set(wells), self)
        all_btn = QPushButton("All", self)
        all_btn.setFocusPolicy(Qt.NoFocus)  # focus stays on the matrix
        all_btn.clicked.connect(self.wells.selectAll)
        wells_row = QHBoxLayout()
        wells_row.addWidget(all_btn)
        wells_row.addWidget(QLabel("click · shift-click a range · drag", self))
        wells_row.addStretch(1)
        wells_box = QVBoxLayout()
        wells_box.addWidget(self.wells)
        wells_box.addLayout(wells_row)

        self.out_dir = QLineEdit(_short(default_out_dir(model.root)), self)
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
        for b in (self.start_button, self.cancel_button, close):
            buttons.addWidget(b)

        form = QFormLayout(self)
        form.addRow("Wells", wells_box)
        form.addRow("Output folder", out_row)
        form.addRow("Playback fps", self.fps)
        form.addRow(self.progress)
        form.addRow(self.status)
        form.addRow(self.log)
        form.addRow(buttons)

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def output_path(self) -> Path:
        return Path(self.out_dir.text()).expanduser()

    # ------------------------------------------------------------ actions

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", str(self.output_path()))
        if chosen:
            self.out_dir.setText(_short(Path(chosen)))

    def start(self) -> None:
        """Plan and encode the selected wells in the worker thread."""
        if self.is_running:
            return
        wells = self.wells.selected()
        if not wells:
            self.status.setText("Select at least one well.")
            return
        out_dir = self.output_path()
        self.progress.setRange(0, 0)  # busy until planned
        self.status.setText(f"Planning export to {_short(out_dir)}…")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = ExportWorker(self._model, out_dir, wells, self.fps.value() or None)
        self.worker.planned.connect(self._on_planned)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.finished_all.connect(self._on_finished)
        # The worker reference is released only once the OS thread has exited
        # (`finished`), never from `_on_finished`: `finished_all` is emitted from
        # inside run(), and dropping the last reference there destroys a QThread
        # that is still running, which aborts the process.
        self.worker.finished.connect(self._on_thread_exited)
        self.worker.start()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText("Cancelling…")

    def _on_planned(self, n_jobs: int, skipped: list) -> None:
        self.progress.setRange(0, max(n_jobs, 1))
        self.progress.setValue(0)
        for path in skipped:
            self.log.appendPlainText(f"{Path(path).name}: already exists, skipped (delete it to redo)")
        if n_jobs == 0:
            self.status.setText(f"Nothing new to export: {len(skipped)} file(s) already in "
                                f"{_short(Path(skipped[0]).parent)}." if skipped else
                                "Nothing to export: no completed recording in the selected wells.")

    def _on_file_progress(self, done: int, total: int) -> None:
        self.status.setText(f"File {self.progress.value() + 1} / {self.progress.maximum()}: "
                            f"frame {done} / {total}")

    def _on_file_done(self, result: ConvertResult) -> None:
        self.progress.setValue(self.progress.value() + 1)
        self.log.appendPlainText(result.summary())

    def _on_finished(self, written: int, error: str) -> None:
        if error:
            self.log.appendPlainText(f"Export stopped: {error}")
        self.status.setText(f"Done: {written} file(s) written." if not error else "Export failed.")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _on_thread_exited(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None

    # ----------------------------------------------------------- lifecycle

    def reject(self) -> None:
        """Close button, Esc and the window's X all end here (QDialog routes
        closeEvent through reject). Stop a running export first, then hide."""
        if self.is_running:
            self.worker.cancel()
            self.worker.wait()
        super().reject()


def _short(path: Path) -> str:
    """`~/...` for paths under the home folder (what the dialog displays)."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)
