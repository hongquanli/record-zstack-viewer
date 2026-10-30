"""The "Export mp4…" dialog: plate matrix, FOVs, output folder, contrast,
resolution, fps, overwrite, progress.

Planning and encoding both run on a `QThread` so the window and its live
following stay responsive; one file at a time (`export_all`). Cancel stops at
the next frame and discards the file in progress. Output defaults to
`~/Downloads/videos-<experiment name>/`; paths are shown with `~` for home.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                            QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                            QLabel, QLineEdit, QPlainTextEdit, QProgressBar,
                            QPushButton, QTableWidget, QTableWidgetItem,
                            QVBoxLayout, QWidget)

from record_zstack_viewer.export.mp4 import (CANCELLED, MAX_SIDE_PX, ExportResult,
                                             default_out_dir, export_all, output_size, plan)
from record_zstack_viewer.model.experiment import ExperimentModel
from record_zstack_viewer.model.geometry import PLATE_FORMATS, well_name_to_rc

_CELL_PX = (10, 26)        # min / max side of a matrix cell
_GRID_WIDTH_PX = 480       # a matrix fits this width for every plate format
_FOV_COLS = 12             # FOV indices wrap to rows of this many
_ABSENT = QColor(235, 235, 235)


@dataclass(frozen=True)
class ExportOptions:
    out_dir: Path
    wells: list[str]
    fovs: set[int] | None
    fps: float | None
    clims: tuple[float, float] | None
    max_side: int | None
    overwrite: bool


class ExportWorker(QThread):
    """Plans and runs an export off the UI thread; `cancel()` stops at the next frame."""

    planned = Signal(int)                   # number of files to write
    file_progress = Signal(int, int)        # frames done, frames total (current file)
    file_done = Signal(object)              # ExportResult
    finished_all = Signal(int, str)         # files written, error text ("" when none)

    def __init__(self, model: ExperimentModel, options: ExportOptions) -> None:
        super().__init__()
        self._model, self._options = model, options
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401 (QThread entry point)
        o, written = self._options, 0
        try:
            jobs = plan(self._model, o.out_dir, wells=o.wells, fovs=o.fovs, overwrite=o.overwrite)
            self.planned.emit(len(jobs))
            for result in export_all(self._model, jobs, fps=o.fps, clims=o.clims,
                                     max_side=o.max_side, on_progress=self.file_progress.emit,
                                     cancelled=lambda: self._cancel):
                written += int(result.ok)
                self.file_done.emit(result)
        except Exception as exc:  # a torn zarr.json or an unreadable store: report, never hang
            self.finished_all.emit(written, str(exc) or type(exc).__name__)
            return
        self.finished_all.emit(written, "")


# ----------------------------------------------------------------- selectors

class MatrixSelector(QTableWidget):
    """A grid of named cells with table selection: click, shift-click a
    rectangle, drag, ctrl/cmd-click to add. Cells not in `present` are greyed
    and unselectable. `selected()` returns the selected names in grid order."""

    def __init__(self, names: dict[tuple[int, int], str], rows: int, cols: int,
                 row_labels: list[str], present: set[str], selected: set[str],
                 show_text: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(rows, cols, parent)
        self._name = dict(names)
        for r in range(rows):
            for c in range(cols):
                item = QTableWidgetItem()
                name = self._name.get((r, c))
                if name in present:
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                    item.setToolTip(name)
                    if show_text:
                        item.setText(name)
                        item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setFlags(Qt.NoItemFlags)
                    item.setBackground(_ABSENT)
                self.setItem(r, c, item)
        self.setHorizontalHeaderLabels([str(c + 1) for c in range(cols)])
        self.setVerticalHeaderLabels(row_labels)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side = max(_CELL_PX[0], min(_CELL_PX[1], _GRID_WIDTH_PX // cols))
        for header in (self.horizontalHeader(), self.verticalHeader()):
            header.setDefaultSectionSize(side)
            header.setMinimumSectionSize(side)
            font = header.font()
            font.setPointSize(7 if cols > 24 else 9)
            header.setFont(font)
        self.setFixedSize(self.verticalHeader().width() + side * cols + 2,
                          self.horizontalHeader().height() + side * rows + 2)
        for rc, name in self._name.items():
            if name in selected:
                self.item(*rc).setSelected(True)

    def selected(self) -> list[str]:
        cells = sorted((i.row(), i.column()) for i in self.selectedItems())
        return [self._name[rc] for rc in cells]


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


def well_selector(wells: list[str], plate_format: str, selected: set[str],
                  parent: QWidget | None = None) -> MatrixSelector:
    """The plate as a matrix; non-well region ids get an extra bottom row."""
    rows, cols, other = grid_shape(wells, plate_format)
    names = {(r, c): f"{_row_label(r)}{c + 1}" for r in range(rows) for c in range(cols)}
    names.update({(rows, i): name for i, name in enumerate(other)})
    labels = [_row_label(r) for r in range(rows)] + (["other"] if other else [])
    return MatrixSelector(names, rows + bool(other), max(cols, len(other)), labels,
                          set(wells), selected, show_text=bool(other), parent=parent)


def fov_selector(n_fovs: int, parent: QWidget | None = None) -> MatrixSelector:
    """FOV indices 0..n-1 (the same grid in every well), wrapped to rows; all selected."""
    cols = min(n_fovs, _FOV_COLS)
    rows = math.ceil(n_fovs / cols)
    names = {(k // cols, k % cols): str(k) for k in range(n_fovs)}
    present = set(names.values())
    matrix = MatrixSelector(names, rows, cols, [""] * rows, present, present,
                            show_text=True, parent=parent)
    matrix.horizontalHeader().hide()  # the cells carry the FOV index themselves
    matrix.verticalHeader().hide()
    matrix.setFixedSize(matrix.width() - matrix.verticalHeader().width(),
                        matrix.height() - matrix.horizontalHeader().height())
    return matrix


# -------------------------------------------------------------------- dialog

class ExportDialog(QDialog):
    """Public surface for tests: `.wells`, `.fovs` (None when every well has one
    FOV), `.out_dir`, `.contrast`, `.full_res`, `.fps`, `.overwrite`,
    `.start_button`, `.cancel_button`, `.log`, `.status`, `.start()`, `.worker`,
    `.options()`."""

    def __init__(self, model: ExperimentModel, current_well: str | None = None,
                 screen_clims: tuple[float, float] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self.worker: ExportWorker | None = None
        self.setWindowTitle("Export recordings to mp4")
        self.setMinimumWidth(600)
        form = QFormLayout(self)

        # -- wells: the plate matrix -------------------------------------------
        wells = model.wells()
        self.wells = well_selector(wells, model.plate_format(),
                                   {current_well} if current_well in wells else set(wells), self)
        form.addRow("Wells", self._with_all_none(self.wells, "click · shift-click a range · drag"))

        # -- FOVs per well: only when a well has more than one ----------------
        n_fovs = max((len(model.fovs(w)) for w in wells), default=1)
        self.fovs = fov_selector(n_fovs, self) if n_fovs > 1 else None
        if self.fovs is not None:
            form.addRow("FOVs per well", self._with_all_none(self.fovs, "same indices in every well"))

        # -- output folder -------------------------------------------------------
        self.out_dir = QLineEdit(_short(default_out_dir(model.root)), self)
        browse = QPushButton("Browse…", self)
        browse.clicked.connect(self._browse)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(browse)
        form.addRow("Output folder", out_row)

        # -- contrast: on screen (default when a recording is displayed) or auto --
        self.contrast = QComboBox(self)
        if screen_clims is not None:
            lo, hi = screen_clims
            self.contrast.addItem(f"As on screen ({lo:.0f} – {hi:.0f})", screen_clims)
        self.contrast.addItem("Auto per video (1 – 99.9 percentile)", None)
        self.contrast.setToolTip("One fixed window for the whole video. 'As on screen' applies the "
                                 "displayed recording's limits to every selected well.")
        form.addRow("Contrast", self.contrast)

        # -- resolution: show what comes out; allow full size over the cap --------
        self._frame_hw = _frame_size(model, wells)
        self.resolution = QLabel(self)
        self.full_res = QCheckBox(f"Keep full resolution (over {MAX_SIDE_PX} px: may not play in QuickTime or Slack)", self)
        self.full_res.toggled.connect(self._update_resolution)
        res_box = QVBoxLayout()
        res_box.addWidget(self.resolution)
        res_box.addWidget(self.full_res)
        form.addRow("Resolution", res_box)
        self._update_resolution()

        # -- fps / overwrite -------------------------------------------------------
        self.fps = QDoubleSpinBox(self)
        self.fps.setRange(0.0, 1000.0)
        self.fps.setDecimals(2)
        self.fps.setSpecialValueText("real time (as recorded)")
        self.fps.setValue(0.0)
        form.addRow("Playback fps", self.fps)
        self.overwrite = QCheckBox("Overwrite existing", self)
        form.addRow("", self.overwrite)

        # -- progress, log, buttons ------------------------------------------------
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
        form.addRow(self.progress)
        form.addRow(self.status)
        form.addRow(self.log)
        form.addRow(buttons)

    def _with_all_none(self, matrix: MatrixSelector, hint: str) -> QVBoxLayout:
        all_btn = QPushButton("All", self)
        none_btn = QPushButton("None", self)
        all_btn.clicked.connect(matrix.selectAll)
        none_btn.clicked.connect(matrix.clearSelection)
        row = QHBoxLayout()
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addWidget(QLabel(hint, self))
        row.addStretch(1)
        box = QVBoxLayout()
        box.addWidget(matrix)
        box.addLayout(row)
        return box

    # ------------------------------------------------------------ state

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def options(self) -> ExportOptions:
        """What Export will do, as currently set in the dialog."""
        return ExportOptions(
            out_dir=Path(self.out_dir.text()).expanduser(),
            wells=self.wells.selected(),
            fovs={int(k) for k in self.fovs.selected()} if self.fovs is not None else None,
            fps=self.fps.value() or None,
            clims=self.contrast.currentData(),
            max_side=None if self.full_res.isChecked() else MAX_SIDE_PX,
            overwrite=self.overwrite.isChecked())

    def _update_resolution(self) -> None:
        if self._frame_hw is None:
            self.resolution.setText("unknown (no recording found)")
            self.full_res.hide()
            return
        h, w = self._frame_hw
        over_cap = max(w, h) > MAX_SIDE_PX
        self.full_res.setVisible(over_cap)
        out_w, out_h = output_size(w, h, None if self.full_res.isChecked() else MAX_SIDE_PX)
        text = f"{out_w} × {out_h}"
        if (out_w, out_h) != (w, h):
            text = f"{w} × {h} → {text}" + (" (downscaled to play anywhere)" if over_cap else "")
        self.resolution.setText(text)

    # ------------------------------------------------------------ actions

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", str(self.options().out_dir))
        if chosen:
            self.out_dir.setText(_short(Path(chosen)))

    def start(self) -> None:
        """Plan and encode the selection in the worker thread."""
        if self.is_running:
            return
        options = self.options()
        if not options.wells:
            self.status.setText("Select at least one well.")
            return
        if options.fovs is not None and not options.fovs:
            self.status.setText("Select at least one FOV.")
            return
        self.progress.setRange(0, 0)  # busy until planned
        self.status.setText(f"Planning export to {_short(options.out_dir)}…")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = ExportWorker(self._model, options)
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


# ------------------------------------------------------------------- helpers

def _short(path: Path) -> str:
    """`~/...` for paths under the home folder (what the dialog displays)."""
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _frame_size(model: ExperimentModel, wells: list[str]) -> tuple[int, int] | None:
    """`(height, width)` of the first recording found, or None."""
    for well in wells:
        for node in model.fovs(well):
            if node.has_recording:
                try:
                    return tuple(model.open_fov(well, node.fov).recording.shape[-2:])
                except Exception:
                    return None
    return None
