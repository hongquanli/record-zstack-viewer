"""Task 5.2 — `PlateNavigator`: hue-coded plate map + adaptive per-well FOV strip.

A pure-Qt raster navigator (QGraphicsView/QGraphicsScene — no ndv, no OpenGL;
ndv stays confined to `ui/fov_viewer.py` per Global Constraints):

- **Tier 1, plate map.** The full plate grid comes from
  `ExperimentModel.plate_format()` + `geometry.PLATE_FORMATS` (falling back to a
  grid just covering the present wells when the format is unknown, e.g. a
  single-region scan; region ids that aren't well names get an extra row below
  the plate). Each present well is one ellipse hue-coded by the aggregate of its
  FOV statuses — blue=complete, amber=in_progress, grey=empty — and *aborted*
  keeps the grey fill but adds a red outline **and a ✕ cross**, a shape cue that
  stays distinguishable from "complete" for colorblind users (locked UI
  decision). Wheel-zoom (anchored under the cursor) + hand-drag pan + per-cell
  tooltips (`well · status · done/expected FOVs`) keep 384/1536 plates usable.
  A release-near-press click (never a drag) selects the well under the cursor.

- **Tier 2, adaptive FOV strip.** Selecting a well with **more than one** FOV
  populates a horizontal thumbnail strip below the plate (icons via
  `ThumbnailCache.get` — this module never decodes frames itself); clicking a
  thumbnail emits `fov_selected(well, fov)`. Selecting a **single-FOV** well
  skips the strip entirely and emits `fov_selected(well, its_one_fov)` right
  away (no 1-item strip, per the visual-companion decision).

- **Live.** `set_status(well, fov, status)` restyles exactly that well's
  existing cell (and, when visible, the strip entry) — no scene rebuild — and
  `update_coverage(done, total, current)` drives the k/N header plus a dashed
  ring marking the well currently being acquired. Both are incremental so a
  live engine can call them per event on big plates.

Model reads are `wells()` / `fovs(well)` / `fov_count_expected(well)` /
`plate_format()` only — building the map is metadata-cheap; pixels are touched
solely through `ThumbnailCache` (one representative plane per thumbnail, and
only for the selected well's strip).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from qtpy.QtCore import QPoint, QSize, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QPainter,
                        QPainterPath, QPen, QPixmap)
from qtpy.QtWidgets import (QApplication, QGraphicsEllipseItem,
                            QGraphicsPathItem, QGraphicsScene, QGraphicsView,
                            QLabel, QListWidget, QListWidgetItem, QVBoxLayout,
                            QWidget)

from record_zstack_viewer.cache.thumbnails import ThumbnailCache
from record_zstack_viewer.model.geometry import PLATE_FORMATS, well_name_to_rc

# Global Constraints hue coding: blue done / amber acquiring / grey not-yet /
# red-✕ aborted. Aborted keeps the grey FILL — its cue is the red outline + ✕
# cross, so it never reads as "complete" to red/green- or blue-blind users.
STATUS_FILL = {
    "complete": "#3b82f6",     # blue
    "in_progress": "#f59e0b",  # amber
    "empty": "#b7bcc4",        # grey
    "aborted": "#b7bcc4",      # grey fill + ABORTED_OUTLINE pen + ✕ shape cue
}
ABORTED_OUTLINE = "#e5484d"

_SELECTION_COLOR = "#111827"   # solid ring around the selected well
_CURRENT_COLOR = "#f59e0b"     # dashed ring around the well being acquired
_GRID_PEN = "#c9cdd4"          # outline of absent grid positions / quiet wells
_LABEL_COLOR = "#6b7280"       # row/column header text

_PITCH = 20.0                  # scene units per well pitch (uniform in x and y)
_WELL_D = 16.0                 # well body diameter
_RING_D = _WELL_D + 5.0        # selection/current ring diameter
_WELL_KEY = 0                  # QGraphicsItem data key carrying the well name
_MIN_SCALE, _MAX_SCALE = 0.02, 60.0
_THUMB_EDGE = 84               # strip icon edge (thumbnails are ~128-256 px)


def _row_letters(row: int) -> str:
    """0-based row index -> spreadsheet letters (0->A ... 25->Z, 26->AA ...);
    the inverse of `well_name_to_rc`'s letter math."""
    n, out = row + 1, ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _rc_to_name(row: int, col: int) -> str:
    """0-based ``(row, col)`` -> well label (``(0, 0)`` -> ``"A1"``)."""
    return f"{_row_letters(row)}{col + 1}"


def _aggregate(statuses: Iterable[str]) -> str:
    """One well hue from its FOV statuses: any aborted -> aborted; all complete
    -> complete; any progress at all -> in_progress; else empty."""
    ss = set(statuses)
    if "aborted" in ss:
        return "aborted"
    if ss == {"complete"}:
        return "complete"
    if "in_progress" in ss or "complete" in ss:
        return "in_progress"
    return "empty"


def _cosmetic_pen(color: str, width: int = 1, style=Qt.SolidLine) -> QPen:
    """A pen whose on-screen width ignores the view zoom (cosmetic)."""
    pen = QPen(QColor(color), width, style)
    pen.setCosmetic(True)
    return pen


@dataclass
class _WellCell:
    """One grid position's graphics: the hue-coded body + the aborted ✕ cue
    (a child of the body, so it moves/hides with it)."""

    well: str
    body: QGraphicsEllipseItem
    cross: QGraphicsPathItem


class _PlateView(QGraphicsView):
    """Zoom/pan canvas for the plate scene with click-vs-drag well selection.

    Wheel zooms about the cursor (clamped); `ScrollHandDrag` pans. A release
    within the platform drag threshold of its press is a *click*: the item under
    the cursor (or its parent chain — the ✕ is a child of the body) names the
    well via `_WELL_KEY` data and is handed to `on_well_clicked`. The view
    auto-fits the scene on resize until the user zooms explicitly.
    """

    def __init__(self, scene: QGraphicsScene, on_well_clicked: Callable[[str], None],
                 parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self._on_well_clicked = on_well_clicked
        self._press_pos: QPoint | None = None
        self._user_zoomed = False
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.Antialiasing, True)

    # -- zoom / fit -----------------------------------------------------------

    def zoom(self, factor: float) -> None:
        """Scale the view by `factor`, clamped to a sane absolute range; any
        explicit zoom turns the automatic fit-on-resize off."""
        current = self.transform().m11() or 1.0
        target = max(_MIN_SCALE, min(_MAX_SCALE, current * float(factor)))
        if abs(target - current) < 1e-12:
            return
        self._user_zoomed = True
        f = target / current
        self.scale(f, f)

    def reset_fit(self) -> None:
        """Re-enable auto-fit and fit the (new) scene now — used after rebuilds."""
        self._user_zoomed = False
        self._fit()

    def _fit(self) -> None:
        rect = self.scene().itemsBoundingRect() if self.scene() else None
        if rect and rect.isValid():
            m = _PITCH * 0.5
            self.fitInView(rect.adjusted(-m, -m, m, m), Qt.KeepAspectRatio)

    # -- events ---------------------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt override)
        delta = event.angleDelta().y()
        if delta:
            self.zoom(1.25 ** (delta / 120.0))
            event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if not self._user_zoomed:
            self._fit()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self._press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().mouseReleaseEvent(event)
        press, self._press_pos = self._press_pos, None
        if (event.button() != Qt.LeftButton or press is None
                or (event.pos() - press).manhattanLength()
                > QApplication.startDragDistance()):
            return                                     # a drag/pan, not a click
        item = self.itemAt(event.pos())
        while item is not None and item.data(_WELL_KEY) is None:
            item = item.parentItem()                   # ✕ / ring -> owning well
        if item is not None:
            self._on_well_clicked(str(item.data(_WELL_KEY)))


class PlateNavigator(QWidget):
    """Plate map + adaptive FOV strip; emits ``fov_selected(well, fov)``.

    `set_model(model)` (an `ExperimentModel`) builds the map; `select_well` is
    the click handler's public equivalent; `set_status` / `update_coverage`
    apply live updates incrementally. Pass a shared `ThumbnailCache` to reuse
    the app's cache/context, else one is created per model root.
    """

    fov_selected = Signal(str, int)

    def __init__(self, parent: QWidget | None = None,
                 thumbnails: ThumbnailCache | None = None) -> None:
        super().__init__(parent)
        self._injected_thumbs = thumbnails
        self._thumbs: ThumbnailCache | None = thumbnails
        self._model = None
        self._fov_status: dict[str, dict[int, str]] = {}   # well -> {fov: status}
        self._expected: dict[str, int] = {}                # well -> expected FOVs
        self._cells: dict[str, _WellCell] = {}             # every drawn grid cell
        self._selected_well: str | None = None
        self._selection_ring: QGraphicsEllipseItem | None = None
        self._current_ring: QGraphicsEllipseItem | None = None

        self._coverage_label = QLabel("Coverage —", self)
        self._scene = QGraphicsScene(self)
        self._view = _PlateView(self._scene, self.select_well, self)

        self._strip = QListWidget(self)                    # adaptive FOV strip
        self._strip.setViewMode(QListWidget.IconMode)
        self._strip.setFlow(QListWidget.LeftToRight)
        self._strip.setWrapping(False)
        self._strip.setMovement(QListWidget.Static)
        self._strip.setResizeMode(QListWidget.Adjust)
        self._strip.setUniformItemSizes(True)
        self._strip.setIconSize(QSize(_THUMB_EDGE, _THUMB_EDGE))
        self._strip.setFixedHeight(_THUMB_EDGE + 44)       # icon + label + scrollbar
        self._strip.setSelectionMode(QListWidget.SingleSelection)
        self._strip.itemClicked.connect(self._on_thumb_clicked)
        self._strip.setVisible(False)                      # shown only for >1 FOV

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._coverage_label)
        layout.addWidget(self._view, 1)
        layout.addWidget(self._strip)

    # ------------------------------ public API -------------------------------

    def set_model(self, model) -> None:
        """Show `model`'s plate: full rebuild of the map, statuses, coverage."""
        self._model = model
        if self._injected_thumbs is None:
            self._thumbs = ThumbnailCache(model.root)
        self._fov_status = {}
        self._expected = {}
        done = total = 0
        for well in model.wells():
            nodes = model.fovs(well)
            self._fov_status[well] = {int(n.fov): str(n.status) for n in nodes}
            expected = model.fov_count_expected(well)
            self._expected[well] = int(expected) if expected else len(nodes)
            done += sum(1 for n in nodes if n.status == "complete")
            total += self._expected[well]
        self._selected_well = None
        self._strip.clear()
        self._strip.setVisible(False)
        self._rebuild_scene()
        self.update_coverage(done, total, None)

    def select_well(self, well: str) -> None:
        """What a well click does: single-FOV wells emit `fov_selected(well,
        the_one_fov)` immediately (no strip); multi-FOV wells show the
        thumbnail strip instead. A well without model data is a no-op."""
        if self._model is None:
            return
        well = str(well)
        nodes = self._model.fovs(well)
        if not nodes:
            return                                          # absent/unknown well
        self._selected_well = well
        self._place_ring(self._selection_ring, well)
        if len(nodes) > 1:
            self._populate_strip(well, nodes)
            self._strip.setVisible(True)
        else:
            self._strip.clear()
            self._strip.setVisible(False)                   # never a 1-item strip
            self.fov_selected.emit(well, int(nodes[0].fov))

    def set_status(self, well: str, fov: int, status: str) -> None:
        """Live per-FOV status update: restyle that well's existing cell (and
        its visible strip entry) in place — no scene rebuild."""
        if status not in STATUS_FILL:
            raise ValueError(
                f"unknown status {status!r}; expected one of {sorted(STATUS_FILL)}")
        well, fov = str(well), int(fov)
        self._fov_status.setdefault(well, {})[fov] = status
        self._restyle(well)
        if well == self._selected_well and self._strip.isVisibleTo(self):
            self._refresh_strip_item(well, fov, status)

    def update_coverage(self, done: int, total: int, current: tuple | None) -> None:
        """Live coverage header (`done/total`) + a dashed ring on the well
        currently being acquired (`current = (well, fov)` or None)."""
        text = f"Coverage {int(done)}/{int(total)}"
        if current:
            text += f" · acquiring {current[0]} · FOV {int(current[1])}"
        self._coverage_label.setText(text)
        self._place_ring(self._current_ring, str(current[0]) if current else None)

    # ------------------------------ scene build ------------------------------

    def _rebuild_scene(self) -> None:
        self._scene.clear()                                 # deletes old items
        self._cells = {}
        self._selection_ring = self._current_ring = None
        if self._model is None:
            return
        parsed: dict[str, tuple[int, int]] = {}
        extras: list[str] = []                              # non-well region ids
        for well in self._model.wells():
            try:
                parsed[well] = well_name_to_rc(well)
            except ValueError:
                extras.append(well)
        fmt = PLATE_FORMATS.get(self._model.plate_format())
        rows, cols = (fmt.rows, fmt.cols) if fmt else (0, 0)
        for r, c in parsed.values():                        # unknown format (or a
            rows, cols = max(rows, r + 1), max(cols, c + 1)  # well off the plate)

        font = QFont()
        font.setPixelSize(int(_PITCH * 0.45))
        for r in range(rows):
            self._add_label(_row_letters(r), 0.5 * _PITCH, (r + 1.5) * _PITCH, font)
        for c in range(cols):
            self._add_label(str(c + 1), (c + 1.5) * _PITCH, 0.5 * _PITCH, font)
        for r in range(rows):
            for c in range(cols):
                self._add_cell(_rc_to_name(r, c), r, c)
        ecols = cols or max(1, min(len(extras), 8))
        for i, well in enumerate(extras):                   # flat rows below plate
            self._add_cell(well, rows + 1 + i // ecols, i % ecols)

        for well in self._cells:
            self._restyle(well)
        self._selection_ring = self._make_ring(_SELECTION_COLOR, Qt.SolidLine)
        self._current_ring = self._make_ring(_CURRENT_COLOR, Qt.DashLine)
        self._view.reset_fit()

    def _add_label(self, text: str, cx: float, cy: float, font: QFont) -> None:
        item = self._scene.addSimpleText(text, font)
        item.setBrush(QBrush(QColor(_LABEL_COLOR)))
        rect = item.boundingRect()
        item.setPos(cx - rect.width() / 2.0, cy - rect.height() / 2.0)

    def _add_cell(self, well: str, r: int, c: int) -> None:
        body = QGraphicsEllipseItem(-_WELL_D / 2, -_WELL_D / 2, _WELL_D, _WELL_D)
        body.setPos((c + 1.5) * _PITCH, (r + 1.5) * _PITCH)
        body.setData(_WELL_KEY, well)
        self._scene.addItem(body)
        arm = _WELL_D * 0.28                                # the aborted ✕ cue
        path = QPainterPath()
        path.moveTo(-arm, -arm)
        path.lineTo(arm, arm)
        path.moveTo(-arm, arm)
        path.lineTo(arm, -arm)
        cross = QGraphicsPathItem(path, body)
        cross.setPen(_cosmetic_pen(ABORTED_OUTLINE, 2))
        cross.setVisible(False)
        self._cells[well] = _WellCell(well, body, cross)

    def _make_ring(self, color: str, style) -> QGraphicsEllipseItem:
        ring = QGraphicsEllipseItem(-_RING_D / 2, -_RING_D / 2, _RING_D, _RING_D)
        ring.setPen(_cosmetic_pen(color, 2, style))
        ring.setBrush(QBrush(Qt.NoBrush))
        ring.setZValue(10)
        ring.setVisible(False)
        self._scene.addItem(ring)
        return ring

    def _place_ring(self, ring: QGraphicsEllipseItem | None, well: str | None) -> None:
        """Move `ring` onto `well`'s cell (or hide it for None/unknown wells).
        The ring adopts the well's `_WELL_KEY` data so a click landing on it
        still resolves to the well underneath."""
        if ring is None:
            return
        cell = self._cells.get(well) if well else None
        if cell is None:
            ring.setVisible(False)
            return
        ring.setPos(cell.body.pos())
        ring.setData(_WELL_KEY, cell.well)
        ring.setVisible(True)

    # ------------------------------ cell styling -----------------------------

    def _restyle(self, well: str) -> None:
        """Re-derive one cell's hue/outline/✕/tooltip from its FOV statuses."""
        cell = self._cells.get(well)
        if cell is None:
            return                                          # off-grid live well
        statuses = self._fov_status.get(well)
        if not statuses:                                    # grid position, no data
            cell.body.setBrush(QBrush(Qt.NoBrush))
            cell.body.setPen(_cosmetic_pen(_GRID_PEN))
            cell.cross.setVisible(False)
            cell.body.setToolTip(well)
            return
        agg = _aggregate(statuses.values())
        cell.body.setBrush(QBrush(QColor(STATUS_FILL[agg])))
        aborted = agg == "aborted"
        cell.body.setPen(_cosmetic_pen(ABORTED_OUTLINE if aborted else _GRID_PEN,
                                       2 if aborted else 1))
        cell.cross.setVisible(aborted)
        done = sum(1 for s in statuses.values() if s == "complete")
        expected = max(self._expected.get(well, 0), len(statuses))
        cell.body.setToolTip(f"{well} · {agg} · {done}/{expected} FOVs")

    # ------------------------------- FOV strip -------------------------------

    def _populate_strip(self, well: str, nodes) -> None:
        self._strip.clear()
        for node in nodes:
            fov = int(node.fov)
            item = QListWidgetItem(QIcon(self._thumb_pixmap(well, fov)), f"FOV {fov}")
            item.setData(Qt.UserRole, fov)
            item.setToolTip(f"{well} · FOV {fov} · {node.status}")
            self._strip.addItem(item)

    def _refresh_strip_item(self, well: str, fov: int, status: str) -> None:
        """Keep the visible strip in sync with a live status change: refresh the
        matching entry's tooltip and re-pull its thumbnail (a no-op image-wise
        unless the cache was invalidated for that FOV)."""
        for i in range(self._strip.count()):
            item = self._strip.item(i)
            if item.data(Qt.UserRole) == fov:
                item.setToolTip(f"{well} · FOV {fov} · {status}")
                item.setIcon(QIcon(self._thumb_pixmap(well, fov)))
                return

    def _on_thumb_clicked(self, item: QListWidgetItem) -> None:
        if self._selected_well is None or item is None:
            return
        fov = item.data(Qt.UserRole)
        if fov is not None:
            self.fov_selected.emit(self._selected_well, int(fov))

    def _thumb_pixmap(self, well: str, fov: int) -> QPixmap:
        """`ThumbnailCache.get` -> grayscale QPixmap; any failure (missing
        store, unreadable PNG) yields a flat placeholder — the strip must
        tolerate partial data (contract: always have a fallback)."""
        try:
            arr = np.ascontiguousarray(self._thumbs.get(well, fov), dtype=np.uint8)
            h, w = arr.shape
            image = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
            return QPixmap.fromImage(image.copy())          # detach from the buffer
        except Exception:
            placeholder = QPixmap(_THUMB_EDGE, _THUMB_EDGE)
            placeholder.fill(QColor("#33373d"))
            return placeholder
