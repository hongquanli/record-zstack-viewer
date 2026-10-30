"""Task 5.3 — `MainWindow`: the app shell that wires navigator + viewer + live.

**Layout A** (locked UI decision): a `PlateNavigator` docked LEFT, the
`FovViewer` as the central widget under a live coverage header, and the
Recording | Z-stack mode toggle in the shell (channel controls come from ndv's
built-in LUT strip — no separate channel dock, and no ndv import here; ndv
stays confined to `ui/fov_viewer.py`).

One `default_context()` and one `ThumbnailCache` are built here and shared by
everything (model opens, navigator thumbnails, live invalidation), so decoded
bytes stay bounded by the single `cache_pool` of the Global Constraints.

**Live wiring.** A UI-thread `QTimer` drives `LiveEngine.tick()` (~1 s). The
app must NEVER call `LiveEngine.start()`: its background thread would race the
non-thread-safe `ExperimentModel.refresh()` against UI-thread model reads; the
QTimer keeps refresh and both callbacks on the UI thread. On `on_change`, a
changed FOV restyles its navigator cell, and — per the LiveEngine consumer
contract — the DISPLAYED FOV is **re-opened** (a held source stays stale) and
swapped in via the data-only `viewer.update_extent`, preserving the user's
slider index and contrast; any failure falls back to a full `set_datasets`
rebuild. `on_status` feeds the coverage header + the navigator's k/N ring.
"""
from __future__ import annotations

import logging
from pathlib import Path

from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import (QDockWidget, QHBoxLayout, QLabel, QMainWindow,
                            QPushButton, QVBoxLayout, QWidget)

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.cache.thumbnails import ThumbnailCache
from record_zstack_viewer.live.engine import LiveEngine
from record_zstack_viewer.model.experiment import ExperimentModel
from record_zstack_viewer.reader.adapter import FovDatasets
from record_zstack_viewer.ui.export_dialog import ExportDialog
from record_zstack_viewer.ui.fov_viewer import FovViewer
from record_zstack_viewer.ui.plate_navigator import PlateNavigator

logger = logging.getLogger(__name__)

_LIVE_INTERVAL_MS = 1000                 # ~1 s UI-thread tick
_MODES = ("recording", "zstack")         # toggle order; recording is the default
_MODE_LABELS = {"recording": "Recording", "zstack": "Z-stack"}


class MainWindow(QMainWindow):
    """The record+z-stack viewer window over one experiment `root` (Layout A).

    Public surface: `.navigator`, `.viewer`, `.coverage_label`,
    `.mode_buttons` (label buttons of the Recording | Z-stack toggle),
    `.set_mode(mode)`, and `.displayed` (the `(well, fov)` on display). The
    live engine is `._live`, driven only by the UI-thread `._live_timer`.
    """

    def __init__(self, root: str | Path, parent: QWidget | None = None,
                 *, context=None, controller=None) -> None:
        super().__init__(parent)
        self.root = Path(root)
        self._controller = controller
        self.setWindowTitle(f"record-zstack viewer — {self.root.name}")
        self.resize(1280, 800)

        # ONE shared context + ONE thumbnail cache (single bounded cache_pool):
        # the model's opens, the navigator's thumbnails, and every thumbnail
        # compute all draw from these same instances. An injected context (from
        # the ViewerController) is used as-is so ALL windows share one cache_pool.
        self._context = context if context is not None else default_context()
        self._thumbnails = ThumbnailCache(self.root, context=self._context)
        self._model = ExperimentModel(self.root, self._context)

        self._mode: str = _MODES[0]
        self._displayed: tuple[str, int] | None = None
        self._datasets: FovDatasets | None = None

        # -- central: coverage header + mode toggle above the FovViewer -------
        self.coverage_label = QLabel("", self)
        self.viewer = FovViewer(self)
        self.mode_buttons: dict[str, QPushButton] = {}
        bar = QHBoxLayout()
        bar.addWidget(self.coverage_label, 1)
        for mode in _MODES:
            btn = QPushButton(_MODE_LABELS[mode], self)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _=False, m=mode: self.set_mode(m))
            self.mode_buttons[mode] = btn
            bar.addWidget(btn)
        self.export_button = QPushButton("Export mp4…", self)
        self.export_button.clicked.connect(self.open_export_dialog)
        bar.addWidget(self.export_button)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addLayout(bar)
        layout.addWidget(self.viewer, 1)
        self.setCentralWidget(central)

        # -- left dock: the plate navigator (Layout A) -------------------------
        self.navigator = PlateNavigator(thumbnails=self._thumbnails)
        self.navigator.set_model(self._model)
        self._known_wells = set(self._model.wells())
        dock = QDockWidget("Plate", self)
        dock.setObjectName("plate_navigator_dock")
        dock.setWidget(self.navigator)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.navigator.fov_selected.connect(self._on_fov_selected)

        self._sync_mode_buttons()
        self._seed_header()

        # -- live: UI-thread QTimer -> tick(); NEVER LiveEngine.start() --------
        # start()'s background thread would run the non-thread-safe
        # model.refresh() while the UI reads the model; the timer keeps
        # refresh + on_change/on_status all on this (the UI) thread.
        self._live = LiveEngine(
            self._model, on_change=self._on_live_change,
            on_status=self._on_live_status, thumbnail_cache=self._thumbnails,
            interval=_LIVE_INTERVAL_MS / 1000.0)
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(_LIVE_INTERVAL_MS)
        self._live_timer.timeout.connect(self._on_live_timer)
        self._live_timer.start()

        # Drop an experiment folder here to open it in a NEW window (via the
        # controller). WA_DeleteOnClose makes the window actually go away on
        # close so the controller's `destroyed`-hook deregisters it; closeEvent
        # stops the live timer FIRST, so no tick ever runs on a deleted window.
        self.setAcceptDrops(True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

    # ------------------------------ public API -------------------------------

    @property
    def displayed(self) -> tuple[str, int] | None:
        """The `(well, fov)` currently on display, or None before a selection."""
        return self._displayed

    def set_mode(self, mode: str) -> None:
        """Switch the Recording | Z-stack toggle to `mode`.

        With a FOV on display (and that phase present) the viewer switches
        immediately, carrying the shared `acq_t` timepoint across; before any
        selection the choice is just remembered for the first open. Selecting
        `mode` for a FOV that lacks that phase keeps the current display.
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if self._datasets is not None and getattr(self._datasets, mode) is None:
            self._sync_mode_buttons()              # un-toggle the impossible choice
            return
        self._mode = mode
        self._sync_mode_buttons()
        if self._datasets is not None and self.viewer.mode != mode:
            t = self.viewer.timepoint              # acq_t selector is shared
            self.viewer.set_datasets(self._datasets, mode)
            self.viewer.set_timepoint(t)

    def open_export_dialog(self) -> ExportDialog:
        """Open the zarr -> mp4 export dialog for this experiment (non-modal)."""
        well = self._displayed[0] if self._displayed is not None else None
        dialog = ExportDialog(self._model, current_well=well, parent=self)
        dialog.show()
        return dialog

    # ------------------------------- selection -------------------------------

    def _on_fov_selected(self, well: str, fov: int) -> None:
        """Navigator selection: open `(well, fov)` fresh (new auto-contrast)."""
        well, fov = str(well), int(fov)
        try:
            datasets = self._model.open_fov(well, fov)
        except Exception:
            logger.exception("open_fov(%s, %s) failed", well, fov)
            return
        mode = self._mode
        if getattr(datasets, mode) is None:        # fall back to the present phase
            mode = next((m for m in _MODES if getattr(datasets, m) is not None), None)
            if mode is None:
                logger.warning("FOV %s/%s has no readable phase", well, fov)
                return
        self.viewer.set_datasets(datasets, mode)
        self._datasets, self._displayed, self._mode = datasets, (well, fov), mode
        self._sync_mode_buttons()

    def _sync_mode_buttons(self) -> None:
        for mode, btn in self.mode_buttons.items():
            btn.setChecked(mode == self._mode)
            btn.setEnabled(self._datasets is None
                           or getattr(self._datasets, mode) is not None)

    # --------------------------------- live ----------------------------------

    def _on_live_timer(self) -> None:
        """One UI-thread tick; errors are logged, never kill the timer."""
        try:
            self._live.tick()
        except Exception:
            logger.exception("live tick failed; continuing")

    def _on_live_change(self, changed: set[tuple[str, int]]) -> None:
        """Per-FOV change fan-out (model.refresh() has already run this tick)."""
        wells_now = set(self._model.wells())
        if wells_now != self._known_wells:         # a brand-new well: redraw the map
            self.navigator.set_model(self._model)
            self._known_wells = wells_now
        for well, fov in changed:
            status = next(
                (n.status for n in self._model.fovs(well) if n.fov == fov), None)
            if status is not None:
                self.navigator.set_status(well, fov, status)
        if self._displayed is not None and tuple(self._displayed) in changed:
            self._refresh_displayed()

    def _refresh_displayed(self) -> None:
        """Data-only refresh of the FOV on display.

        Per the LiveEngine consumer contract the FOV is RE-OPENED (a held
        source binds its store handle and stays stale after refresh()), then
        swapped in via `update_extent` so the user's index and contrast
        survive. Any failure falls back to the full `set_datasets` path.
        """
        well, fov = self._displayed
        try:
            datasets = self._model.open_fov(well, fov)
            source = getattr(datasets, self._mode)
            if source is None:
                raise ValueError(f"{self._mode!r} phase missing for {well}/{fov}")
            self.viewer.update_extent(source)
            self._datasets = datasets              # mode toggles use fresh sources
            self._sync_mode_buttons()              # a phase appearing mid-run enables its toggle
        except Exception:
            logger.exception("data-only refresh failed for %s FOV %s; "
                             "rebuilding the view", well, fov)
            self._on_fov_selected(well, fov)       # full set_datasets fallback

    def _on_live_status(self, status: dict) -> None:
        """Coverage header + navigator k/N ring from a LiveEngine status dict."""
        done = int(status.get("coverage_done") or 0)
        total = status.get("coverage_total")
        total = int(total) if total else self._model_total()
        well, fov = status.get("current_well"), status.get("current_fov")
        acquiring = bool(status.get("acquiring"))
        current = ((str(well), int(fov))
                   if acquiring and well is not None and fov is not None else None)
        self.navigator.update_coverage(done, total, current)
        if status.get("aborted"):
            text = f"Aborted — {done}/{total} FOVs"
        elif status.get("done"):
            text = f"Run complete — {done}/{total} FOVs"
        elif current is not None:
            text = f"Acquiring {current[0]} · FOV {current[1]} — {done}/{total} FOVs"
        else:
            text = f"{done}/{total} FOVs"
        self.coverage_label.setText(text)

    def _model_total(self) -> int:
        """Expected FOV total from the model (expected-per-well, else present)."""
        return sum(self._model.fov_count_expected(w) or len(self._model.fovs(w))
                   for w in self._model.wells())

    def _seed_header(self) -> None:
        """Initial coverage header from the model, before any live event."""
        done = sum(1 for w in self._model.wells()
                   for n in self._model.fovs(w) if n.status == "complete")
        self.coverage_label.setText(f"{done}/{self._model_total()} FOVs")

    # ------------------------------ drag-and-drop ----------------------------

    def dragEnterEvent(self, event):  # noqa: N802 (Qt naming)
        """Accept a drop that carries at least one local-file URL."""
        md = event.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        """Open each dropped local path that is an experiment (in a new window)."""
        md = event.mimeData()
        self._handle_dropped_paths(
            [u.toLocalFile() for u in md.urls() if u.isLocalFile()])
        event.acceptProposedAction()

    def _handle_dropped_paths(self, paths: list[str]) -> None:
        """Open every dropped path that `is_experiment` via the controller.

        Each open is a NEW window sharing the controller's one cache_pool. With
        no controller (e.g. a standalone window) this is a clean no-op.
        """
        from record_zstack_viewer.ui.controller import is_experiment  # lazy: avoid a cycle
        if self._controller is None:
            return
        for p in paths:
            root = Path(p)
            if is_experiment(root):
                self._controller.open_experiment(root)   # -> NEW window

    # -------------------------------- lifecycle ------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # Stop the live QTimer BEFORE the window is deleted (WA_DeleteOnClose
        # schedules deletion once this accepted close returns), so a closing or
        # closed window never ticks. The subsequent `destroyed` signal lets the
        # ViewerController deregister this window from its registry.
        self._live_timer.stop()
        for dialog in self.findChildren(ExportDialog):   # stop a running export cleanly
            dialog.close()
        super().closeEvent(event)
