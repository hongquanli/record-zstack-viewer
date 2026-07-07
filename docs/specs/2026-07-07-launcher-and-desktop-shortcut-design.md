# Design: Drag-and-Drop Launcher + Ubuntu Desktop Shortcut

- **Date:** 2026-07-07
- **Author / approver:** Hongquan
- **Repo:** `record-zstack-viewer` (extends the completed viewer on `main` @ d4941ae)
- **Status:** Design approved 2026-07-07 — ready for implementation planning (Opus)
- **Builds on:** `docs/specs/2026-07-05-record-zstack-viewer-design.md` (the viewer)

---

## 1. Goal

Two additions so the viewer is usable without the command line:

1. **A drag-and-drop launcher** — launched with no path, the app shows a small persistent window ("drop an experiment folder here" + a Browse button). Dropping (or browsing to) a valid experiment folder opens a **viewer window** for it; the launcher **stays open** so you can open several experiments. Dropping a folder onto an **already-open viewer** opens **another viewer window**.
2. **An Ubuntu desktop-shortcut installer** — a script that installs a clickable app-grid/desktop entry (icon + `.desktop`) launching the drag-and-drop launcher, working on **Ubuntu 22.04, 24.04, and 26.04 (prioritize 26.04 / Wayland)**.

### Decisions locked (2026-07-07)

| Decision | Choice |
|---|---|
| Launcher model | **Persistent launcher spawns a viewer window per experiment** (launcher stays open) |
| Drop onto an open viewer | **Opens a new viewer window** (not switch-in-place) |
| Direct `PATH` launch | Opens the viewer **directly, without** also showing the launcher |
| Memory across windows | **One shared TensorStore context / `cache_pool`** for all windows (bounded total, regardless of window count) |
| Shortcut icon | **Generated in-repo** (PNG + SVG, well-plate/microscopy motif) |
| Ubuntu coverage | One script for **22.04 / 24.04 / 26.04**; 26.04 (Wayland) prioritized |

### In scope

- A `LauncherWindow`, an app-level `ViewerController` (owns the shared context + window registry + `open_experiment`), drop support on `MainWindow`, and `__main__` wiring.
- A generated icon committed to the repo.
- `scripts/install_desktop_shortcut.sh` (install + `--uninstall`).

### Out of scope (YAGNI)

- Recent-files / MRU list, thumbnails-in-launcher, multi-select drop of many folders at once.
- macOS/Windows shortcuts (Ubuntu only, per request).
- Switching an open viewer's experiment in place (explicitly chosen against — new window instead).
- Any change to the reader/model/live/cache internals beyond adding an optional injected `context`.

---

## 2. Components

### 2.1 `ViewerController` (`record_zstack_viewer/ui/controller.py`, new)

The single owner of app-wide state for the multi-window model. **No Qt widgets** beyond holding references.

- Constructs and holds **one** `default_context()` (the shared `cache_pool`) — passed to every window so all windows share one bounded decoded-pixel budget.
- `open_experiment(root: Path) -> MainWindow | None`: validate `root` (see §3); on success build a `MainWindow(root, context=self._context, controller=self)`, show it, register it, return it; on failure return `None` (caller shows the message). Registers a close hook to drop the reference.
- `windows() -> list[MainWindow]`; keeps references so windows aren't garbage-collected.
- Holds a reference to the `LauncherWindow` when one exists.
- **Does not** own thumbnail caches — each `MainWindow` builds its own `ThumbnailCache(root, context=shared)` (caches are per-experiment-root; the shared context still bounds pixel memory).

**Interface:**
```python
class ViewerController:
    def __init__(self, context=None): ...          # builds default_context() if None
    def open_experiment(self, root: Path) -> "MainWindow | None": ...
    def windows(self) -> list["MainWindow"]: ...
```

### 2.2 `LauncherWindow` (`record_zstack_viewer/ui/launcher.py`, new)

A small `QWidget`/`QMainWindow` (qtpy), persistent:
- A drop zone label ("Drop a Squid experiment folder here") + a **Browse…** `QPushButton` (opens `QFileDialog.getExistingDirectory`).
- `setAcceptDrops(True)`; `dragEnterEvent` accepts a single local-directory URL; `dropEvent` resolves the path → `controller.open_experiment(root)`.
- On invalid folder (validation fails): show an **inline message** in the window (and/or a transient status label) — do NOT open a window, do NOT crash. Stay open.
- Closing the launcher does not force-close viewers; Qt quits the app when the last window (launcher or viewer) closes (default `quitOnLastWindowClosed`).

### 2.3 `MainWindow` drop support (`record_zstack_viewer/ui/app.py`, modify)

- Add optional constructor params: `context=None` (use the shared one when provided; else build its own — preserves standalone use/tests) and `controller=None`.
- `setAcceptDrops(True)`; `dragEnterEvent`/`dropEvent`: a dropped local directory → `controller.open_experiment(root)` (→ **new window**). If no controller (e.g. constructed directly in a test), ignore drops gracefully.
- No change to the existing live-wiring, mode toggle, or the ndv path. Drops are handled at the `QWidget` level only (no ndv involvement).

### 2.4 `__main__` (`record_zstack_viewer/__main__.py`, modify)

- Parse args as today. Build `QApplication`, then a `ViewerController`.
- **No `path` (or `.` default when cwd is not a valid experiment):** create a `LauncherWindow(controller)` and show it.
- **`PATH` given and valid:** `controller.open_experiment(root)` directly (no launcher). If `PATH` is given but invalid → `parser.error` (as today).
- Decision: the bare `python -m record_zstack_viewer` (no arg) now opens the **launcher** rather than defaulting to cwd — the desktop shortcut relies on this.

### 2.5 Icon (`record_zstack_viewer/resources/`, new)

- A generated **SVG** (`icon.svg`) + a rasterized **PNG** (`icon-256.png`, and optionally 128/64) committed in-repo. A simple, clean well-plate/microscopy motif (a rounded-rect plate with a grid of wells, a couple highlighted). Generated by a small committed script (`resources/make_icon.py`) so it's reproducible, but the output PNG/SVG are committed too (so install needs no render step).

### 2.6 Ubuntu installer (`scripts/install_desktop_shortcut.sh`, new)

A bash script (POSIX-ish, bash) that:
1. **Resolves the interpreter** — the Python that has the package + deps installed. Default: the `python` on `PATH` that can `import record_zstack_viewer`; allow override via `--python /path/to/python`. (This is the crux: the `.desktop` `Exec` must use that interpreter, since deps live in a conda/venv, not system Python.) Verify it can import the package; abort with a clear message if not.
2. **Installs the icon** into `~/.local/share/icons/hicolor/256x256/apps/record-zstack-viewer.png` (+ scalable SVG into `.../scalable/apps/`), from the in-repo `resources/`.
3. **Writes** `~/.local/share/applications/record-zstack-viewer.desktop`:
   ```
   [Desktop Entry]
   Type=Application
   Name=Record + Z-Stack Viewer
   Comment=View Squid record + z-stack well-plate scans (OME-Zarr)
   Exec=<python> -m record_zstack_viewer
   Icon=record-zstack-viewer
   Terminal=false
   Categories=Science;Graphics;Viewer;
   StartupWMClass=record_zstack_viewer
   ```
   `Exec` launches the **launcher** (no path). `Icon=` uses the theme name (installed above) with the hicolor install so it resolves on all three releases.
4. Runs `update-desktop-database ~/.local/share/applications` (if present) and `gtk-update-icon-cache`/`xdg-icon-resource` best-effort (guarded — absence is non-fatal).
5. `--uninstall`: remove the `.desktop` + installed icons, refresh caches.
6. **Ubuntu-version handling:** the `.desktop`/hicolor mechanism is **identical and stable** across 22.04/24.04/26.04, so one code path serves all. The **only** cross-version nuance is the Qt display platform:
   - **26.04 (Wayland default)** and 24.04: Qt auto-detects Wayland. If the ndv/OpenGL canvas is blank under native Wayland, `QT_QPA_PLATFORM=xcb` (XWayland) is the reliable GL fallback.
   - The installer does **not** hard-force a platform by default (forcing `xcb` loses HiDPI/fractional scaling on 26.04): default `Exec=<python> -m record_zstack_viewer`. A **`--force-x11`** flag instead emits `Exec=env QT_QPA_PLATFORM=xcb <python> -m record_zstack_viewer` (the correct way to set an env var for a `.desktop` launch — there is no `Env=` key in the spec), for users who hit the blank-canvas case under native Wayland. The default install prints a one-line note pointing at `--force-x11`.
   - Optional `--desktop-copy`: also drop a copy in `~/Desktop/` and mark it trusted (`gio set … metadata::trusted true` + `chmod +x`) for double-click on GNOME (22.04/24.04); on 26.04 the app-grid entry is the primary, reliable path.

---

## 3. Experiment validation (shared)

A dropped/browsed/`PATH` folder is a valid experiment iff it looks like the contract's layout. Reuse the existing machinery — a lightweight check:
- It's a directory containing a `recording/` and/or `zarr/` subtree (the two store trees), OR `validate_experiment(root).ok` is True.
- Prefer a cheap structural probe (dir existence) over a full validate for responsiveness; fall back to `validate_experiment` for the message. On failure the launcher shows *why* (e.g. "no recording/ or zarr/ folder found").
- This lives in **one module-level helper** `is_experiment(root: Path) -> bool` in `ui/controller.py`, imported by the launcher, MainWindow-drop, and `__main__` so all three agree. (Cheap structural check; the launcher may additionally call `validate_experiment` to compose the failure message.)

---

## 4. Memory / lifecycle

- **One `cache_pool` for all windows** (shared context from the controller) → total decoded-pixel memory is bounded no matter how many viewers are open. Each window still has its own handle-LRU and per-root thumbnail cache (small, bounded).
- Windows are registered in the controller so they aren't GC'd; on `closeEvent` a window deregisters. Each window's `LiveEngine` QTimer stops on close.
- App exits when the last window (launcher or any viewer) closes (Qt default).

---

## 5. Error handling & edge cases

- **Invalid drop** (not a dir / not an experiment / permission error): launcher shows an inline message, stays open; MainWindow-drop is a no-op with a brief status. Never crash.
- **Non-local drop** (a URL that isn't a local file): ignored.
- **Multiple folders dropped at once:** open each valid one (v1: or take the first and note the rest — implementer's call, but don't crash). *Decision: open each valid folder in its own window.*
- **`PATH` given but invalid:** `parser.error` (unchanged).
- **Installer:** interpreter can't import the package → abort with the exact `pip install -e .` hint; missing `update-desktop-database`/`gio` → warn and continue; re-install overwrites cleanly; `--uninstall` idempotent.
- **Wayland blank canvas** (26.04): documented `QT_QPA_PLATFORM=xcb` fallback + `--force-x11`.

---

## 6. Testing

Testable here (offscreen, `QT_QPA_PLATFORM=offscreen`, macOS):
- **Controller:** `open_experiment(valid_root)` returns a `MainWindow` and registers it; opening N experiments yields N windows **sharing one context object** (assert identity); `open_experiment(invalid)` returns `None`, registers nothing.
- **Validation helper:** valid fixture → True; a dir with neither `recording/` nor `zarr/` → False; a non-existent path → False.
- **LauncherWindow:** a simulated drop of a valid fixture path calls `controller.open_experiment` with that root (spy); an invalid drop does not, and surfaces a message; launcher stays open.
- **MainWindow drop:** a drop calls `controller.open_experiment` (→ new window); with no controller, the drop is a no-op (no crash).
- **Icon:** the generator produces a valid PNG of the expected size (and a well-formed SVG).
- **`.desktop` generation:** the installer's file-content generation is factored so its output can be checked (correct `Exec` interpreter path, `Icon` name, install target paths). Test the pure content-builder (e.g. a small `python` helper or by asserting on the emitted file in a `--dry-run`/`--print` mode).

**Cannot test here (honest limit):** actual Ubuntu app-grid appearance, real Wayland/X11 GL rendering, and double-click behavior — those require an Ubuntu 22.04/24.04/26.04 machine. The installer will support a **`--dry-run`** that prints exactly what it would write (paths + file contents) so the user can review before running for real, and the script is written defensively. Final GUI verification is the user's on the target machines.

---

## 7. Files

```
record_zstack_viewer/
├── ui/
│   ├── controller.py     # NEW  ViewerController (shared context, window registry, open_experiment)
│   ├── launcher.py       # NEW  LauncherWindow (drop zone + Browse)
│   └── app.py            # MOD  MainWindow: optional context/controller, drop support
├── __main__.py           # MOD  no-path → launcher; PATH → direct open; via controller
└── resources/            # NEW  icon.svg, icon-256.png (+make_icon.py)
scripts/
└── install_desktop_shortcut.sh   # NEW  install/--uninstall/--dry-run/--python/--force-x11/--desktop-copy
tests/
├── test_controller.py    # NEW
├── test_launcher.py       # NEW
├── test_desktop_install.py# NEW  (.desktop content builder + icon)
└── test_app_smoke.py      # MOD  MainWindow drop
```

---

## 8. Why this meets the request

- **Drag-and-drop window:** a persistent launcher spawns a viewer per experiment; drops on a viewer open new windows; all share one bounded memory pool.
- **Desktop shortcut for 22.04/24.04/26.04 (26.04-priority):** one installer, stable `.desktop`/hicolor mechanism across releases, interpreter captured correctly, generated icon, Wayland handled with a documented XWayland fallback and `--force-x11`; `--dry-run` for review since GUI/desktop behavior can only be finally verified on Ubuntu.
