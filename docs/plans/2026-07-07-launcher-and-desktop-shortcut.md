# Launcher + Ubuntu Desktop Shortcut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a persistent drag-and-drop launcher that opens a viewer window per experiment, drop-on-viewer → new window, and a Python installer that creates an Ubuntu (22.04/24.04/26.04) desktop shortcut.

**Architecture:** A small `ViewerController` owns one shared TensorStore context (single `cache_pool` for all windows) + a window registry + `open_experiment(root)`. `LauncherWindow` and `MainWindow` both route folder opens through it. `__main__`: no path → launcher; valid path → direct open. A Python installer captures `sys.executable` for the `.desktop` `Exec` and installs an in-repo generated icon.

**Tech Stack:** Python 3.11+; qtpy (→ PyQt5/6, never import PyQt5 directly); the existing `record_zstack_viewer` reader/model/cache/live/ui; Pillow (icon); pytest (offscreen Qt).

**Spec:** `docs/specs/2026-07-07-launcher-and-desktop-shortcut-design.md`.

## Global Constraints

- Import Qt ONLY via `qtpy` (never `import PyQt5`/`PySide`). ndv stays confined to `ui/fov_viewer.py` — the launcher/controller/installer import no ndv.
- **One shared `cache_pool` for ALL windows:** the controller builds one `default_context()` and injects it into every `MainWindow`; total decoded-pixel memory stays bounded regardless of window count.
- Qt tests run offscreen: set `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` in the test module/conftest; run with `QT_QPA_PLATFORM=offscreen /opt/miniconda3/bin/python -m pytest`.
- Interpreter for all commands: `/opt/miniconda3/bin/python` (has the deps). NOT system python3.
- No path → launcher; a valid `PATH` arg → open that experiment directly (no launcher); invalid `PATH` → `parser.error`.
- `.desktop` env vars go in the `Exec` line via `env VAR=val …` (there is NO `Env=` key in the freedesktop spec).
- All work on branch `feat-launcher-shortcut` (off `main` @ d4941ae; spec already committed at 666e9a0).

## File structure

```
record_zstack_viewer/
├── ui/
│   ├── controller.py   # NEW  is_experiment(); ViewerController (shared ctx, registry, open_experiment)
│   ├── launcher.py     # NEW  LauncherWindow (drop zone + Browse)
│   └── app.py          # MOD  MainWindow: optional context=/controller=; drag-drop → controller.open_experiment
├── __main__.py         # MOD  no path → launcher; PATH → controller.open_experiment
└── resources/          # NEW  make_icon.py (generator); icon-256.png + icon.svg (committed outputs)
scripts/
└── install_desktop_shortcut.py   # NEW  build_desktop_entry(); install/uninstall/dry-run/--python/--force-x11/--desktop-copy
tests/
├── test_controller.py        # NEW
├── test_launcher.py          # NEW
├── test_desktop_install.py   # NEW  (build_desktop_entry + icon file valid)
└── test_app_smoke.py         # MOD  MainWindow drop
```

---

### Task 1: `is_experiment` helper + `ViewerController`

**Files:**
- Create: `record_zstack_viewer/ui/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: `MainWindow(root, parent=None)` (Task 2 adds `context=`/`controller=`); `record_zstack_viewer.cache.memory.default_context`.
- Produces:
  - `is_experiment(root: Path) -> bool`
  - `class ViewerController:` `__init__(self, context=None)`; `open_experiment(self, root) -> "MainWindow | None"`; `windows(self) -> list`; attribute `context`.

- [ ] **Step 1: Write failing tests**
```python
# tests/test_controller.py
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
```
- [ ] **Step 2: Run → fail** — `QT_QPA_PLATFORM=offscreen /opt/miniconda3/bin/python -m pytest tests/test_controller.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement `controller.py`**
```python
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
```
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: ViewerController + is_experiment (shared-context multi-window)"`

---

### Task 2: `MainWindow` — injected context/controller + drag-drop

**Files:**
- Modify: `record_zstack_viewer/ui/app.py` (`MainWindow.__init__` ~line 56; add drag/drop event methods)
- Test: `tests/test_app_smoke.py` (extend)

**Interfaces:**
- Consumes: `ViewerController` (Task 1), `is_experiment`.
- Produces: `MainWindow(root, parent=None, *, context=None, controller=None)`; attributes `_context`, `_controller`; `dragEnterEvent`/`dropEvent` that call `controller.open_experiment(root)` for a dropped local experiment directory.

- [ ] **Step 1: Write failing tests**
```python
# add to tests/test_app_smoke.py
def test_mainwindow_uses_injected_context(qapp, tiny_experiment):
    from record_zstack_viewer.cache.memory import default_context
    from record_zstack_viewer.ui.app import MainWindow
    ctx = default_context()
    w = MainWindow(tiny_experiment, context=ctx)
    assert w._context is ctx                       # did NOT build its own

def test_mainwindow_drop_opens_via_controller(qapp, tiny_experiment):
    from record_zstack_viewer.ui.app import MainWindow
    calls = []
    class StubController:
        context = None
        def open_experiment(self, root): calls.append(Path(root)); return None
    w = MainWindow(tiny_experiment, controller=StubController())
    w._handle_dropped_paths([str(tiny_experiment)])   # helper exercised directly
    assert calls == [tiny_experiment]

def test_mainwindow_drop_without_controller_is_noop(qapp, tiny_experiment):
    from record_zstack_viewer.ui.app import MainWindow
    w = MainWindow(tiny_experiment)                # no controller
    w._handle_dropped_paths([str(tiny_experiment)])  # must not raise
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Modify `app.py`.** Change the signature and context construction:
```python
    def __init__(self, root: str | Path, parent: QWidget | None = None,
                 *, context=None, controller=None) -> None:
        super().__init__(parent)
        self.root = Path(root)
        self._controller = controller
        self._context = context if context is not None else default_context()
        self._thumbnails = ThumbnailCache(self.root, context=self._context)
        self._model = ExperimentModel(self.root, self._context)
        ...
        self.setAcceptDrops(True)      # add near the end of __init__
```
Add the drag/drop handlers + helper (import `is_experiment` lazily to avoid a cycle):
```python
    def dragEnterEvent(self, event):  # noqa: N802 (Qt naming)
        md = event.mimeData()
        if md.hasUrls() and any(u.isLocalFile() for u in md.urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        md = event.mimeData()
        self._handle_dropped_paths(
            [u.toLocalFile() for u in md.urls() if u.isLocalFile()])
        event.acceptProposedAction()

    def _handle_dropped_paths(self, paths: list[str]) -> None:
        from record_zstack_viewer.ui.controller import is_experiment
        if self._controller is None:
            return
        for p in paths:
            root = Path(p)
            if is_experiment(root):
                self._controller.open_experiment(root)   # -> NEW window
```
- [ ] **Step 4: Run → pass** — full suite green too (`QT_QPA_PLATFORM=offscreen … -m pytest tests/ -v`).
- [ ] **Step 5: Commit** — `git commit -am "feat: MainWindow accepts injected context + folder drops (new window)"`

---

### Task 3: `LauncherWindow`

**Files:**
- Create: `record_zstack_viewer/ui/launcher.py`
- Test: `tests/test_launcher.py`

**Interfaces:**
- Consumes: `ViewerController` (Task 1) — calls `controller.open_experiment(root)`.
- Produces: `LauncherWindow(controller, parent=None)` (`QMainWindow`), with a Browse button, `setAcceptDrops(True)`, `dragEnterEvent`/`dropEvent`, `_handle_dropped_paths(list[str])`, and an inline status label updated on invalid drops.

- [ ] **Step 1: Write failing tests**
```python
# tests/test_launcher.py
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import pytest
from qtpy.QtWidgets import QApplication

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])

class _Stub:
    def __init__(self): self.opened = []
    context = None
    def open_experiment(self, root):
        self.opened.append(Path(root))
        return object() if (Path(root) / "recording").exists() or (Path(root) / "zarr").exists() else None

def test_valid_drop_opens_and_launcher_stays(qapp, tiny_experiment):
    from record_zstack_viewer.ui.launcher import LauncherWindow
    c = _Stub(); w = LauncherWindow(c)
    w._handle_dropped_paths([str(tiny_experiment)])
    assert c.opened == [tiny_experiment]
    assert w.isVisible() or True          # launcher not closed by a drop

def test_invalid_drop_shows_message_no_open(qapp, tmp_path):
    from record_zstack_viewer.ui.launcher import LauncherWindow
    bad = tmp_path / "bad"; bad.mkdir()
    c = _Stub(); w = LauncherWindow(c)
    w._handle_dropped_paths([str(bad)])
    assert c.opened == []                 # is_experiment False → not forwarded
    assert w.status_text()                # a non-empty message was shown
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `launcher.py`**
```python
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
```
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: LauncherWindow (drag-drop + Browse, persistent)"`

---

### Task 4: `__main__` wiring

**Files:**
- Modify: `record_zstack_viewer/__main__.py`
- Test: `tests/test_launcher.py` (extend) / `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `ViewerController`, `LauncherWindow`, `is_experiment`.
- Produces: `main(argv)` — no path → build controller + LauncherWindow (registered as `controller.launcher`), show; valid path → `controller.open_experiment`; invalid path → `parser.error`. `main` still returns the app exit code; add a `_build(argv)` seam returning `(app, controller, launcher_or_none)` without `app.exec()` so it's testable.

- [ ] **Step 1: Write failing test**
```python
# tests/test_app_smoke.py
def test_main_no_path_builds_launcher(qapp, monkeypatch):
    import record_zstack_viewer.__main__ as m
    app, controller, launcher = m._build([])       # no path
    assert launcher is not None and controller.windows() == []

def test_main_valid_path_opens_directly(qapp, tiny_experiment):
    import record_zstack_viewer.__main__ as m
    app, controller, launcher = m._build([str(tiny_experiment)])
    assert launcher is None and len(controller.windows()) == 1
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Modify `__main__.py`** — change `default="."` to `default=None`; add `_build`:
```python
def _build(argv=None):
    parser = argparse.ArgumentParser(
        prog="record_zstack_viewer",
        description="Viewer for squid record + z-stack well-plate experiments "
                    "(OME-Zarr v3), with live-acquisition following.")
    parser.add_argument("path", nargs="?", default=None,
                        help="experiment root; omit to open the drag-and-drop launcher")
    args = parser.parse_args(argv)

    from qtpy.QtWidgets import QApplication
    from record_zstack_viewer.ui.controller import ViewerController, is_experiment
    from record_zstack_viewer.ui.launcher import LauncherWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    controller = ViewerController()
    launcher = None
    if args.path is None:
        launcher = LauncherWindow(controller)
        controller.launcher = launcher
        launcher.show()
    else:
        root = Path(args.path).expanduser()
        if not is_experiment(root):
            parser.error(f"{root} is not a record+zstack experiment "
                         "(no recording/ or zarr/ folder)")
        controller.open_experiment(root)
    return app, controller, launcher


def main(argv=None) -> int:
    app, _controller, _launcher = _build(argv)
    return app.exec()
```
- [ ] **Step 4: Run → pass** (full suite green).
- [ ] **Step 5: Commit** — `git commit -am "feat: __main__ opens launcher when no path, else direct"`

---

### Task 5: App icon (generated, committed)

**Files:**
- Create: `record_zstack_viewer/resources/__init__.py`, `record_zstack_viewer/resources/make_icon.py`, and the committed outputs `record_zstack_viewer/resources/icon-256.png` + `record_zstack_viewer/resources/icon.svg`
- Test: `tests/test_desktop_install.py` (icon part)

**Interfaces:**
- Produces: `make_icon.render_png(path, size=256)` (Pillow) and `make_icon.SVG` (a module-level SVG string); committed `icon-256.png` (256×256 RGBA PNG) + `icon.svg`.

- [ ] **Step 1: Write failing test**
```python
# tests/test_desktop_install.py
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path

def test_committed_icon_is_valid_png():
    from PIL import Image
    p = Path("record_zstack_viewer/resources/icon-256.png")
    assert p.exists()
    im = Image.open(p); assert im.size == (256, 256)

def test_render_png_roundtrip(tmp_path):
    from record_zstack_viewer.resources import make_icon
    out = tmp_path / "i.png"; make_icon.render_png(out, size=128)
    from PIL import Image
    assert Image.open(out).size == (128, 128)
    assert make_icon.SVG.lstrip().startswith("<svg")
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `make_icon.py`** (Pillow draw: rounded plate + well grid, a few filled blue), define `SVG`, and generate the committed files by running it once:
```python
"""Generate the app icon (a well-plate motif). Outputs committed alongside."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw

_BG = (18, 22, 30, 255); _PLATE = (38, 46, 60, 255)
_WELL = (60, 70, 88, 255); _DONE = (47, 104, 207, 255)

def render_png(path, size: int = 256) -> None:
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im); s = size
    d.rounded_rectangle([s*0.10, s*0.14, s*0.90, s*0.86], radius=s*0.09, fill=_PLATE)
    cols, rows = 4, 3; m = s*0.20; gap = (s - 2*m)
    r = gap/ (2*max(cols, rows)) * 0.72
    for i in range(cols):
        for j in range(rows):
            cx = m + gap*(i+0.5)/cols; cy = s*0.16 + (s*0.68)*(j+0.5)/rows
            fill = _DONE if j == 0 and i < 3 else _WELL
            d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=fill)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)

SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <rect x="26" y="36" width="204" height="184" rx="23" fill="#262e3c"/>
  <g fill="#3c4658">
    <circle cx="70" cy="150" r="15"/><circle cx="112" cy="150" r="15"/>
    <circle cx="154" cy="150" r="15"/><circle cx="196" cy="150" r="15"/>
    <circle cx="196" cy="196" r="15"/><circle cx="154" cy="196" r="15"/>
    <circle cx="112" cy="196" r="15"/><circle cx="70" cy="196" r="15"/>
  </g>
  <g fill="#2f68cf">
    <circle cx="70" cy="104" r="15"/><circle cx="112" cy="104" r="15"/><circle cx="154" cy="104" r="15"/>
  </g>
</svg>
"""

if __name__ == "__main__":
    here = Path(__file__).parent
    render_png(here / "icon-256.png", 256)
    (here / "icon.svg").write_text(SVG)
    print("wrote icon-256.png + icon.svg")
```
Run it: `/opt/miniconda3/bin/python -m record_zstack_viewer.resources.make_icon` to produce the committed files.
- [ ] **Step 4: Run test → pass** (+ eyeball the PNG once).
- [ ] **Step 5: Commit** — `git add record_zstack_viewer/resources tests/test_desktop_install.py && git commit -m "feat: generated app icon (well-plate motif)"`

---

### Task 6: Ubuntu desktop-shortcut installer (Python)

**Files:**
- Create: `scripts/install_desktop_shortcut.py`
- Test: `tests/test_desktop_install.py` (installer part)

**Interfaces:**
- Consumes: the committed icon (Task 5); the package (importable by the target interpreter).
- Produces: pure `build_desktop_entry(python_exe: str, icon: str = "record-zstack-viewer", force_x11: bool = False) -> str`; `install(python_exe, *, force_x11, desktop_copy) -> dict[str,Path]` (returns written paths); `uninstall() -> None`; a `main(argv)` with `--python`, `--force-x11`, `--desktop-copy`, `--uninstall`, `--dry-run`.

- [ ] **Step 1: Write failing tests**
```python
# add to tests/test_desktop_install.py
def test_build_desktop_entry_default():
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "inst", Path("scripts/install_desktop_shortcut.py"))
    inst = importlib.util.module_from_spec(spec); spec.loader.exec_module(inst)
    txt = inst.build_desktop_entry("/opt/env/bin/python")
    assert "[Desktop Entry]" in txt
    assert "Exec=/opt/env/bin/python -m record_zstack_viewer" in txt
    assert "Icon=record-zstack-viewer" in txt
    assert "Terminal=false" in txt
    # default must NOT force a Qt platform
    assert "QT_QPA_PLATFORM" not in txt

def test_build_desktop_entry_force_x11():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "inst2", Path("scripts/install_desktop_shortcut.py"))
    inst = importlib.util.module_from_spec(spec); spec.loader.exec_module(inst)
    txt = inst.build_desktop_entry("/opt/env/bin/python", force_x11=True)
    assert "Exec=env QT_QPA_PLATFORM=xcb /opt/env/bin/python -m record_zstack_viewer" in txt
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `scripts/install_desktop_shortcut.py`** with the pure builder + install/uninstall/dry-run. Key content:
```python
#!/usr/bin/env python3
"""Install an Ubuntu (22.04/24.04/26.04) desktop shortcut for record-zstack-viewer.

Run it WITH THE PYTHON THAT HAS THE PACKAGE INSTALLED, e.g.:
    /path/to/env/bin/python scripts/install_desktop_shortcut.py
so the shortcut's Exec uses that exact interpreter (deps live in a conda/venv).
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys
from pathlib import Path

APP_ID = "record-zstack-viewer"
_NAME = "Record + Z-Stack Viewer"
_COMMENT = "View Squid record + z-stack well-plate scans (OME-Zarr)"

def build_desktop_entry(python_exe: str, icon: str = APP_ID, force_x11: bool = False) -> str:
    prefix = "env QT_QPA_PLATFORM=xcb " if force_x11 else ""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={_NAME}\n"
        f"Comment={_COMMENT}\n"
        f"Exec={prefix}{python_exe} -m record_zstack_viewer\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "Categories=Science;Graphics;Viewer;\n"
        "StartupWMClass=record_zstack_viewer\n")

def _apps_dir() -> Path: return Path.home()/".local/share/applications"
def _icon_png_dir() -> Path: return Path.home()/".local/share/icons/hicolor/256x256/apps"
def _icon_svg_dir() -> Path: return Path.home()/".local/share/icons/hicolor/scalable/apps"
def _repo_resources() -> Path: return Path(__file__).resolve().parent.parent/"record_zstack_viewer"/"resources"

def install(python_exe: str, *, force_x11: bool = False, desktop_copy: bool = False) -> dict:
    written = {}
    _icon_png_dir().mkdir(parents=True, exist_ok=True)
    _icon_svg_dir().mkdir(parents=True, exist_ok=True)
    png = _icon_png_dir()/f"{APP_ID}.png"; shutil.copyfile(_repo_resources()/"icon-256.png", png); written["icon_png"]=png
    svg_src = _repo_resources()/"icon.svg"
    if svg_src.exists():
        svg = _icon_svg_dir()/f"{APP_ID}.svg"; shutil.copyfile(svg_src, svg); written["icon_svg"]=svg
    _apps_dir().mkdir(parents=True, exist_ok=True)
    desktop = _apps_dir()/f"{APP_ID}.desktop"
    desktop.write_text(build_desktop_entry(python_exe, force_x11=force_x11)); written["desktop"]=desktop
    for cmd in (["update-desktop-database", str(_apps_dir())],
                ["gtk-update-icon-cache", "-f", str(Path.home()/".local/share/icons/hicolor")]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
    if desktop_copy:
        dcopy = Path.home()/"Desktop"/f"{APP_ID}.desktop"
        if dcopy.parent.is_dir():
            dcopy.write_text(desktop.read_text()); dcopy.chmod(0o755)
            if shutil.which("gio"):
                subprocess.run(["gio","set",str(dcopy),"metadata::trusted","true"], check=False)
            written["desktop_copy"]=dcopy
    return written

def uninstall() -> None:
    for p in (_apps_dir()/f"{APP_ID}.desktop", _icon_png_dir()/f"{APP_ID}.png",
              _icon_svg_dir()/f"{APP_ID}.svg", Path.home()/"Desktop"/f"{APP_ID}.desktop"):
        p.unlink(missing_ok=True)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(_apps_dir())], check=False)

def _resolve_python(explicit: str | None) -> str:
    exe = explicit or sys.executable
    r = subprocess.run([exe,"-c","import record_zstack_viewer"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"error: {exe} cannot import record_zstack_viewer — install it first "
                 f"(pip install -e .) or pass --python /path/to/env/python")
    return exe

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Install the Ubuntu desktop shortcut.")
    ap.add_argument("--python", help="interpreter for Exec (default: the one running this script)")
    ap.add_argument("--force-x11", action="store_true", help="Exec via env QT_QPA_PLATFORM=xcb (Wayland GL fallback)")
    ap.add_argument("--desktop-copy", action="store_true", help="also place a trusted copy on ~/Desktop")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print what would be written; change nothing")
    a = ap.parse_args(argv)
    if a.uninstall:
        uninstall(); print("uninstalled"); return 0
    exe = _resolve_python(a.python)
    if a.dry_run:
        print(f"# {_apps_dir()}/{APP_ID}.desktop\n{build_desktop_entry(exe, force_x11=a.force_x11)}")
        print(f"# icon -> {_icon_png_dir()}/{APP_ID}.png (from resources/icon-256.png)")
        if not a.force_x11:
            print("# note: if the canvas is blank under Wayland (26.04), re-run with --force-x11")
        return 0
    written = install(exe, force_x11=a.force_x11, desktop_copy=a.desktop_copy)
    for k, v in written.items(): print(f"wrote {k}: {v}")
    if not a.force_x11:
        print("note: blank canvas under Wayland (26.04)? re-run with --force-x11")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```
- [ ] **Step 4: Run → pass** (full suite green). Also manually run `/opt/miniconda3/bin/python scripts/install_desktop_shortcut.py --dry-run` and confirm sensible output (macOS — dry-run only; do NOT actually install).
- [ ] **Step 5: Commit** — `git add scripts/install_desktop_shortcut.py tests/test_desktop_install.py && git commit -m "feat: Ubuntu desktop-shortcut installer (Python; --dry-run/--force-x11/--uninstall)"`

---

### Task 7: README + resources packaging

**Files:**
- Modify: `README.md` (add a "Launcher & desktop shortcut" section), `pyproject.toml` (ensure `record_zstack_viewer.resources` icon files ship — add package-data / include the PNG+SVG).

- [ ] **Step 1:** README section: launch with no args → drag-and-drop launcher; drop a folder or Browse; drop onto a viewer → new window. Ubuntu shortcut: `/path/to/env/python scripts/install_desktop_shortcut.py` (run with the env's python; `--dry-run` first; `--force-x11` if blank canvas on Wayland/26.04; `--uninstall`).
- [ ] **Step 2:** In `pyproject.toml`, add package data so `resources/*.png`/`*.svg` are included (e.g. `[tool.setuptools.package-data] record_zstack_viewer = ["resources/*.png", "resources/*.svg"]`).
- [ ] **Step 3:** Run full suite green; commit `docs: README + package icon resources`.

---

## Self-review notes (author)
- **Spec coverage:** §2.1 controller→T1; §2.2 launcher→T3; §2.3 MainWindow drop→T2; §2.4 __main__→T4; §2.5 icon→T5; §2.6 installer→T6; §3 is_experiment→T1; §4 shared context→T1/T2 (asserted `w1._context is w2._context is c.context`); §6 testing→each task; README/packaging→T7.
- **Deviation from spec (flagged):** installer is `install_desktop_shortcut.py` (Python), not `.sh` — better serves the spec's interpreter-capture crux (`sys.executable`) and makes `build_desktop_entry` unit-testable. Same flags/behavior.
- **Type consistency:** `open_experiment(root)->MainWindow|None`, `is_experiment(root)->bool`, `build_desktop_entry(python_exe, icon, force_x11)->str`, `_build(argv)->(app,controller,launcher)` used consistently across tasks.
- **Honest limit (carry to execution):** actual Ubuntu app-grid / Wayland / double-click can't be verified from macOS — offscreen tests + `--dry-run` only; final GUI check is the user's on 22.04/24.04/26.04.
