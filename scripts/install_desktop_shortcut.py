#!/usr/bin/env python3
"""Install an Ubuntu (22.04 / 24.04 / 26.04) desktop shortcut for record-zstack-viewer.

Run it WITH THE PYTHON THAT HAS THE PACKAGE INSTALLED, e.g.::

    /path/to/env/bin/python scripts/install_desktop_shortcut.py

so the shortcut's ``Exec`` uses that exact interpreter (the deps live in a
conda/venv, not the system Python). Use ``--dry-run`` first to review.

The ``.desktop``/hicolor mechanism is identical across 22.04/24.04/26.04. The
only cross-release nuance is the Qt display platform: on Wayland (24.04/26.04)
Qt auto-detects; if the ndv/OpenGL canvas is blank, re-run with ``--force-x11``
(emits ``Exec=env QT_QPA_PLATFORM=xcb ...`` — XWayland, the reliable GL fallback).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

APP_ID = "record-zstack-viewer"
_NAME = "Record + Z-Stack Viewer"
_COMMENT = "View Squid record + z-stack well-plate scans (OME-Zarr)"


def build_desktop_entry(python_exe: str, icon: str = APP_ID, force_x11: bool = False) -> str:
    """Return the `.desktop` file text. `force_x11` prefixes Exec with env QT_QPA_PLATFORM=xcb."""
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


def _apps_dir() -> Path:
    return Path.home() / ".local/share/applications"


def _icon_png_dir() -> Path:
    return Path.home() / ".local/share/icons/hicolor/256x256/apps"


def _icon_svg_dir() -> Path:
    return Path.home() / ".local/share/icons/hicolor/scalable/apps"


def _repo_resources() -> Path:
    return Path(__file__).resolve().parent.parent / "record_zstack_viewer" / "resources"


def install(python_exe: str, *, force_x11: bool = False, desktop_copy: bool = False) -> dict:
    """Write the icon(s) + `.desktop` and refresh caches. Returns {label: path}."""
    written: dict[str, Path] = {}
    _icon_png_dir().mkdir(parents=True, exist_ok=True)
    _icon_svg_dir().mkdir(parents=True, exist_ok=True)
    png = _icon_png_dir() / f"{APP_ID}.png"
    shutil.copyfile(_repo_resources() / "icon-256.png", png)
    written["icon_png"] = png
    svg_src = _repo_resources() / "icon.svg"
    if svg_src.exists():
        svg = _icon_svg_dir() / f"{APP_ID}.svg"
        shutil.copyfile(svg_src, svg)
        written["icon_svg"] = svg
    _apps_dir().mkdir(parents=True, exist_ok=True)
    desktop = _apps_dir() / f"{APP_ID}.desktop"
    desktop.write_text(build_desktop_entry(python_exe, force_x11=force_x11))
    written["desktop"] = desktop
    for cmd in (["update-desktop-database", str(_apps_dir())],
                ["gtk-update-icon-cache", "-f",
                 str(Path.home() / ".local/share/icons/hicolor")]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=False)
    if desktop_copy:
        dcopy = Path.home() / "Desktop" / f"{APP_ID}.desktop"
        if dcopy.parent.is_dir():
            dcopy.write_text(desktop.read_text())
            dcopy.chmod(0o755)
            if shutil.which("gio"):
                subprocess.run(["gio", "set", str(dcopy), "metadata::trusted", "true"],
                               check=False)
            written["desktop_copy"] = dcopy
    return written


def uninstall() -> None:
    """Remove the installed `.desktop` + icons (idempotent)."""
    for p in (_apps_dir() / f"{APP_ID}.desktop",
              _icon_png_dir() / f"{APP_ID}.png",
              _icon_svg_dir() / f"{APP_ID}.svg",
              Path.home() / "Desktop" / f"{APP_ID}.desktop"):
        p.unlink(missing_ok=True)
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(_apps_dir())], check=False)


def _resolve_python(explicit: str | None) -> str:
    exe = explicit or sys.executable
    r = subprocess.run([exe, "-c", "import record_zstack_viewer"], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"error: {exe} cannot import record_zstack_viewer — install it first "
                 "(pip install -e .) or pass --python /path/to/env/python")
    return exe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Install the Ubuntu desktop shortcut.")
    ap.add_argument("--python", help="interpreter for Exec (default: the one running this script)")
    ap.add_argument("--force-x11", action="store_true",
                    help="Exec via env QT_QPA_PLATFORM=xcb (Wayland GL fallback)")
    ap.add_argument("--desktop-copy", action="store_true",
                    help="also place a trusted copy on ~/Desktop")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written; change nothing")
    a = ap.parse_args(argv)

    if a.uninstall:
        uninstall()
        print("uninstalled")
        return 0
    exe = _resolve_python(a.python)
    if a.dry_run:
        print(f"# {_apps_dir()}/{APP_ID}.desktop\n{build_desktop_entry(exe, force_x11=a.force_x11)}")
        print(f"# icon -> {_icon_png_dir()}/{APP_ID}.png (from resources/icon-256.png)")
        if not a.force_x11:
            print("# note: blank canvas under Wayland (26.04)? re-run with --force-x11")
        return 0
    written = install(exe, force_x11=a.force_x11, desktop_copy=a.desktop_copy)
    for k, v in written.items():
        print(f"wrote {k}: {v}")
    if not a.force_x11:
        print("note: blank canvas under Wayland (26.04)? re-run with --force-x11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
