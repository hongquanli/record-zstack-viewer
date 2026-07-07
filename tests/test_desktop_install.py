"""Task 5/6 — app icon generation + Ubuntu desktop-shortcut installer.

Pure/offline checks only (macOS-friendly): the committed icon is a valid PNG,
`render_png` round-trips, and (Task 6) the `.desktop` content builder emits the
right `Exec`/`Icon`. The actual Ubuntu app-grid / Wayland / double-click
behaviour is NOT testable here — verify on 22.04/24.04/26.04.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def test_committed_icon_is_valid_png():
    from PIL import Image
    p = _REPO / "record_zstack_viewer/resources/icon-256.png"
    assert p.exists()
    assert Image.open(p).size == (256, 256)


def test_render_png_roundtrip(tmp_path):
    from record_zstack_viewer.resources import make_icon
    out = tmp_path / "i.png"
    make_icon.render_png(out, size=128)
    from PIL import Image
    assert Image.open(out).size == (128, 128)
    assert make_icon.SVG.lstrip().startswith("<svg")


def _load_installer():
    spec = importlib.util.spec_from_file_location(
        "rzv_installer", _REPO / "scripts/install_desktop_shortcut.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_desktop_entry_default():
    inst = _load_installer()
    txt = inst.build_desktop_entry("/opt/env/bin/python")
    assert "[Desktop Entry]" in txt
    assert "Exec=/opt/env/bin/python -m record_zstack_viewer" in txt
    assert "Icon=record-zstack-viewer" in txt
    assert "Terminal=false" in txt
    assert "QT_QPA_PLATFORM" not in txt          # default must NOT force a platform


def test_build_desktop_entry_force_x11():
    inst = _load_installer()
    txt = inst.build_desktop_entry("/opt/env/bin/python", force_x11=True)
    assert ("Exec=env QT_QPA_PLATFORM=xcb /opt/env/bin/python "
            "-m record_zstack_viewer") in txt


def test_installer_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    """--dry-run prints the entry and touches no install locations."""
    inst = _load_installer()
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setattr(inst.Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(inst, "_resolve_python", lambda explicit: "/opt/env/bin/python")
    rc = inst.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Exec=/opt/env/bin/python -m record_zstack_viewer" in out
    assert not (home / ".local/share/applications").exists()   # nothing written
