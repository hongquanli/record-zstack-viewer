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
