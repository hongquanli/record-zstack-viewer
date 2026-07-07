"""Generate the app icon (a well-plate motif).

Run ``python -m record_zstack_viewer.resources.make_icon`` to (re)write the
committed ``icon-256.png`` + ``icon.svg`` next to this file. The outputs are
committed so installing the desktop shortcut needs no render step.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

_PLATE = (38, 46, 60, 255)   # plate body
_WELL = (60, 70, 88, 255)    # empty well
_DONE = (47, 104, 207, 255)  # acquired well (accent blue, matches the navigator)


def render_png(path, size: int = 256) -> None:
    """Draw the icon at `size`×`size` (RGBA, transparent margin) and save to `path`."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    s = size
    d.rounded_rectangle(
        [s * 0.10, s * 0.14, s * 0.90, s * 0.86], radius=s * 0.09, fill=_PLATE)
    cols, rows = 4, 3
    left, right = s * 0.20, s * 0.80
    top, bottom = s * 0.24, s * 0.78
    r = min((right - left) / cols, (bottom - top) / rows) * 0.34
    for i in range(cols):
        for j in range(rows):
            cx = left + (right - left) * (i + 0.5) / cols
            cy = top + (bottom - top) * (j + 0.5) / rows
            fill = _DONE if j == 0 and i < 3 else _WELL  # top row: 3 "acquired"
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
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
