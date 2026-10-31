# record-zstack-viewer

Standalone, memory-frugal desktop viewer for Squid **Record + Z-Stack** well-plate
scans stored as **OME-Zarr v3** (squid PR #564/#579). It gives you plate
navigation, per-FOV recording playback and z-stack viewing, and live updates while
an acquisition is still running.

- **Plate navigator** — a hue-coded plate map (blue = complete, amber = acquiring,
  grey = not-yet, red-✕ = aborted) plus an adaptive per-well FOV thumbnail strip.
- **Per-FOV viewer** — a multi-plane *recording* `(acq_t, frame, z, y, x)` shown
  grayscale, or a *z-stack* `(acq_t, c, z, y, x)` composited across channels, with
  auto-contrast on first display. The `acq_t` timepoint is shared across modes.
- **Live following** — while a run is in progress, coverage and per-FOV status
  update in place and the displayed FOV re-reads newly-written planes without a
  widget rebuild.

## Install

Requires Python ≥ 3.11.

```bash
pip install -e .
```

## Run

```bash
python -m record_zstack_viewer /path/to/experiment   # open an experiment directly
python -m record_zstack_viewer                        # no path -> drag-and-drop launcher
```

The path is the experiment root (the directory holding `recording/` and/or
`zarr/`). Each window follows a growing acquisition automatically (a ~1 s
UI-thread poll/tick).

## Export recordings to mp4

**Export mp4…** (top bar) converts recordings to H.264 mp4 files that play
anywhere. Pick wells on the plate matrix (click, shift-click a range, drag; the
displayed well is pre-selected), one file per recording store (`A1_fov_0.mp4`, mirroring `fov_0.ome.zarr`,
with `_z{j}` for plane stores and `_t{t}` after the well for time-lapse runs). Output defaults
to `~/Downloads/mp4-<experiment>/`, never inside the data folder. The videos are
a copy of the screen: the contrast limits and colour of the displayed recording
apply to every file in the export (the dialog shows them). A recording keeps the
limits you set when you switch wells, so set them once and export. Playback is
real time as recorded unless you set an fps. An mp4 already there is rewritten
(the zarr is only ever read).
Encoding runs in the background; live following keeps working.

The same thing from a shell, no Qt needed:

```bash
python -m record_zstack_viewer.export /path/to/experiment                 # all wells
python -m record_zstack_viewer.export /path/to/experiment --well A1 --out ~/videos
```

**Same converter inside Squid.** `record_zstack_viewer/export/convert.py` is
self-contained (numpy, tensorstore, imageio-ffmpeg only). Squid can import it,
or copy that one file, and call it when a recording store finalizes, so a
"save mp4s" toggle at acquisition time produces exactly the files the viewer
would, in the same `~/Downloads/mp4-<experiment>/` folder:

```python
from record_zstack_viewer.export.convert import convert_store, default_out_dir, video_name

out = default_out_dir(experiment_dir) / video_name(well, fov, t, z, n_t, n_z)
result = convert_store(store_dir, out)       # window/colour from the store; result.ok, .frames, .window, .color, .summary()
```

What the conversion does, and does not, change: full pixel resolution, every
captured frame, real-time rate. An mp4 cannot hold 16-bit data, so each video
goes through one linear 16-to-8 window. From the viewer that window is the
displayed contrast limits; from the CLI, and in Squid, it is the video's own
min and max, which clips nothing. The 8-bit value is then painted on the
viewer's black-to-channel-colour ramp: the displayed colour from the viewer,
the store's own `omero` channel colour otherwise (white stays gray). Window and
colour are printed per file and stored in the mp4's `comment` metadata, so any
file states how it was made and can be regenerated from the zarr. H.264 at
CRF 12 (preset ultrafast, speed over file size) is near-lossless, not bit-exact: about one gray level
of error for gray video, about three for coloured video, because players need
colour stored at half resolution (yuv420p). The zarr stays the ground truth.
Odd sides are cropped by one pixel (yuv420p needs even sides). Frames wider
than 4096 px (1x1 binning on large sensors) exceed what some players decode;
VLC handles them. A store that never completed exports the frames it captured,
under the same name; a frame slot the recorder never wrote is a black frame,
exactly as the zarr reads it. Frames stream one at a time through ffmpeg (from
the `imageio-ffmpeg` wheel), so memory does not grow with video length.

## Launcher & desktop shortcut

Launched with **no path**, the app opens a small **launcher**: drop a Squid
experiment folder onto it (or click **Browse…**) to open a viewer window. The
launcher stays open, so you can open several experiments; dropping a folder
onto an already-open viewer opens **another** window. All windows share one
bounded memory pool.

**Ubuntu desktop shortcut** (22.04 / 24.04 / 26.04) — run the installer *with
the Python that has the package installed* so the shortcut launches that exact
interpreter:

```bash
/path/to/env/bin/python scripts/install_desktop_shortcut.py --dry-run   # review first
/path/to/env/bin/python scripts/install_desktop_shortcut.py             # install (app grid)
/path/to/env/bin/python scripts/install_desktop_shortcut.py --uninstall
```

It installs the icon + a `.desktop` entry to `~/.local/share/…` (no sudo). On
Wayland (24.04/26.04), if the image canvas is blank, re-run with `--force-x11`
(launches via XWayland). `--desktop-copy` also drops a trusted icon on the
Desktop.

## CPU / headless note

Rendering goes through a single **ndv/OpenGL** path (there is no separate CPU
renderer). A CPU-only workstation is fine — its integrated-GPU OpenGL runs the
viewer without any special setup. Only a **truly GL-less host** (a headless
server, some VMs, or OpenGL-over-RDP) needs a software GL stack; there, force Mesa
llvmpipe:

```bash
LIBGL_ALWAYS_SOFTWARE=1 python -m record_zstack_viewer /path/to/experiment
```

## Degrades gracefully without the optional sidecars

The viewer targets current squid output but treats three additive sidecars as
optional, with a fallback for each (see `CONTRACT.md`):

- **FOV positions** — uses the per-store OME `translation` (or a `coordinates.csv`
  export) when present; **absent**, it reconstructs the FOV grid from the
  `acquisition.yaml` `wellplate_scan` geometry (and shows an unpositioned FOV list
  when even that is missing).
- **`events.jsonl` live status** — tails it as the primary live channel when
  present; **absent**, it falls back to a cheap file-watch + on-disk poll.
- **`thumbnails/`** — prefers squid's precomputed PNGs; **absent**, it computes a
  thumbnail from one decoded plane on demand.

Partially-written stores are read safely: an unwritten chunk reads as its
`fill_value` and re-reads real pixels once the acquisition writes it.

## How it works

- **Reads go through TensorStore only** (`driver: "zarr3"`) — never a whole-array
  read. A display request costs exactly the visible plane(s).
- **Memory is O(1) in dataset size.** Decoded pixels are bounded by a single shared
  TensorStore `cache_pool` (default ~192 MB), plus a small store-handle LRU and a
  bounded thumbnail cache. Computed thumbnails are written under
  `platformdirs.user_cache_dir`, never into the experiment folder.
- **Qt via `qtpy`** (resolves to PyQt5, matching squid); `pyqtgraph` for widgets;
  `ndv`/`vispy` for the GPU render surface; `cmap`, `imageio`, `platformdirs`,
  `PyYAML` round out the stack.

## Documentation

- Design spec: [`docs/specs/2026-07-05-record-zstack-viewer-design.md`](docs/specs/2026-07-05-record-zstack-viewer-design.md)
- Implementation plan (contract-first, TDD): [`docs/plans/2026-07-06-record-zstack-viewer.md`](docs/plans/2026-07-06-record-zstack-viewer.md)
- Versioned data contract (stable layout + optional sidecars + fallbacks): [`CONTRACT.md`](CONTRACT.md)

## Development

The test suite runs headless (Qt/ndv need an offscreen platform plugin):

```bash
pip install -e ".[test]"
QT_QPA_PLATFORM=offscreen python -m pytest tests/
```
