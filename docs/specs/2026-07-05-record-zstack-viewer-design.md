# Design: Record + Z-Stack Viewer for Well-Plate Scans

- **Date:** 2026-07-05
- **Author / approver:** Hongquan
- **Repo:** `record-zstack-viewer` (Python package `record_zstack_viewer`)
- **Status:** Design approved & review-hardened 2026-07-05 (format re-verified against branch HEAD; auto-contrast, recording time-model, per-FOV coordinate persistence, cache location, single ndv `DataWrapper` folded in). All decisions resolved — upstream per-FOV `coordinates.csv` persistence **approved** (§7/§16). Ready for implementation planning (with Opus).
- **Consumes:** the OME-Zarr output of Squid PR #564 (`feat/record-zstack-acquisition`) and PR #579 (`feat/multi-plane-recording`, stacked on #564)
- **Reference (non-binding):** `Cephla-Lab/ndviewer_light` — reuse concepts, not constraints

---

## 1. Goal

A **standalone desktop viewer** that opens a Squid "Record + Z-Stack" experiment folder and lets a user navigate a well plate, then inspect, for any FOV, its **recording** (single-channel video, possibly at multiple z-planes) and/or its **z-stack** (multichannel Z volume). It must:

- consume **very little memory** — the hard guarantee is **footprint is O(1) in dataset size** (a 2 GB and a 2 TB experiment cost the same); in practice total RSS ≈ the Python/Qt/GL runtime baseline **+ a tunable cache budget (~128–256 MB)**, targeting a few-hundred-MB steady state;
- **work with ongoing acquisitions** — read data that is still being written, updating incrementally;
- **show plate navigation** — a spatial well/FOV map, not a flat slider;
- run **on the acquisition PC** without competing with the running scan.

### Decisions locked during brainstorming (2026-07-05)

| Decision | Choice |
|---|---|
| Platform | Standalone desktop, Python + GPU |
| Runs on | Acquisition PC — strict memory budget |
| Primary job | Live QC **and** post-hoc review, both first-class |
| Data scale | Design for the large case (never falls over) |
| Rendering stack | **ndv** (per-FOV) + **custom Qt plate navigator** + **TensorStore** reader |
| GPU vs CPU | Single **ndv/OpenGL** path. "CPU-only" machines run ndv on their **integrated-GPU OpenGL** (Intel/AMD iGPU); truly GL-less hosts (headless/VM/RDP) use **software OpenGL (Mesa llvmpipe)** via an env var — no custom CPU renderer |
| Format scope | **Zarr v3 (PR 564/579) only**, behind a format-adapter seam for later formats |
| Z-stack display | 2D slice + channel composite (Z/T sliders + LUTs); no volume rendering in v1 |
| Live source | File-watch + slow poll (standalone-safe); optional IPC push is a later fast-path |

**UI decided via visual companion (2026-07-05)** — Layout A (docked left navigator + main viewer), adaptive FOV list, hue-coded well status, right-side channel dock (details in §6/§8). Mockups persist at `.superpowers/brainstorm/*/content/{app-layout,layout-v2,plate-nav,fov-viewer}.html`.

### In scope (v1)

- Reading the PR 564/579 OME-Zarr v3 layout (recording + z-stack), including #579 multi-plane recordings.
- Plate → well → FOV navigation with live status.
- FOV viewer: recording playback (T scrub, optional Z-plane selector) and z-stack composite (Z + T sliders, channel LUTs).
- Live incremental updates during acquisition.
- Memory-bounded operation.

### Out of scope (v1, YAGNI)

- Legacy formats (INDIVIDUAL_IMAGES, OME-TIFF, HCS-plate OME-Zarr). The reader has an adapter seam so these drop in later; not implemented now.
- True 3D volume / MIP rendering.
- Stitching FOVs into a single seamless mm-accurate mosaic (we lay out a *grid*, see §7).
- Editing/annotation/analysis. This is a viewer.
- Browser/remote access (a separate future track if remote monitoring is ever needed).

---

## 2. The data format (authoritative — the implementer should not re-derive this)

Everything below is verified against the PR #564/#579 code and tests on branches `feat/record-zstack-acquisition` and `feat/multi-plane-recording`.

### 2.1 Experiment folder

```
{base}/{experiment_id}_{YYYY-MM-DD_HH-MM-SS.ffffff}/
├── acquisition_channels.yaml     # objective, confocal_mode, channels (both phases, deduped by name)
├── acquisition.yaml              # full Record/Z-Stack settings snapshot (incl. wellplate_scan)
├── .done                         # empty completion marker (written even on abort)
├── recording/                    # present only if the recording phase ran
│   └── t{t}/{region}/fov_{k}.ome.zarr          # #564 single-plane
│       (or)          fov_{k}_z{j}.ome.zarr      # #579 multi-plane (j = 0..Nz-1)
└── zarr/                         # present only if the z-stack phase ran  (NOTE: "zarr", not "zstack")
    └── {region}/fov_{k}.ome.zarr/0
```

- `{region}` = **well name** verbatim (`A1`, `B12`, …). `fov_{k}` = FOV index within that well.
- `.done` marks the whole experiment finished (or aborted). There is **no** `coordinates.csv`, `acquisition parameters.json`, or `configurations.xml` on this path.

### 2.2 The recording ↔ z-stack asymmetry (design around this)

| | **Recording** | **Z-stack** |
|---|---|---|
| Path | `recording/t{t}/{well}/fov_{k}[_z{j}].ome.zarr` | `zarr/{well}/fov_{k}.ome.zarr/0` |
| Array location | **store root** (`datasets.path="."`) | **`/0`** (group metadata one level up) |
| Shape | `(T, 1, 1, Y, X)` | `(Nt, C, NZ, Y, X)` |
| What T means | **video frames** (`round(fps·duration)`) | **acquisition timepoints** |
| Where timepoints live | **directories** `t{t}/` | the **T axis** of one accumulating store |
| Channels | 1 | C (multichannel) |
| Z | 1 per store; multiple via `_z{j}` **sibling stores** | `NZ` on the Z axis of one store |

**Rule of thumb for the reader:** a store path ending in `/0` → array is at `/0`; otherwise array is at the store root. (Mirrors `ZarrWriter._is_ome_ngff_array_path`.)

**Time is overloaded — do not collapse it.** For a recording FOV, *acquisition timepoints* live as separate store directories (`recording/t{t}/…`) while each store's own **T axis is video frames**. So one recording FOV logically spans **(acq_t across t-dirs) × (z across `_z{j}` siblings) × (frame within a store)** → `(acq_t, frame, z, Y, X)`. Moving in acquisition time = *switching stores*, not scrubbing the frame axis. The z-stack's T axis, by contrast, **is** acquisition time. The reader and viewer model these as distinct axes and keep `acq_t` aligned when toggling recording↔z-stack (§4, §8).

### 2.3 On-disk technical facts

- **Zarr v3**, **OME-NGFF 0.5**, axis order **T, C, Z, Y, X**. Read with **TensorStore** (`driver: "zarr3"`). *napari-ome-zarr cannot read Zarr v3 — do not rely on it.*
- **Single-scale — NO multiscale pyramid.** `multiscales[0].datasets` has exactly one level. Overviews/thumbnails must be computed by the viewer (§9).
- **Chunking:** default `FULL_FRAME` → chunk `(1,1,1,Y,X)` (one frame = one chunk = one file). `TILED_512`/`TILED_256` → `(1,1,1,512,512)` / `(1,1,1,256,256)` for real sub-plane tiling.
- **Sharding:** none under `FAST`/`NONE` compression (each chunk its own file); per-Z-level shard under `BALANCED`/`BEST`.
- **Compression:** default `FAST` = blosc-LZ4 clevel 1 + byte shuffle (fast decode). Also zstd variants.
- **dtype:** from the frames — monochrome `uint16` (or `uint8`).
- **`is_hcs=False`** always on this path → **no** `plate.ome.zarr/{row}/{col}` HCS hierarchy; flat per-FOV stores only.

### 2.4 Metadata the viewer consumes

- **`omero.channels[]`** (per store): `label`, `color` (hex), `emission_wavelength {value, unit}`, `active`, `window {start,end,min,max}` → channel names + LUT colors. **⚠ `window` is the full dtype range** (e.g. `0–65535` for uint16), **not** a usable display range — squid writes no percentile/auto-contrast, so trusting it renders dim (12-bit-ish) fluorescence near-black. The viewer must **auto-scale contrast itself** (§8); use `window` only as clamp bounds.
- **`_squid`** block (per store): `structure` (`"5D-TCZYX"`), `pixel_size_um`, `z_step_um`, `time_increment_s`, `chunk_mode`, `compression`, `shape`, `dtype`, `is_hcs`, `acquisition_complete` (bool), `aborted` (bool, when aborted), and for recordings `captured_frames`/`expected_frames`, `dropped_frames`, `write_errors`.
- **#579 multi-plane recording** adds to each `fov_{k}_z{j}` store's `_squid`: `plane_index`, `plane_z_offset_um` (= `bottom_z_offset_um + j·dz_um`), `n_planes`. **Absence of these keys ⇒ single-plane** (`n_planes=1`, `plane_index=0`).
- **`acquisition.yaml → wellplate_scan`**: `regions[]` (`name`, `center_mm`, `shape`), `scan_size_mm`, `overlap_percent`, `xy_mode`. This is the **only** source of stage geometry.

### 2.5 The XY-position gap (a real constraint)

There is **no absolute per-FOV XY on disk today** — coordinate transforms are scale-only (no translation). **Decided fix (§7):** persist a per-FOV `coordinates.csv` upstream in PR #564/#579 → mm-exact placement, with reconstruction demoted to a fallback for pre-existing data. Until that lands, the fallback reconstructs the grid from `acquisition.yaml`, and its within-well FOV order must replicate squid's scan ordering exactly.

---

## 3. Architecture overview

Seven layers; data flows bottom-up, user events top-down. Each layer is independently testable and communicates through a narrow interface.

```
┌──────────────────────────────────────────────────────────────┐
│  App shell (Qt main window)                                     │
│   ┌───────────────────────────┐  ┌──────────────────────────┐  │
│   │  PlateNavigator (custom)  │  │  FovViewer (ndv widget)  │  │
│   │  plate → well → FOV, live │→ │  recording | z-stack     │  │
│   └───────────────────────────┘  └──────────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│  LiveEngine (file-watch + poll) ── signals ──▶ model refresh    │
├──────────────────────────────────────────────────────────────┤
│  ThumbnailCache (lazy overviews; sidecar + RAM LRU)            │
├──────────────────────────────────────────────────────────────┤
│  ExperimentModel (tree; position reconstruction; yaml parse)   │
├──────────────────────────────────────────────────────────────┤
│  Reader layer:  FovStore  (TensorStore; plane/tile on demand)  │
│                 behind FormatAdapter seam (Zarr-v3 impl in v1)  │
├──────────────────────────────────────────────────────────────┤
│  MemoryGovernor (bounded caches: tiles, store handles, thumbs) │
└──────────────────────────────────────────────────────────────┘
```

Key invariant across all layers: **no operation reads a whole array**. Reads are per-plane or per-tile, sized to what is visible.

---

## 4. Component: Reader layer (`FovStore` + `FormatAdapter`)

**Responsibility.** Open one FOV dataset (recording or z-stack) lazily and expose a uniform, memory-frugal N-D source, hiding every §2 asymmetry.

**Interface (illustrative).**
```python
class FovSource(Protocol):
    dims: tuple[str, ...]          # canonical names, e.g. ("t","c","z","y","x")
    shape: tuple[int, ...]
    dtype: np.dtype
    metadata: FovMetadata          # channels, colors, windows, pixel_size_um, z_step_um,
                                   # time_increment_s, plane offsets, acquisition_complete, aborted
    def read_plane(self, index: dict[str,int]) -> np.ndarray            # one (Y,X) plane
    def read_tile(self, index: dict[str,int], yx_slice) -> np.ndarray   # chunk-aligned sub-plane
    def filled_extent(self, axis: str) -> int   # how many indices along axis are actually written
```

**Two concrete adapters (Zarr-v3 in v1):**
- **Recording adapter.** Canonical dims **`("acq_t","frame","z","y","x")`** (single channel), composed across **two directory dimensions plus one in-store axis**: `acq_t` = the `recording/t{t}/{well}/` timepoint dirs, `z` = the `fov_k_z{j}` sibling stores (ordered by `plane_index`; a lone `fov_k.ome.zarr` ⇒ `z`-size 1), `frame` = each store's own T axis (array at **root**). `read_plane({acq_t,frame,z})` dispatches to the store at `recording/t{acq_t}/{well}/fov_k[_z{z}].ome.zarr` and reads frame `frame`. Exposes `plane_z_offset_um` per z. Degenerate cases collapse cleanly (single timepoint ⇒ `acq_t`-size 1; single plane ⇒ `z`-size 1).
- **Z-stack adapter.** Opens the array at **`/0`**. Canonical dims **`("acq_t","c","z","y","x")`** — its `acq_t` (the store's T axis) aligns with the recording adapter's `acq_t`, so one shared acquisition-timepoint selector drives both modes.

**Filled-extent detection (drives live + avoids reading fill_value).** Reliable **only under the default `FAST` compression**, where each chunk is its own file → **chunk-file presence = written frame**. ⚠ Under `BALANCED`/`BEST` **sharding** one shard file aggregates many chunks, so presence no longer reveals per-frame progress; and `_squid.captured_frames`/`acquisition_complete` are written at **finalize/abort, not live**. So: FAST ⇒ per-frame extent from chunk presence; sharded ⇒ degrade to whole-store *"acquiring"* via store mtime, or a fill_value probe on the frontier chunk. Unwritten indices are reported as *pending*, never read as zeros silently.

**Depends on:** TensorStore, MemoryGovernor (tile/handle caches). **Does not** depend on Qt — pure, unit-testable against fixture stores.

**Why TensorStore (not zarr-python / dask-per-file):** it is squid's own writer backend, it reads Zarr v3 + sharding + blosc natively, does async + internal caching, and returns numpy for exactly the slice requested. This replaces the `dask.from_delayed`-per-file graph in `ndviewer2`, which is memory-light but slow to construct for large plates.

---

## 5. Component: ExperimentModel

**Responsibility.** Turn an experiment folder into a navigable tree and answer "what exists, where, and is it done."

**Tree:** `Experiment → wells(region) → FOVs(k) → {recording?, zstack?}`. **Acquisition time is a per-dataset axis, not a tree level** (directories for recordings, an array axis for z-stacks — §2.2), so it surfaces as a shared *timepoint selector* in the viewer rather than as tree depth.
- Recording datasets are assembled across `recording/t{t}/{well}/` dirs (→ `acq_t`) and `fov_k_z*` siblings (→ `z`); a FOV with any recording store is one recording dataset.
- Z-stack datasets come from `zarr/{well}/fov_k.ome.zarr/0`; all timepoints live on its T axis.
- Parses `acquisition.yaml` / `acquisition_channels.yaml`; reads `.done`.

**Per-node status:** `empty` · `in_progress` · `complete` · `aborted`, derived from `_squid.acquisition_complete` / `aborted` and filled-extent.

**Interface:** `wells()`, `fovs(well)`, `fov_datasets(well, k) -> {recording, zstack}`, `status(node)`, `plate_geometry()`. Emits change events consumed by the navigator when the LiveEngine pokes it.

**Depends on:** Reader layer (for metadata/extent), filesystem. No Qt.

---

## 6. Component: PlateNavigator (custom Qt canvas)

**Responsibility.** Spatial navigation with live status. Three zoom tiers:

1. **Plate map** — wells positioned from plate geometry (§7); each well a cell colored by aggregate status (**colorblind-safe: pair color with a fill/shape state, or use blue/orange rather than red/green**); count/coverage badges. **Zoom/pan + hover tooltips** so 384/1536-well plates stay usable at ~6–9 px/well. Click → well.
2. **Well mosaic** — the well's FOVs as a reconstructed grid of **thumbnails** (from ThumbnailCache); status ring per FOV. Click → FOV.
3. **FOV selection** — hands `(well, k)` + chosen dataset (recording/zstack) to the FovViewer.

**Rendering:** a lightweight `QGraphicsView`/`pyqtgraph` canvas (both already in squid's stack). Thumbnails only — never full frames — so panning a 384-well plate stays cheap. Live: cells appear/recolor as the model changes; no full redraw.

**Decided layout (visual companion, 2026-07-05):** shell = **Layout A** — a persistent left navigator dock + the main FOV viewer. The FOV thumbnail strip is **adaptive**: shown only for wells with >1 FOV; single-FOV wells (most recordings) open the FOV directly on well-click, and the freed space goes to a bigger plate map + a small FOV info block. Well status = **hue-coded** (blue done / amber acquiring / grey not-yet / grey + red ✕ aborted). The plate map has **scroll-zoom + pan + hover tooltips** for 384/1536, and a **live coverage header** (`n/N wells · acquiring {well} (FOV k/m)`).

**Depends on:** ExperimentModel, ThumbnailCache. **Does not** import ndv.

---

## 7. Plate/FOV positioning (the XY gap, §2.5)

**DECIDED (2026-07-05) — persist per-FOV coordinates upstream.** The real per-FOV `(x, y, z_ref)_mm` exist at acquisition time but aren't written today. We will add a small change to the record+zstack path (PR #564/#579) to write a **`coordinates.csv` at the experiment root**, mirroring classic multipoint's columns (`region, fov, z_level, x (mm), y (mm), z (um)`; one row per FOV, add `time` if per-timepoint). The viewer then reads it directly for **mm-exact** placement — plate map *and* within-well FOV positions — and the reconstruction below becomes a **fallback only for data acquired before this lands** (or other formats). *(Acquisition-side task, tracked in §16.)*

**Fallback — reconstruct from `acquisition.yaml` (pre-coordinates data only):**
- **Wells (exact):** place each `region` at its `center_mm`. If absent, compute from the well name (`A1`→row/col) × standard `WELL_SPACING_MM`.
- **Plate format:** "96/384" isn't persisted → **infer** from `regions[].center_mm` spacing matched to `sample_formats.csv`, with a **manual override** (inference is ambiguous on partial plates / custom regions).
- **FOVs within a well:** derive `NX×NY` from `scan_size_mm`/`overlap_percent`/`pixel_size_um`/frame size and map `fov_k → (i,j)`. ⚠ This **must replicate squid's `ScanCoordinates` ordering exactly, including the S-Pattern serpentine**, or thumbnails land in the wrong cells — port and unit-test against squid's implementation.

**Non-wellplate scans:** in **"Current Position" mode** `acquisition.yaml` has **no `wellplate_scan` block** and `region_id` isn't an `A1`-style name → both geometry and name-fallback fail. Detect this and **degrade to a flat FOV list** (no plate map).

This math lives in a **pure, unit-tested `geometry` module** (no Qt, no IO) — highest-value to test because it's fiddly and format-derived (and ideally deleted by the upstream fix).

---

## 8. Component: FovViewer (ndv — single rendering path)

**Responsibility.** Render one FOV dataset — recording or z-stack — from a lazy `FovSource` via a single `ndv.ArrayViewer`.

- **Recording mode** — dims `(acq_t, frame[, z], y, x)`, single channel. **`frame`** is the play/scrub axis (play/pause, speed, frame counter); **`acq_t`** is a *separate* timepoint selector (not the scrub); if `n_planes>1`, a **z-plane selector** labeled by `plane_z_offset_um`; gray colormap by default.
- **Z-stack mode** — dims `(acq_t, c, z, y, x)`, `channel_mode="composite"`, `channel_axis=c`; **Z** slider + the shared **`acq_t`** timepoint selector; per-channel **LUTs** (colors from `omero`) with **auto-scaled contrast** (below).
- A FOV with **both** phases shows a compact **Recording | Z-stack** toggle; the **`acq_t` selector stays put when toggling**, so you compare the same timepoint across modes.

**Feeding ndv — one data path (`ndv.DataWrapper` over `FovSource`).** Rather than layering dask/xarray over the reader (which stacks 3–4 caches and bypasses the reader's own), implement a **custom `ndv.DataWrapper`** (ndv supports pluggable wrappers, `models/_data_wrapper.py`) that ndv slices directly through `FovSource.read_plane/read_tile`. Bound memory with **TensorStore's own `cache_pool`** (a byte cap we configure) instead of a home-grown tile-LRU — one cache we control end-to-end (§11). Only the visible plane(s) materialize on the GPU.

**Contrast — auto-scale, never trust `omero.window` (§2.4).** On first display of a channel, compute an auto range (e.g. 1–99.9 percentile) from the current decoded plane and set the LUT limits from it; use `omero.window.min/max` only as hard clamp bounds. Otherwise dim 12-bit-ish fluorescence in a uint16 store renders near-black. Cache the range per channel; let the user override.

**GPU vs "CPU-only" — one path, ndv/OpenGL.** ndv renders exclusively through a GPU API (vispy→OpenGL, pygfx→WebGPU); it has **no** software-raster path, and we deliberately do **not** build one. "CPU-only" machines still have an OpenGL-capable GPU: **integrated graphics** (Intel/AMD iGPU — anything driving a monitor) expose OpenGL 4.x, far above vispy's 2.1+ floor, hardware-accelerated. The only genuinely GL-less cases — headless servers, some VMs, OpenGL-over-RDP — are handled **without code** by pointing vispy at **Mesa llvmpipe** software OpenGL (`LIBGL_ALWAYS_SOFTWARE=1` on Linux; the same trick that runs napari headless): functional, slower.

**ndv isolation (API-churn mitigation, learned from `ndviewer_light`).** ndv is used only behind a thin `FovViewer` wrapper exposing `set_source()` / `update_extent()`; do **not** depend on undocumented ndv internals for in-place data swaps — pin one known data-set path per ndv version with a clean rebuild fallback. **The rebuild fallback must preserve view state** (zoom/pan, per-channel contrast/LUT, current indices) so a live refresh never resets what the user is looking at. One fragile dependency, one file. (If a hard no-OpenGL requirement ever appears, this wrapper is the single place a numpy/pyqtgraph raster backend would drop in — explicitly out of scope now.)

**Decided control layout (visual companion, 2026-07-05):** a **Recording | Z-stack** toggle at top. Recording mode exposes **two distinct time controls** — a *Frame* scrub (play/pause/fps — the video) and a *separate* **Timepoint** stepper (`acq_t`, shared with z-stack so toggling preserves it) — plus **Z-plane** pills (labeled by `plane_z_offset_um`, shown only when `n_planes>1`) and an **auto/manual contrast** control. Z-stack mode uses a **right-side channel dock** (per channel: color swatch, name, eye toggle, contrast slider) with **Z** and the shared **Timepoint** sliders under the image.

**Depends on:** Reader layer, ndv/vispy. **Does not** import the navigator.

---

## 9. Component: ThumbnailCache (the "no pyramid" answer)

**Responsibility.** Cheap, live overviews for the navigator.

- **Prefer a squid-written thumbnail if present** (upstream affordance, §16 / to-do `2026-07-06-record-zstack-fov-thumbnails.md`): read `thumbnails/{region}/fov_{k}.png` (or the plate overview) directly and skip decoding. Only compute on the fly when absent (older data).
- When computing: a small thumbnail (~128–256 px) from one representative plane (recording: a mid/first frame; z-stack: mid-Z of a chosen channel, or a max-projection over Z). Downsample from a single decoded plane — never the whole array.
- **Two-level cache:** persistent disk cache in an **OS user-cache dir** (`platformdirs.user_cache_dir`) keyed by experiment path+id + store completion/mtime, plus a bounded **RAM LRU**. **Do not default into the experiment folder** — it's often read-only/NAS/archived, watched by squid, or **sync-replicated (this project lives in Dropbox)**; thousands of thumbnail files there cause sync storms and pollute the dataset. In-folder caching is opt-in.
- **Live:** invalidate/recompute a thumbnail when its FOV's status/extent changes; emit to the navigator.

**Depends on:** Reader layer, MemoryGovernor. No Qt.

---

## 10. Component: LiveEngine

**Responsibility.** Keep the model current during acquisition, incrementally.

**Primary channel — tail `events.jsonl`** (upstream affordance, §16 / to-do `2026-07-06-record-zstack-live-status-and-coverage.md`). When present, tail the experiment-root event log (`run_started / fov_started / heartbeat / store_finalized / timepoint_done / aborted / run_done`) → O(1) live extent, current-position highlight, coverage (k/N), and prompt completion/abort, independent of compression/sharding. This is the cheap, authoritative live signal.

**Fallback — file-watch + poll** (data without an event log, incl. already-acquired experiments):
- `QFileSystemWatcher` on `recording/`, `zarr/`, and the root (`.done`), **plus** a slow (~1–2 s) poll (network/SMB/NFS shares often miss FS events).
- Cheap per-store signature: store/dir **mtime** + `_squid.acquisition_complete` (once sealed) + filled-extent (chunk presence under the FAST default, §4). Use **one `scandir` per store**, never a stat per frame-file.
- Works around the verified squid gaps: no heartbeat, z-stack `acquisition_complete` only at teardown, no in-process signal — so the fallback infers progress from pixels/mtime and is coarser than the event log.

- **Incremental application:** append new FOV/well/timepoint nodes; for the **open** FOV, extend the recording's `frame`/`acq_t` extent or refresh the z-stack's filled `(acq_t,z)` — **never tear down** the ndv widget or rebuild the whole model. Flip status on events/markers.
- **Reading during writes is safe:** frames flush synchronously and are cross-process visible; unwritten frames read as `fill_value 0` (`zarr.json` declares the full shape up front) → show as *pending*, not zeros. If a store's `zarr.json` is briefly mid-write, back off and retry.

**Depends on:** ExperimentModel, Reader layer. Emits Qt signals to the UI.

---

## 11. Memory strategy (the hard requirement)

The guarantee is **footprint independent of dataset size**; the caches below are the tunable budget on top of the interpreter/Qt/GL baseline. Concrete caps, all configurable:

- **GPU/display:** only visible planes — 1 per visible channel (z-stack composite) or 1 (recording). ndv/vispy upload exactly the current slice.
- **Decoded-pixel cache = TensorStore `cache_pool`** (a byte cap we configure, default ~128–256 MB), *not* a separate home-grown LRU — the reader, the ndv `DataWrapper`, and prefetch all share this one cache we control (§8). It bounds RAM on huge frames under `TILED_*` and dedupes repeat reads; this is what makes "O(1) footprint" enforceable rather than aspirational.
- **Open-store handle LRU:** cap concurrent TensorStore handles (default ~32); close idle ones. Fixes `ndviewer_light`'s "one open handle per FOV for the whole session" fd blow-up on plate runs.
- **Thumbnail cache:** bounded count in RAM; overflow lives on the sidecar disk cache.
- **Never** hold `dask`/numpy views of full arrays; **never** prefetch the whole T axis of a recording.
- **Prefetch (bounded):** optionally prefetch the next few `t`/`z` neighbors on a worker thread for smooth scrub/playback, counted against the tile-LRU budget. IO stays **off the GUI thread** (another `ndviewer_light` lesson — it did all IO on the Qt thread).

---

## 12. Concurrency

- Qt main thread: UI + ndv only.
- A small worker pool (or `QThreadPool`) for: TensorStore reads, thumbnail computation, live signature scans. Results marshalled back via Qt signals.
- The LiveEngine poll and FS-watch callbacks never block the UI; heavy work is dispatched to workers.

---

## 13. Module / project layout

```
record_zstack_viewer/
├── reader/
│   ├── adapter.py          # FormatAdapter seam + registry
│   ├── zarr_v3.py          # recording + z-stack adapters (TensorStore)
│   ├── fov_source.py       # FovSource protocol, multi-plane compositor
│   └── metadata.py         # _squid / omero / yaml parsing
├── model/
│   ├── experiment.py       # ExperimentModel, tree, status
│   └── geometry.py         # plate/FOV position reconstruction (PURE, tested)
├── cache/
│   ├── memory_governor.py  # tile LRU, handle LRU, byte accounting
│   └── thumbnails.py       # ThumbnailCache (sidecar + RAM)
├── live/
│   └── engine.py           # file-watch + poll, incremental refresh
├── ui/
│   ├── app.py              # main window / shell
│   ├── plate_navigator.py  # custom Qt canvas (3 tiers); pure Qt raster (no GL)
│   └── fov_viewer.py       # ndv wrapper (only file that imports ndv; isolates ndv API)
├── tests/                  # fixtures = tiny synthetic Zarr-v3 stores
└── __main__.py             # `python -m record_zstack_viewer [path]`
```

Every non-UI module is Qt-free and unit-testable in isolation.

---

## 14. Error handling & edge cases

- **Both phases / neither:** a FOV may have recording only, z-stack only, or both → viewer offers what exists; a well with neither is `empty`.
- **Aborted / incomplete store:** `_squid.aborted`/`acquisition_complete=false` with partial chunks → show what's written, badge as incomplete; missing planes render as "pending," not zeros.
- **Mid-write metadata:** retry-with-backoff on transient `zarr.json` read failures.
- **Missing / partial / mid-write `acquisition.yaml`:** fall back to name-based well layout + acquisition-order FOV grid; warn positions are approximate; retry if mid-write.
- **"Current Position" / non-wellplate scan:** no `wellplate_scan` block and non-`A1` region ids → **degrade to a flat FOV list** instead of a plate map (§7).
- **Multi-plane with holes** (abort between planes): order by present `plane_index`; gaps shown as pending z entries.
- **Huge single frame** (design-for-large): rely on `TILED_*` tiling + tile-LRU; if a store is `FULL_FRAME` with a very large Y×X, decode the one chunk but downsample immediately for the current zoom (don't keep full-res beyond the tile budget).
- **Network filesystem:** poll fallback active; handle-LRU keeps fd/connection count bounded.
- **No discrete GPU:** integrated graphics (Intel/AMD iGPU) provide OpenGL well beyond vispy's needs → ndv runs hardware-accelerated, no special handling. **Fully GL-less host (headless / some VMs / OpenGL-over-RDP):** run vispy on software OpenGL (Mesa **llvmpipe**, `LIBGL_ALWAYS_SOFTWARE=1`) — functional, slower, no code change.

---

## 15. Testing strategy

- **Pure units (no Qt, no GPU):** `geometry` (well/FOV reconstruction, pattern mapping, format inference); `metadata` parsing (`_squid`, `omero`, multi-plane grouping, single-plane fallback); reader index math (root vs `/0`, dims mapping); filled-extent detection.
- **Reader integration:** against **tiny synthetic Zarr-v3 fixtures** (write with TensorStore in the same TCZYX/chunk/compression config squid uses) — recording `(T,1,1,Y,X)`, multi-plane `_z{j}` siblings, z-stack `(Nt,C,NZ,Y,X)`.
- **Live:** a fixture writer that grows a store chunk-by-chunk (atomic writes) → assert incremental model updates and no whole-array reads (instrument the reader).
- **Memory:** assert RSS ceiling holds while scrubbing a large fixture (cache caps respected; handle count bounded).
- **UI smoke:** open fixtures, switch tiers, toggle recording/z-stack, play a recording — direct-instantiation style (matches squid's non-`qtbot` test convention where possible).
- **Adapter-seam smoke test:** even though legacy formats are deferred, run one fixture in the classic **HCS-plate zarr** shape (`plate.ome.zarr/{row}/{col}/{fov}/0`, which the same squid writer produces with `is_hcs=True`) through `FormatAdapter` so "extensible" isn't untested.

A small **acquisition simulator** (in the spirit of `ndviewer_light/simulate_live_acquisition.py`, but writing the 564/579 Zarr-v3 layout with atomic chunk writes) doubles as a live-mode fixture and a demo.

---

## 16. Open items / future

- **Upstream (APPROVED 2026-07-05, §7) — acquisition-side task:** write a per-FOV `coordinates.csv` on the record+zstack path (PR #564/#579), mirroring classic multipoint's columns → mm-exact mosaic; the viewer reads coordinates directly and the reconstruction module is fallback-only. Tracked at `AI-docs/Squid/to-do/2026-07-05-record-zstack-persist-fov-coordinates.md`.
- **Upstream — live-viewing affordances (approved 2026-07-06; verified squid gaps).** Squid already writes config/plate/channel sidecars at run start and declares full store shapes up front, but has **no** progress heartbeat, seals z-stack completion only at teardown, and drives **no** in-process live signal a standalone viewer could use. So:
  - **`events.jsonl` live status/heartbeat** (`fov_started / heartbeat / store_finalized / timepoint_done / aborted / run_done`, + a `run_started` manifest with plate format & expected FOV counts) → O(1) live extent, current-position, coverage k/N, prompt abort; also gives per-FOV z-stack completeness mid-run. Tracked at `AI-docs/Squid/to-do/2026-07-06-record-zstack-live-status-and-coverage.md`.
  - **Per-FOV thumbnail / live plate overview** → instant, read-free plate map. Tracked at `AI-docs/Squid/to-do/2026-07-06-record-zstack-fov-thumbnails.md`.
  - *(Optional, in-app only)* wire the recording phase to a display callback so squid's own live view shows recordings — noted in the live-status doc; irrelevant to the standalone viewer.
- **Legacy format adapters** (INDIVIDUAL_IMAGES / OME-TIFF / HCS-plate zarr) behind the existing seam.
- **True 3D / MIP** for z-stacks.
- **Seamless stitched mosaic** (needs exact XY) and cross-FOV overview zoom.

---

## 17. Why this meets the requirements

- **Very little memory:** O(visible + bounded caches); TensorStore per-tile reads bounded by a single shared `cache_pool`; handle + thumbnail LRUs; no whole-array loads.
- **Ongoing acquisitions:** LiveEngine incremental refresh; safe reads of partially-written stores; pending-aware rendering.
- **Plate navigation:** first-class 3-tier spatial navigator with live status, reconstructed from `acquisition.yaml` geometry.
- **Recording (single-channel) + z-stack (multichannel):** one ndv viewer with two modes handling both, including #579 multi-plane recordings.
- **Standalone desktop on the acquisition PC:** pure-Python, GPU via ndv/vispy, reuses the validated stack; Qt-free core for testability.
- **CPU-only viewing:** relies on ndv entirely — integrated-GPU OpenGL (present on Intel/AMD iGPUs) runs ndv/vispy hardware-accelerated without a discrete card; the rare fully-GL-less host uses software OpenGL (llvmpipe) via an env var. Navigator + thumbnails are pure-CPU regardless. No custom CPU renderer.
