# record-zstack-viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, memory-frugal desktop viewer for Squid "Record + Z-Stack" well-plate scans (OME-Zarr v3 from PR #564/#579) that navigates a plate, views per-FOV recordings and z-stacks, and updates live during acquisition.

**Architecture:** Seven Qt-free-core layers behind a thin Qt/ndv UI (reader → model/geometry → caches → live engine → navigator + FOV viewer). **Contract-first:** Phase 0 freezes `CONTRACT.md` and a fixture generator + validator; every later task is TDD against those fixtures, so the viewer is never blocked on the squid-side additions (they are a separate track — see `AI-docs/Squid/to-do/`). The viewer targets today's already-stable zarr layout and degrades gracefully when the new sidecars (`coordinates.csv`, `events.jsonl`, thumbnails) are absent.

**Tech Stack:** Python 3.11; TensorStore (zarr3 reads); ndv + vispy (GPU render, single path); qtpy→PyQt5 + pyqtgraph (UI); numpy; PyYAML; platformdirs; pytest.

**Design spec (authoritative):** `docs/specs/2026-07-05-record-zstack-viewer-design.md`. Read §2 (data format) and §7–§11 before starting.

## Global Constraints

Every task's requirements implicitly include these:

- **Python ≥ 3.11.** Package name `record_zstack_viewer`.
- **Reads go through TensorStore only** (`driver: "zarr3"`). Never use napari-ome-zarr (no Zarr v3 support).
- **No operation reads a whole array.** Only per-plane / per-chunk-tile reads sized to what is visible. Never prefetch a full T axis.
- **Memory is O(1) in dataset size.** The single decoded-pixel cache is a **TensorStore `cache_pool`** (byte cap, default 192 MB) shared by reader + ndv + prefetch. Plus a bounded store-handle LRU (default 32) and a bounded thumbnail RAM cache. No home-grown tile LRU.
- **Single ndv/OpenGL render path.** `ndv` imported **only** in `ui/fov_viewer.py`. No custom CPU renderer; CPU-only machines use integrated-GPU OpenGL, headless via `LIBGL_ALWAYS_SOFTWARE=1` (llvmpipe).
- **Qt via `qtpy`** (resolves to PyQt5, matching squid). Non-UI modules import **no** Qt.
- **Contract is versioned** (`record_zstack_viewer.contract.version.CONTRACT_VERSION`). Tolerate missing/partial sidecars — always have a fallback.
- **Thumbnail disk cache lives in `platformdirs.user_cache_dir`**, keyed by experiment path+id. Never write into the experiment folder by default.
- **Contrast auto-scales** (1–99.9 percentile of the current plane) on first display; `omero.window` is a full-dtype range → use only as clamp bounds.
- **Recording is modeled `(acq_t, frame, z, y, x)`** (acq_t = `recording/t{t}/` dirs; z = `fov_k_z{j}` siblings; frame = store T axis). **Z-stack is `(acq_t, c, z, y, x)`** (acq_t = store T axis). The `acq_t` selector is shared across modes.
- **Well status colors are hue-coded** (blue done / amber acquiring / grey not-yet / red-✕ aborted); pair with a shape cue for colorblind-safety.

---

## File structure

```
record_zstack_viewer/
├── __init__.py
├── __main__.py                 # CLI entry: python -m record_zstack_viewer [path]
├── contract/
│   ├── __init__.py
│   ├── version.py              # CONTRACT_VERSION
│   └── validate.py             # validate_experiment(path) -> ValidationReport
├── reader/
│   ├── __init__.py
│   ├── metadata.py             # FovMetadata, ChannelMeta, parse zarr.json/_squid/omero + yaml
│   ├── fov_source.py           # FovSource protocol + shared helpers
│   ├── zarr_v3.py              # tensorstore open; RecordingFovSource, ZStackFovSource
│   └── adapter.py              # FormatAdapter: pick source for a FOV's datasets
├── model/
│   ├── __init__.py
│   ├── geometry.py             # well-name math, plate layout, FOV-grid reconstruction (PURE)
│   └── experiment.py           # ExperimentModel: scan tree, status, coords
├── cache/
│   ├── __init__.py
│   ├── memory.py               # tensorstore Context/cache_pool + store-handle LRU
│   └── thumbnails.py           # ThumbnailCache (platformdirs + RAM LRU; prefer squid thumb)
├── live/
│   ├── __init__.py
│   └── engine.py               # EventLogTailer (primary) + PollWatcher (fallback) -> signals
└── ui/
    ├── __init__.py
    ├── app.py                  # QMainWindow shell; wires navigator + viewer + live
    ├── plate_navigator.py      # 3-tier navigator (custom Qt raster; no GL)
    └── fov_viewer.py           # ndv wrapper: DataWrapper over FovSource, 2 modes
tests/
├── conftest.py
├── fixtures/generate.py        # conforming experiment generator (static + live)
├── test_contract.py test_metadata.py test_geometry.py test_reader_recording.py
├── test_reader_zstack.py test_reader_multiplane.py test_experiment.py
├── test_thumbnails.py test_live_events.py test_live_poll.py
└── test_fov_viewer.py test_navigator.py
CONTRACT.md
pyproject.toml
README.md
```

---

# Phase 0 — Contract + fixtures + validator (the shared truth)

> **AS-BUILT NOTE (2026-07-06):** Phase 0 is implemented and committed on branch `phase0-contract`. The embedded code blocks below (the Task 0.2 `CONTRACT.md` text and the Task 0.3 fixture) are the *original* plan and are **superseded** by the committed artifacts. Deltas applied during execution: channels nested under `ome.omero` (not spread); manifest `Nz` split into `recording_nz`/`zstack_nz`; and a per-FOV `coordinateTransformations` `[scale, translation]` added (translation = stage position in µm). **Authoritative: `CONTRACT.md` + the committed code** (`tests/fixtures/generate.py`, `record_zstack_viewer/contract/validate.py`). Treat the Phase-0 snippets below as historical.

### Task 0.1: Package skeleton + `pyproject.toml` + contract version

**Files:**
- Create: `pyproject.toml`, `record_zstack_viewer/__init__.py`, `record_zstack_viewer/contract/__init__.py`, `record_zstack_viewer/contract/version.py`
- Test: `tests/test_contract.py`

**Interfaces:**
- Produces: `record_zstack_viewer.contract.version.CONTRACT_VERSION: str`

- [ ] **Step 1: Write the failing test**
```python
# tests/test_contract.py
def test_contract_version_is_semver():
    from record_zstack_viewer.contract.version import CONTRACT_VERSION
    parts = CONTRACT_VERSION.split(".")
    assert len(parts) == 2 and all(p.isdigit() for p in parts)
```
- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_contract.py -v` → FAIL (module not found).
- [ ] **Step 3: Write `pyproject.toml`**
```toml
[project]
name = "record-zstack-viewer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "tensorstore>=0.1.60", "numpy>=1.26", "ndv>=0.3", "vispy>=0.14",
  "qtpy>=2.4", "PyQt5>=5.15", "pyqtgraph>=0.13", "PyYAML>=6", "platformdirs>=4",
]
[project.optional-dependencies]
test = ["pytest>=8"]
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```
Then `record_zstack_viewer/__init__.py` (empty), `contract/__init__.py` (empty), and:
```python
# record_zstack_viewer/contract/version.py
CONTRACT_VERSION = "1.0"
```
- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_contract.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "chore: package skeleton + contract version"`

---

### Task 0.2: `CONTRACT.md` — the versioned data contract

**Files:** Create `CONTRACT.md`

This is documentation (no test). Write the full producer→consumer interface. Content:

````markdown
# record-zstack-viewer data contract — v1.0

Producer: Squid Record + Z-Stack (PR #564/#579). Consumer: record-zstack-viewer.
All paths relative to `{experiment}/`. Additive sidecars are OPTIONAL; the viewer
has a fallback for each. `format_version` gates changes.

## Stable (already produced by squid — verified)
- `recording/t{t}/{well}/fov_{k}.ome.zarr` (single plane) or `fov_{k}_z{j}.ome.zarr`
  (multi-plane). Array at STORE ROOT. Shape `(T,1,1,Y,X)`, T=video frames. 1 channel.
- `zarr/{well}/fov_{k}.ome.zarr/0`. Array at `/0` (group metadata one level up).
  Shape `(Nt,C,NZ,Y,X)`, Nt=acquisition timepoints.
- Zarr v3, OME-NGFF 0.5, TCZYX. Chunk `(1,1,1,Y,X)` (FULL_FRAME default) or tiled.
  Compression blosc-lz4 (FAST default; sharded under BALANCED/BEST). dtype uint16/uint8.
  `zarr.json` declares full shape at creation; unwritten chunks read as fill_value 0.
- Per-store `zarr.json attributes`:
  - `ome.omero.channels[]`: {label, color(hex), window{start,end,min,max}, active,
    emission_wavelength?}. NOTE window = full dtype range, NOT a display default.
  - `_squid`: {structure:"5D-TCZYX", pixel_size_um, z_step_um, time_increment_s,
    acquisition_complete(bool), aborted?(bool), captured_frames?, expected_frames?}.
    Multi-plane recording stores add {plane_index, plane_z_offset_um, n_planes};
    absence ⇒ single plane (n_planes=1, plane_index=0).
- `acquisition.yaml` (at run START): `wellplate_scan.regions[]{name,center_mm,shape}`,
  `scan_size_mm`, `overlap_percent`, `xy_mode`.
- `acquisition_channels.yaml` (at run START): objective, channels (names+colors).
- `.done` (empty, at run END; also on abort).

## Additive v1.0 (optional; fallbacks noted)
### coordinates.csv  (fallback: reconstruct from acquisition.yaml + geometry)
Experiment root. Columns: `region, fov, z_level, x (mm), y (mm), z (um)` (+ `time`
for time-lapse). One row per FOV. Mirrors classic multipoint.

### events.jsonl  (fallback: file-watch + poll)
Experiment root, append-only, one JSON object/line, each has `type`,`ts`,`format_version`.
- `run_started`: {plate_format, wells:[...], expected_fovs_per_well:{well:int},
  Nt, Nz, channels:[{label,color}], fps, duration_s}
- `fov_started`: {t, region, fov, phase:"recording"|"zstack", x_mm, y_mm, z_ref_mm}
- `heartbeat`: {t, region, fov, phase, written:int}   (throttled)
- `store_finalized`: {t, region, fov, phase, path, acquisition_complete, aborted, captured, expected}
- `timepoint_done`: {t}
- `aborted`: {reason}   |   `run_done`: {}

### thumbnails/  (fallback: compute from a decoded plane)
`thumbnails/{well}/fov_{k}.png` (~128–256 px). Optional running `downsampled/plate.png`.
````

- [ ] **Step 1:** Write `CONTRACT.md` with the content above.
- [ ] **Step 2: Commit** — `git add CONTRACT.md && git commit -m "docs: data contract v1.0"`

---

### Task 0.3: Fixture generator — conforming experiments (static)

**Files:** Create `tests/fixtures/generate.py`, `tests/conftest.py`; Test: `tests/test_fixture_smoke.py`

**Interfaces:**
- Produces:
  - `generate_experiment(root: Path, *, wells: list[str], fovs_per_well: int, nt: int, recording_frames: int, recording_planes: int, zstack_channels: list[tuple[str,str]], zstack_nz: int, hw: tuple[int,int]=(64,64), rich: bool=True, complete: bool=True) -> Path`
    — writes a full experiment. `rich=True` also writes coordinates.csv/events.jsonl/thumbnails; `rich=False` omits them (fallback fixtures). `complete=False` leaves the last FOV's stores partially written + no `.done`.
  - pytest fixtures `tiny_experiment` (rich, complete) and `tiny_experiment_poor` (no sidecars).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_fixture_smoke.py
import tensorstore as ts
def test_fixture_has_recording_and_zstack(tiny_experiment):
    rec = tiny_experiment / "recording" / "t0" / "A1" / "fov_0.ome.zarr"
    zs  = tiny_experiment / "zarr" / "A1" / "fov_0.ome.zarr" / "0"
    assert rec.exists() and (zs).exists()
    a = ts.open({"driver":"zarr3","kvstore":{"driver":"file","path":str(rec)}}).result()
    assert a.shape[0] >= 1  # frames on T axis
```
- [ ] **Step 2: Run to verify it fails** — FAIL (no fixture).
- [ ] **Step 3: Implement `tests/fixtures/generate.py`**
```python
from pathlib import Path
import json, numpy as np, tensorstore as ts

def _open_create(path: Path, shape, chunk, dtype="uint16"):
    return ts.open({
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunk)}},
            "data_type": dtype,
            "codecs": [{"name": "blosc",
                        "configuration": {"cname": "lz4", "clevel": 1, "shuffle": "shuffle"}}],
            "fill_value": 0,
        },
        "create": True, "delete_existing": True,
    }).result()   # VERIFY codec spelling against installed tensorstore if this errors

def _patch_zarr_json(store_dir: Path, attributes: dict):
    zj = store_dir / "zarr.json"
    meta = json.loads(zj.read_text())
    meta.setdefault("attributes", {}).update(attributes)
    zj.write_text(json.dumps(meta, indent=2))

def _omero(channels):  # channels: list[(label,color_hex)]
    return {"channels": [
        {"label": lbl, "color": col, "active": True,
         "window": {"start": 0, "end": 65535, "min": 0, "max": 65535}}
        for lbl, col in channels]}

def _synthetic(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = (2000*np.exp(-(((yy-h*0.4)**2+(xx-w*0.4)**2)/(2*(h*0.15)**2)))).astype("uint16")
    return (blob + rng.integers(0, 300, size=(h, w))).astype("uint16")

def generate_experiment(root, *, wells, fovs_per_well, nt, recording_frames,
                        recording_planes, zstack_channels, zstack_nz,
                        hw=(64, 64), rich=True, complete=True):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    h, w = hw
    events = []
    def ev(**kw): events.append({**kw, "ts": 0.0, "format_version": "1.0"})
    ev(type="run_started", plate_format="96", wells=wells,
       expected_fovs_per_well={wl: fovs_per_well for wl in wells}, Nt=nt, Nz=recording_planes,
       channels=[{"label": l, "color": c} for l, c in zstack_channels],
       fps=15.0, duration_s=recording_frames/15.0)

    # config sidecars at START
    (root / "acquisition_channels.yaml").write_text(
        "objective: 20x\nchannels:\n" + "".join(f"  - name: {l}\n" for l, _ in zstack_channels))
    regions = "\n".join(f"    - name: {wl}\n      center_mm: [{10.0+i*9}, {10.0}]\n      shape: Square"
                        for i, wl in enumerate(wells))
    (root / "acquisition.yaml").write_text(
        "wellplate_scan:\n  xy_mode: Select Wells\n  scan_size_mm: 1.0\n  overlap_percent: 10\n"
        f"  regions:\n{regions}\n")

    coords = ["region,fov,z_level,x (mm),y (mm),z (um)"]
    n_wells = len(wells)
    for t in range(nt):
        for wi, wl in enumerate(wells):
            for k in range(fovs_per_well):
                last = (t == nt-1 and wi == n_wells-1 and k == fovs_per_well-1)
                partial = last and not complete
                x_mm, y_mm = 10.0 + wi*9 + (k % 3)*0.3, 10.0 + (k // 3)*0.3
                coords.append(f"{wl},{k},0,{x_mm:.3f},{y_mm:.3f},3000.0")
                # ---- recording: one store per plane ----
                for j in range(recording_planes):
                    name = f"fov_{k}.ome.zarr" if recording_planes == 1 else f"fov_{k}_z{j}.ome.zarr"
                    sd = root / "recording" / f"t{t}" / wl / name
                    arr = _open_create(sd, (recording_frames, 1, 1, h, w), (1, 1, 1, h, w))
                    n_written = recording_frames if not partial else max(1, recording_frames // 2)
                    ev(type="fov_started", t=t, region=wl, fov=k, phase="recording",
                       x_mm=x_mm, y_mm=y_mm, z_ref_mm=3.0)
                    for f in range(n_written):
                        arr[f, 0, 0].write(_synthetic(h, w, seed=t*1000+wi*100+k*10+f)).result()
                    squid = {"structure": "5D-TCZYX", "pixel_size_um": 0.5, "z_step_um": 2.0,
                             "time_increment_s": 1/15.0,
                             "acquisition_complete": (not partial), "captured_frames": n_written,
                             "expected_frames": recording_frames}
                    if recording_planes > 1:
                        squid |= {"plane_index": j, "plane_z_offset_um": -2.0 + j*2.0,
                                  "n_planes": recording_planes}
                    _patch_zarr_json(sd, {"ome": {"multiscales": [{"datasets": [{"path": "."}],
                                     "axes": [{"name": a} for a in "tczyx"]}], **_omero([("REC", "#FFFFFF")])},
                                     "_squid": squid})
                    if not partial:
                        ev(type="store_finalized", t=t, region=wl, fov=k, phase="recording",
                           path=str(sd), acquisition_complete=True, aborted=False,
                           captured=n_written, expected=recording_frames)
                # ---- z-stack: one store per (well,fov), T axis accumulates ----
                grp = root / "zarr" / wl / f"fov_{k}.ome.zarr"
                arrp = grp / "0"
                nc = len(zstack_channels)
                a = (_open_create(arrp, (nt, nc, zstack_nz, h, w), (1, 1, 1, h, w))
                     if not arrp.exists()
                     else ts.open({"driver": "zarr3",
                                   "kvstore": {"driver": "file", "path": str(arrp)}}).result())
                for c in range(nc):
                    for z in range(zstack_nz if not partial else 1):
                        a[t, c, z].write(_synthetic(h, w, seed=9+t+c*3+z)).result()
                grp.mkdir(parents=True, exist_ok=True)
                (grp / "zarr.json").write_text(json.dumps({
                    "zarr_format": 3, "node_type": "group",
                    "attributes": {"ome": {"multiscales": [{"datasets": [{"path": "0"}],
                                   "axes": [{"name": ax} for ax in "tczyx"]}],
                                   **_omero(zstack_channels)},
                                   "_squid": {"structure": "5D-TCZYX", "pixel_size_um": 0.5,
                                   "z_step_um": 1.0, "time_increment_s": 60.0,
                                   "acquisition_complete": (not partial)}}}, indent=2))
        ev(type="timepoint_done", t=t)

    if rich:
        (root / "coordinates.csv").write_text("\n".join(coords) + "\n")
        (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        thumbs = root / "thumbnails"
        for wl in wells:
            (thumbs / wl).mkdir(parents=True, exist_ok=True)
            for k in range(fovs_per_well):
                _write_png(thumbs / wl / f"fov_{k}.png", _synthetic(32, 32, k))
    if complete:
        (root / ".done").write_text("")
    return root

def _write_png(path: Path, arr8):
    # minimal 16-bit->8-bit PNG via imageio if available, else numpy .npy fallback
    try:
        import imageio.v3 as iio
        iio.imwrite(path, (arr8 / 256).astype("uint8"))
    except Exception:
        np.save(path.with_suffix(".npy"), arr8)
```
```python
# tests/conftest.py
import pytest
from tests.fixtures.generate import generate_experiment

@pytest.fixture
def tiny_experiment(tmp_path):
    return generate_experiment(tmp_path/"exp", wells=["A1","B2"], fovs_per_well=1, nt=2,
        recording_frames=5, recording_planes=3,
        zstack_channels=[("DAPI","#4a9eff"),("GFP","#34d058")], zstack_nz=4)

@pytest.fixture
def tiny_experiment_poor(tmp_path):
    return generate_experiment(tmp_path/"exp", wells=["A1"], fovs_per_well=6, nt=1,
        recording_frames=5, recording_planes=1,
        zstack_channels=[("DAPI","#4a9eff")], zstack_nz=3, rich=False)
```
- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_fixture_smoke.py -v`. If tensorstore rejects the codec/metadata dict, adjust to the installed spec (run `python -c "import tensorstore,sys;print(tensorstore.__version__)"`), then re-run.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "test: conforming fixture generator (static)"`

---

### Task 0.4: Contract validator

**Files:** Create `record_zstack_viewer/contract/validate.py`; Test: `tests/test_contract.py` (extend)

**Interfaces:**
- Produces: `validate_experiment(root: Path) -> ValidationReport` where `ValidationReport` is a dataclass `{ok: bool, errors: list[str], warnings: list[str]}`. Errors = stable-contract violations; warnings = missing optional sidecars.

- [ ] **Step 1: Write failing tests**
```python
def test_validator_passes_rich_fixture(tiny_experiment):
    from record_zstack_viewer.contract.validate import validate_experiment
    r = validate_experiment(tiny_experiment)
    assert r.ok and not r.errors

def test_validator_warns_on_missing_sidecars(tiny_experiment_poor):
    from record_zstack_viewer.contract.validate import validate_experiment
    r = validate_experiment(tiny_experiment_poor)
    assert r.ok  # stable contract still satisfied
    assert any("coordinates.csv" in w for w in r.warnings)
    assert any("events.jsonl" in w for w in r.warnings)
```
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `validate.py`** — walk `recording/` and `zarr/`, open each `.ome.zarr` (root or `/0`) with tensorstore, assert ndim==5 and the array-location rule; check `acquisition.yaml`/`acquisition_channels.yaml` exist (error if absent); check `coordinates.csv`/`events.jsonl`/`thumbnails/` (warning if absent). Return `ValidationReport`.
```python
from dataclasses import dataclass, field
from pathlib import Path
import tensorstore as ts

@dataclass
class ValidationReport:
    ok: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

def _open(path: Path):
    return ts.open({"driver":"zarr3","kvstore":{"driver":"file","path":str(path)}}).result()

def validate_experiment(root: Path) -> ValidationReport:
    root = Path(root); r = ValidationReport()
    if not (root/"acquisition.yaml").exists(): r.errors.append("missing acquisition.yaml")
    if not (root/"acquisition_channels.yaml").exists(): r.errors.append("missing acquisition_channels.yaml")
    for store in list((root/"recording").rglob("*.ome.zarr")):
        try:
            a = _open(store)
            if a.ndim != 5: r.errors.append(f"{store}: ndim {a.ndim} != 5")
        except Exception as e: r.errors.append(f"{store}: {e}")
    for grp in list((root/"zarr").rglob("*.ome.zarr")):
        try:
            a = _open(grp/"0")
            if a.ndim != 5: r.errors.append(f"{grp}/0: ndim {a.ndim} != 5")
        except Exception as e: r.errors.append(f"{grp}/0: {e}")
    for opt in ("coordinates.csv","events.jsonl","thumbnails"):
        if not (root/opt).exists(): r.warnings.append(f"optional {opt} absent (fallback used)")
    r.ok = not r.errors
    return r
```
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** — `git commit -am "feat: contract validator"`

---

# Phase 1 — Reader layer (TDD against fixtures)

### Task 1.1: Metadata parsing (`_squid` / `omero` / yaml)

**Files:** Create `record_zstack_viewer/reader/metadata.py`, `reader/__init__.py`; Test: `tests/test_metadata.py`

**Interfaces:**
- Produces:
  - `@dataclass ChannelMeta{label:str, color:str, window:tuple[float,float], active:bool}`
  - `@dataclass FovMetadata{pixel_size_um:float, z_step_um:float, time_increment_s:float, channels:list[ChannelMeta], acquisition_complete:bool, aborted:bool, n_planes:int, plane_z_offsets:list[float]|None}`
  - `read_store_zarr_json(store_dir:Path)->dict`
  - `fov_metadata_from_attrs(attrs:dict)->FovMetadata`

- [ ] **Step 1: Write failing test**
```python
def test_reads_channels_and_squid(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    grp = tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr"
    attrs = read_store_zarr_json(grp)["attributes"]
    m = fov_metadata_from_attrs(attrs)
    assert [c.label for c in m.channels] == ["DAPI","GFP"]
    assert m.channels[0].color == "#4a9eff"
    assert m.acquisition_complete is True

def test_single_plane_defaults(tiny_experiment_poor):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    st = tiny_experiment_poor/"recording"/"t0"/"A1"/"fov_0.ome.zarr"
    m = fov_metadata_from_attrs(read_store_zarr_json(st)["attributes"])
    assert m.n_planes == 1
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `metadata.py`** — `read_store_zarr_json` reads `<dir>/zarr.json`; `fov_metadata_from_attrs` reads `attrs["ome"]["omero"]["channels"]` → `ChannelMeta` (window = (start,end)); `attrs["_squid"]` → sizes/flags; `n_planes = _squid.get("n_planes", 1)`; `plane_z_offsets` collected by caller for multi-plane (see Task 1.4). Include `parse_acquisition_yaml(path)->dict` and `parse_channels_yaml(path)->list[ChannelMeta]` (PyYAML).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: reader metadata parsing"`

---

### Task 1.2: `FovSource` protocol + tensorstore open helper (with cache context)

**Files:** Create `reader/fov_source.py`, `cache/memory.py`, `cache/__init__.py`; Test: `tests/test_reader_recording.py` (start)

**Interfaces:**
- Produces:
  - `class FovSource(Protocol)`: attrs `dims:tuple[str,...]`, `shape:tuple[int,...]`, `dtype`, `metadata:FovMetadata`; methods `read_plane(index:dict[str,int])->np.ndarray`, `read_tile(index:dict[str,int], y0,y1,x0,x1)->np.ndarray`, `filled_extent(axis:str)->int`.
  - `cache.memory.default_context(byte_limit:int=192*1024*1024) -> tensorstore.Context` (configures `cache_pool.total_bytes_limit`).
  - `cache.memory.open_array(path, context) -> tensorstore array` (opens with shared context; used by all reads).

- [ ] **Step 1: Write failing test**
```python
def test_open_array_shares_cache_context(tiny_experiment):
    from record_zstack_viewer.cache.memory import default_context, open_array
    ctx = default_context()
    st = tiny_experiment/"recording"/"t0"/"A1"/"fov_0_z0.ome.zarr"
    a = open_array(st, ctx)
    assert a.ndim == 5 and a.shape[1:3] == (1,1)
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** `default_context` → `ts.Context({"cache_pool": {"total_bytes_limit": byte_limit}})`. `open_array(path, ctx)` → `ts.open({"driver":"zarr3","kvstore":{"driver":"file","path":str(path)},"context":..., "recheck_cached_data":"open"}, context=ctx).result()`. Define the `FovSource` Protocol in `fov_source.py`.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: FovSource protocol + tensorstore cache context"`

---

### Task 1.3: `ZStackFovSource` (array at `/0`, dims acq_t,c,z,y,x)

**Files:** Create/modify `reader/zarr_v3.py`; Test: `tests/test_reader_zstack.py`

**Interfaces:**
- Consumes: `open_array`, `read_store_zarr_json`, `fov_metadata_from_attrs`.
- Produces: `ZStackFovSource(group_dir:Path, context)` implementing `FovSource` with `dims=("acq_t","c","z","y","x")`.

- [ ] **Step 1: Write failing test**
```python
def test_zstack_source_reads_plane(tiny_experiment):
    from record_zstack_viewer.reader.zarr_v3 import ZStackFovSource
    from record_zstack_viewer.cache.memory import default_context
    s = ZStackFovSource(tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr", default_context())
    assert s.dims == ("acq_t","c","z","y","x")
    assert s.shape == (2,2,4,64,64)
    plane = s.read_plane({"acq_t":0,"c":1,"z":2})
    assert plane.shape == (64,64)
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** — open `group_dir/"0"`; `read_plane` → `arr[t,c,z].read().result()` (numpy); `read_tile` slices y/x; metadata from the group `zarr.json`; `filled_extent("acq_t")` from `_squid` or by probing which `t` slices are non-fill (see Task 1.5).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: ZStackFovSource"`

---

### Task 1.4: `RecordingFovSource` + multi-plane compositor (dims acq_t,frame,z,y,x)

**Files:** Modify `reader/zarr_v3.py`; Test: `tests/test_reader_recording.py`, `tests/test_reader_multiplane.py`

**Interfaces:**
- Produces: `RecordingFovSource(well_dir_by_t:dict[int,Path], fov:int, context)` implementing `FovSource` with `dims=("acq_t","frame","z","y","x")`. It maps `(acq_t,z)` → the store `recording/t{acq_t}/{well}/fov_{k}[_z{z}].ome.zarr`, `frame` → that store's T axis. `metadata.plane_z_offsets` ordered by `plane_index`.
- Consumes: `open_array`, metadata helpers.

- [ ] **Step 1: Write failing tests**
```python
def test_recording_source_dims_and_planes(tiny_experiment):
    from record_zstack_viewer.reader.zarr_v3 import build_recording_source
    from record_zstack_viewer.cache.memory import default_context
    s = build_recording_source(tiny_experiment, well="A1", fov=0, context=default_context())
    assert s.dims == ("acq_t","frame","z","y","x")
    assert s.shape == (2,5,3,64,64)              # 2 timepoints, 5 frames, 3 planes
    assert s.read_plane({"acq_t":1,"frame":4,"z":2}).shape == (64,64)

def test_single_plane_recording_has_z1(tiny_experiment_poor):
    from record_zstack_viewer.reader.zarr_v3 import build_recording_source
    from record_zstack_viewer.cache.memory import default_context
    s = build_recording_source(tiny_experiment_poor, well="A1", fov=0, context=default_context())
    assert s.shape[2] == 1
```
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** `build_recording_source(root, well, fov, context)` — glob `recording/t*/{well}/fov_{fov}*.ome.zarr`, group by t, order planes by `plane_index` (or single), build the source; `read_plane` dispatches to the right store + frame.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: RecordingFovSource + multi-plane compositor"`

---

### Task 1.5: `filled_extent` (FAST chunk-presence; sharded fallback)

**Files:** Modify `reader/zarr_v3.py`; Test: `tests/test_reader_live_extent.py`

**Interfaces:**
- Produces: `FovSource.filled_extent(axis)` returns written count along `frame`/`acq_t`; uses `_squid.captured_frames` when present, else counts present chunk files under FAST, else returns declared size with a `partial=True` flag surfaced via `is_extent_exact()->bool`.

- [ ] **Step 1: Write failing test** — build a `complete=False` fixture (half the frames), assert `filled_extent("frame")` < declared and `read_plane` on an unwritten frame returns all-zeros (fill_value) flagged pending.
- [ ] **Step 2: Run → fail.** → **Step 3: Implement** (scan the store's chunk keys once via `scandir`, not per-frame stat). → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: live filled-extent detection"`

---

### Task 1.6: `FormatAdapter` — open the right source(s) for a FOV

**Files:** Create `reader/adapter.py`; Test: `tests/test_reader_recording.py` (extend)

**Interfaces:**
- Produces: `open_fov(root, well, fov, context) -> FovDatasets` where `FovDatasets = {recording: FovSource|None, zstack: FovSource|None}`. Registry keyed by format so legacy adapters can be added later.

- [ ] Steps 1–5 (TDD): test that a FOV with both phases returns both sources; commit `git commit -am "feat: FormatAdapter open_fov"`.

---

# Phase 2 — Experiment model + geometry (PURE)

### Task 2.1: `geometry` — well names, plate layout, FOV grid + S-pattern

**Files:** Create `model/geometry.py`, `model/__init__.py`; Test: `tests/test_geometry.py`

**Interfaces:**
- Produces (all pure):
  - `well_name_to_rc(name:str)->tuple[int,int]` (`"A1"->(0,0)`, `"B12"->(1,11)`, multi-letter rows).
  - `PLATE_FORMATS: dict[str, PlateFormat]` with rows/cols/spacing_mm/a1_mm (6/12/24/96/384/1536, values from squid `sample_formats.csv`).
  - `infer_plate_format(region_centers_mm:list[tuple[float,float]])->str` (nearest spacing match; returns "unknown" if ambiguous).
  - `fov_grid_shape(scan_size_mm, overlap_percent, fov_size_mm)->tuple[int,int]`.
  - `spattern_index_to_ij(k:int, nx:int, ny:int)->tuple[int,int]` and `unidirectional_index_to_ij(...)`.

- [ ] **Step 1: Write failing tests**
```python
def test_well_name_to_rc():
    from record_zstack_viewer.model.geometry import well_name_to_rc
    assert well_name_to_rc("A1")==(0,0)
    assert well_name_to_rc("B12")==(1,11)
    assert well_name_to_rc("AA3")==(26,2)

def test_spattern_serpentine():
    from record_zstack_viewer.model.geometry import spattern_index_to_ij
    # 3-wide grid: row0 L→R, row1 R→L
    assert [spattern_index_to_ij(k,3,2) for k in range(6)] == [(0,0),(0,1),(0,2),(1,2),(1,1),(1,0)]
```
- [ ] **Step 2: Run → fail.** → **Step 3: Implement** (real math; `PLATE_FORMATS` copied from squid values in the spec §2 / sample_formats). → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: geometry (well names, plate formats, scan patterns)"`

---

### Task 2.2: `ExperimentModel` — scan tree, coordinates, status

**Files:** Create `model/experiment.py`; Test: `tests/test_experiment.py`

**Interfaces:**
- Produces:
  - `@dataclass FovNode{well:str, fov:int, has_recording:bool, has_zstack:bool, xy_mm:tuple[float,float]|None, status:str}` (`status ∈ {"empty","in_progress","complete","aborted"}`).
  - `ExperimentModel(root:Path, context)` with `wells()->list[str]`, `fovs(well)->list[FovNode]`, `fov_count_expected(well)->int|None`, `plate_format()->str`, `open_fov(well,fov)->FovDatasets`, `refresh()`.
  - Positioning (translation-first): read the per-store OME `coordinateTransformations.translation` when present (mm-exact); else `coordinates.csv`; else reconstruct via `geometry` from `acquisition.yaml`; else (no `wellplate_scan`) flat FOV list with `xy_mm=None`.

- [ ] **Step 1: Write failing tests**
```python
def test_model_lists_wells_and_fovs(tiny_experiment):
    from record_zstack_viewer.model.experiment import ExperimentModel
    from record_zstack_viewer.cache.memory import default_context
    m = ExperimentModel(tiny_experiment, default_context())
    assert set(m.wells()) == {"A1","B2"}
    f = m.fovs("A1")[0]
    assert f.has_recording and f.has_zstack and f.status == "complete"
    assert f.xy_mm is not None                     # from the store's OME translation (or coordinates.csv)

def test_model_reconstructs_without_coords(tiny_experiment_poor):
    from record_zstack_viewer.model.experiment import ExperimentModel
    from record_zstack_viewer.cache.memory import default_context
    m = ExperimentModel(tiny_experiment_poor, default_context())
    assert len(m.fovs("A1")) == 6
    assert m.fovs("A1")[0].xy_mm is not None        # reconstructed grid
```
- [ ] **Step 2: Run → fail.** → **Step 3: Implement** (walk `recording/`+`zarr/`, union FOVs, derive status from `_squid`+`.done`, positions per the 3-tier fallback). → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: ExperimentModel"`

---

# Phase 3 — Caches

### Task 3.1: Store-handle LRU

**Files:** Modify `cache/memory.py`; Test: `tests/test_cache_memory.py`

**Interfaces:**
- Produces: `HandleCache(context, max_open:int=32)` with `get(path)->ts array` (LRU; closes/evicts beyond `max_open`). Reader tasks switch to `HandleCache.get` instead of opening directly.

- [ ] Steps 1–5: test that opening >max_open distinct stores keeps `len(cache)<=max_open`; commit `git commit -am "feat: store-handle LRU"`.

---

### Task 3.2: `ThumbnailCache` (platformdirs + prefer squid thumbnail)

**Files:** Create `cache/thumbnails.py`; Test: `tests/test_thumbnails.py`

**Interfaces:**
- Produces: `ThumbnailCache(experiment_root, ram_items:int=512)` with `get(well,fov)->np.ndarray` (~128–256 px, uint8). Order: squid `thumbnails/{well}/fov_{k}.png` → disk cache in `platformdirs.user_cache_dir("record-zstack-viewer")/<hash(root)>/` → compute from a decoded plane (mid frame / mid-Z) and persist.

- [ ] **Step 1: Write failing tests**
```python
def test_thumbnail_prefers_squid_png(tiny_experiment):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    tc = ThumbnailCache(tiny_experiment)
    t = tc.get("A1", 0)
    assert t.ndim == 2 and t.dtype.name == "uint8"

def test_thumbnail_computes_when_absent(tiny_experiment_poor):
    from record_zstack_viewer.cache.thumbnails import ThumbnailCache
    tc = ThumbnailCache(tiny_experiment_poor)
    assert tc.get("A1", 0).ndim == 2          # computed
    # and it did NOT write into the experiment folder:
    assert not (tiny_experiment_poor/".ndviewer_cache").exists()
```
- [ ] **Step 2: Run → fail.** → **Step 3: Implement.** → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: ThumbnailCache (user-cache dir, squid-thumb preferred)"`

---

# Phase 4 — Live engine

### Task 4.1: `events.jsonl` tailer (primary live channel)

**Files:** Create `live/engine.py`, `live/__init__.py`; Test: `tests/test_live_events.py`

**Interfaces:**
- Produces: `EventLogTailer(path)` with `poll()->list[dict]` (returns new events since last call; tolerant of a half-written trailing line) and `has_log()->bool`.

- [ ] **Step 1: Write failing test**
```python
def test_tailer_reads_incrementally(tmp_path):
    from record_zstack_viewer.live.engine import EventLogTailer
    p = tmp_path/"events.jsonl"; p.write_text('{"type":"run_started"}\n')
    t = EventLogTailer(p)
    assert [e["type"] for e in t.poll()] == ["run_started"]
    with p.open("a") as f: f.write('{"type":"fov_started","region":"A1"}\n')
    assert [e["type"] for e in t.poll()] == ["fov_started"]
    with p.open("a") as f: f.write('{"type":"partial')   # half line
    assert t.poll() == []                                 # ignored until complete
```
- [ ] **Step 2: Run → fail.** → **Step 3: Implement** (track byte offset; only parse complete `\n`-terminated lines). → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: events.jsonl tailer"`

---

### Task 4.2: `LiveEngine` — events-primary, poll-fallback, emits updates

**Files:** Modify `live/engine.py`; Test: `tests/test_live_poll.py`

**Interfaces:**
- Produces: `LiveEngine(model, on_change:Callable[[set[tuple[str,int]]],None], on_status:Callable[[dict],None])` with `start()`/`stop()`. Uses `EventLogTailer` if `events.jsonl` exists; else a poll loop computing a cheap per-store signature (dir mtime + `_squid.acquisition_complete`). Both drive `model.refresh()` and fire `on_change` with the changed `(well,fov)` set. **No Qt import** — the tick can be driven by a Qt timer from the UI or a thread in tests.

- [ ] **Step 1: Write failing test** — start with a `complete=False` fixture, then append more frames + a `store_finalized` event (or grow the store for poll mode), call `engine.tick()`, assert `on_change` fires for the affected FOV and its status flips.
- [ ] **Step 2: Run → fail.** → **Step 3: Implement** (`tick()` public for tests; `start()` schedules `tick` on an interval). → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: LiveEngine (events + poll fallback)"`

---

# Phase 5 — UI (Qt + ndv)

> UI tests use `qtpy` + `pytest-qt` if available; otherwise direct instantiation with `QApplication.instance() or QApplication([])` and no event loop. Guard GL: set `os.environ.setdefault("QT_QPA_PLATFORM","offscreen")` in `conftest.py` for headless CI.

### Task 5.1: `fov_viewer.py` — ndv `DataWrapper` over `FovSource`

**Files:** Create `ui/fov_viewer.py`, `ui/__init__.py`; Test: `tests/test_fov_viewer.py`

**Interfaces:**
- Produces:
  - `FovSourceDataWrapper(source:FovSource)` — subclass of ndv's data-wrapper base so ndv slices via `source.read_plane`. (Verify the exact base/hook against installed ndv: `python -c "import ndv, inspect; from ndv.models import _data_wrapper as w; print([n for n in dir(w)])"`; implement the documented `DataWrapper` interface — `sizes()`, `isel()`/`__getitem__`, `dtype`.)
  - `FovViewer(QWidget)` with `set_datasets(datasets:FovDatasets, mode:str)`, `set_timepoint(t:int)`, `auto_contrast()`; a **Recording | Z-stack** toggle; recording exposes Frame scrub + separate acq_t Timepoint stepper + Z-plane pills; z-stack exposes Z slider + shared Timepoint + right-side channel dock. Contrast auto-scaled on first show; `omero.window` only as bounds. ndv imported **only here**.

- [ ] **Step 1: Write failing test**
```python
def test_datawrapper_slices_via_source(tiny_experiment):
    from record_zstack_viewer.reader.adapter import open_fov
    from record_zstack_viewer.cache.memory import default_context
    from record_zstack_viewer.ui.fov_viewer import FovSourceDataWrapper
    ds = open_fov(tiny_experiment, "A1", 0, default_context())
    w = FovSourceDataWrapper(ds.zstack)
    assert tuple(w.sizes().values()) == (2,2,4,64,64)   # acq_t,c,z,y,x
    plane = w.isel({"acq_t":0,"c":0,"z":0})
    assert plane.shape == (64,64)
```
- [ ] **Step 2: Run → fail** (also run the `inspect` probe above and adapt the wrapper to the real ndv API). → **Step 3: Implement wrapper + widget.** → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: FovViewer + ndv DataWrapper"`

*(Sub-tasks 5.1a auto-contrast, 5.1b recording frame/timepoint controls, 5.1c z-stack channel dock each follow the same TDD micro-cycle; keep ndv confined to this file.)*

---

### Task 5.2: `plate_navigator.py` — 3-tier navigator (custom Qt raster)

**Files:** Create `ui/plate_navigator.py`; Test: `tests/test_navigator.py`

**Interfaces:**
- Produces: `PlateNavigator(QWidget)` with `set_model(model)`, signal `fov_selected(well:str, fov:int)`, `set_status(well, fov, status)`, `update_coverage(done:int, total:int, current:tuple|None)`. Tier 1 plate map (hue-coded status + shape cue, zoom/pan, tooltips); adaptive FOV strip (shown only when a well has >1 FOV); pure Qt/`pyqtgraph` raster (no GL, no ndv import). Thumbnails via `ThumbnailCache`.

- [ ] **Step 1: Write failing test** — instantiate with `tiny_experiment` model (offscreen), assert `wells` render and clicking a single-FOV well emits `fov_selected(well,0)` without showing a 1-item strip. (Use signal spy or direct method call.)
- [ ] **Step 2: Run → fail.** → **Step 3: Implement.** → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: PlateNavigator (3-tier, hue status, adaptive FOV list)"`

---

### Task 5.3: `app.py` + `__main__.py` — shell wiring (Layout A)

**Files:** Create `ui/app.py`, `record_zstack_viewer/__main__.py`; Test: `tests/test_app_smoke.py`

**Interfaces:**
- Produces: `MainWindow(root:Path)` (Layout A: docked `PlateNavigator` left + `FovViewer` main; live coverage header). `__main__` parses `[path]`, `default_context()`, builds `ExperimentModel`, starts `LiveEngine` on a Qt timer, shows the window. Navigator `fov_selected` → `viewer.set_datasets(model.open_fov(...))`; `LiveEngine.on_change` → navigator status + (if open) `viewer.update_extent`; timepoint selector shared.

- [ ] **Step 1: Write failing smoke test** — `MainWindow(tiny_experiment)` constructs offscreen, `.navigator` and `.viewer` exist, selecting a FOV sets the viewer's datasets. 
- [ ] **Step 2: Run → fail.** → **Step 3: Implement.** → **Step 4: pass.** → **Step 5: Commit** `git commit -am "feat: app shell + entry point"`

---

# Phase 6 — Packaging & docs

### Task 6.1: README + run instructions + CPU/headless note

**Files:** Create `README.md`; Test: none (docs).
- [ ] Document: install (`pip install -e .`), run (`python -m record_zstack_viewer /path/to/experiment`), CPU-only note (`LIBGL_ALWAYS_SOFTWARE=1` for headless/RDP), and that it degrades gracefully without the optional sidecars. Commit `git commit -am "docs: README"`.

### Task 6.2: End-to-end live test with the growing fixture

**Files:** Add `tests/fixtures/generate.py::grow_experiment(root, step)` (append frames/timepoints incrementally); Test: `tests/test_e2e_live.py`
- [ ] TDD: start `MainWindow` on a fresh fixture, run `grow_experiment` a few steps calling `LiveEngine.tick()` between, assert coverage increases and a previously-`in_progress` FOV becomes `complete` without a widget rebuild. Commit `git commit -am "test: e2e live growth"`.

---

## Parallel track (NOT this plan — squid repo)

The three acquisition-side additions the contract references are separate tasks in `AI-docs/Squid/to-do/` (`2026-07-05-...coordinates`, `2026-07-06-...live-status-and-coverage`, `2026-07-06-...thumbnails`). The viewer works without them (fallbacks) and improves when they land; both sides validate against this plan's fixtures + `contract/validate.py`.

## Self-review notes (author)

- **Spec coverage:** reader/asymmetry §2/§4 → Tasks 1.1–1.6; geometry/positioning §7 → 2.1–2.2; caches/memory §9/§11 → 3.1–3.2; live §10 → 4.1–4.2; UI §6/§8 (Layout A, adaptive list, hue status, right dock, auto-contrast, frame/timepoint split) → 5.1–5.3; CPU/headless §8 → 6.1; contract §2 + upstream §16 → Phase 0.
- **Known verify-against-installed-version points (flagged inline, not placeholders):** tensorstore zarr3 create-spec/codec spelling (0.3); ndv `DataWrapper` base + `sizes()/isel()` names (5.1); ndv canvas/GL in headless tests. These are real API calls with a one-line verification command each.
- **Type consistency:** `FovSource` dims `(acq_t,frame,z,y,x)` / `(acq_t,c,z,y,x)` used consistently across reader, DataWrapper, and viewer; `FovDatasets{recording,zstack}` used by adapter, model, viewer, app.
