"""Task 2.2 — `ExperimentModel`: scan-tree enumeration, per-FOV status, and the
translation-first positioning chain.

The positioning priority is exercised end to end:
  1. per-store OME translation (`FovMetadata.position_um`, um -> mm)  -- primary
  2. `coordinates.csv` at the experiment root                        -- fallback
  3. geometry reconstruction from `acquisition.yaml`                 -- fallback
  4. no `wellplate_scan` -> flat FOV list, `xy_mm=None`              -- last resort

`tiny_experiment` / `tiny_experiment_poor` both carry OME translations (the
fixture emits them even with `rich=False`), so the reconstruction (step 3) and
flat-list (step 4) tests build fixtures with translations STRIPPED — the only way
to force the model past step 1.
"""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import tensorstore as ts

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.model.experiment import ExperimentModel, FovNode
from record_zstack_viewer.reader.metadata import (
    fov_metadata_from_attrs, read_store_zarr_json)
from tests.fixtures.generate import generate_experiment


def _strip_translations(root: Path) -> None:
    """Drop the OME `translation` transform from every store's zarr.json (keep
    `scale`), so `FovMetadata.position_um` is None and the model must fall back."""
    for tree in ("recording", "zarr"):
        base = root / tree
        if not base.is_dir():
            continue
        for store in base.rglob("*.ome.zarr"):
            zj = store / "zarr.json"
            if not zj.exists():
                continue
            meta = json.loads(zj.read_text())
            try:
                ds = meta["attributes"]["ome"]["multiscales"][0]["datasets"][0]
            except (KeyError, IndexError, TypeError):
                continue
            xforms = ds.get("coordinateTransformations")
            if xforms:
                ds["coordinateTransformations"] = [
                    t for t in xforms if t.get("type") != "translation"]
            zj.write_text(json.dumps(meta, indent=2))


def _assert_positions_absent(root: Path) -> None:
    """Precondition guard: no store still carries a translation (step 1 is dead)."""
    for tree in ("recording", "zarr"):
        for store in (root / tree).rglob("*.ome.zarr"):
            attrs = read_store_zarr_json(store).get("attributes") or {}
            assert fov_metadata_from_attrs(attrs).position_um is None


# --- enumeration + status + translation-first positioning (steps 1) ----------

def test_model_lists_wells_and_fovs(tiny_experiment):
    m = ExperimentModel(tiny_experiment, default_context())
    assert set(m.wells()) == {"A1", "B2"}
    f = m.fovs("A1")[0]
    assert isinstance(f, FovNode)
    assert f.has_recording and f.has_zstack
    assert f.status == "complete"
    assert f.xy_mm is not None                       # from the OME translation


def test_model_xy_from_translation(tiny_experiment):
    """A1/fov_0 translation is (x=10000um, y=10000um) -> (10.0, 10.0) mm."""
    m = ExperimentModel(tiny_experiment, default_context())
    assert m.fovs("A1")[0].xy_mm == pytest.approx((10.0, 10.0))
    # B2 sits one 96-well pitch (9 mm) to the right in x
    assert m.fovs("B2")[0].xy_mm == pytest.approx((19.0, 10.0))


def test_model_open_fov_returns_both_phases(tiny_experiment):
    m = ExperimentModel(tiny_experiment, default_context())
    ds = m.open_fov("A1", 0)
    assert ds.recording is not None and ds.zstack is not None
    assert ds.zstack.dims == ("acq_t", "c", "z", "y", "x")


def test_model_plate_format_and_expected(tiny_experiment):
    m = ExperimentModel(tiny_experiment, default_context())
    assert m.plate_format() == "96"                  # region pitch 9 mm -> 96-well
    assert m.fov_count_expected("A1") == 1           # from events.jsonl run_started


def test_refresh_is_idempotent(tiny_experiment):
    m = ExperimentModel(tiny_experiment, default_context())
    before = {(n.well, n.fov, n.status, n.xy_mm) for w in m.wells() for n in m.fovs(w)}
    m.refresh()
    after = {(n.well, n.fov, n.status, n.xy_mm) for w in m.wells() for n in m.fovs(w)}
    assert before == after


def test_refresh_atomic_under_concurrent_readers(tiny_experiment):
    """`refresh()` swaps state atomically: readers hammering `wells()`/`fovs()` while
    another thread `refresh()`es must never see a partial/empty tree nor raise.

    The old piecemeal rebuild (`self._nodes = {}` then an in-place fill loop) fails
    this — a reader catches either an empty/partial `wells()` set or a `RuntimeError:
    dictionary changed size during iteration` from `sorted(self._nodes)` racing the
    fill. Mirrors the 4.2a HandleCache concurrency test (barrier + 1e-6 switch
    interval, restored in `finally`, errors collected and asserted on the main thread).
    """
    m = ExperimentModel(tiny_experiment, default_context())
    valid = {"A1", "B2"}                                 # the only complete well set
    assert set(m.wells()) == valid

    errors: list[BaseException] = []
    stop = threading.Event()
    n_readers = 8
    start = threading.Barrier(n_readers + 1)             # readers + the one refresher

    def reader() -> None:
        try:
            start.wait()                                 # release all threads together
            while not stop.is_set():
                got = set(m.wells())
                assert got == valid, got                 # never empty/partial while data exists
                for w in got:
                    m.fovs(w)                            # exercise the dict `.get` path too
        except BaseException as e:                       # capture; assert on the main thread
            errors.append(e)

    def refresher() -> None:
        try:
            start.wait()
            for _ in range(500):
                m.refresh()
        except BaseException as e:
            errors.append(e)
        finally:
            stop.set()                                   # let readers finish

    # Force near-per-bytecode thread switches so the window between the (old) node-dict
    # clear and its refill is actually interleaved with a reader.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=reader) for _ in range(n_readers)]
        threads.append(threading.Thread(target=refresher))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)
    assert not errors, errors[0]


# --- refresh() surfaces newly-written data (drops long-lived cached handles) --

def _grow_recording(root: Path, well: str, fov: int, *, first: int, last: int,
                    value: int) -> None:
    """Write frames [first, last) into the single-plane recording store for
    (well, fov) at t0 and bump its `_squid` (captured_frames / acquisition_complete),
    simulating a live acquisition growing the store on disk."""
    store = root / "recording" / "t0" / well / f"fov_{fov}.ome.zarr"
    arr = ts.open({"driver": "zarr3",
                   "kvstore": {"driver": "file", "path": str(store)}}).result()
    h, w = arr.shape[-2:]
    for f in range(first, last):
        arr[f, 0, 0].write(np.full((h, w), value, dtype="uint16")).result()
    zj = store / "zarr.json"
    meta = json.loads(zj.read_text())
    meta["attributes"]["_squid"]["captured_frames"] = last
    meta["attributes"]["_squid"]["acquisition_complete"] = True
    zj.write_text(json.dumps(meta, indent=2))


def test_refresh_surfaces_newly_written_data(tmp_path):
    """A plane that was pending (fill 0) at open reads its real data after the store
    grows and refresh() drops the cached handle.

    Reads go through long-lived cached TensorStore handles opened with
    recheck_cached_data="open", which serve a previously-pending plane's stale fill
    even after it is written; refresh() must re-open so the next read surfaces it.
    """
    root = generate_experiment(
        tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
        recording_frames=6, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2,
        rich=False, complete=False)                  # the only FOV is left partial

    m = ExperimentModel(root, default_context())
    rec = m.open_fov("A1", 0).recording
    idx = {"acq_t": 0, "frame": 5, "z": 0}
    assert rec.filled_extent("frame") == 3           # captured_frames=3 of 6 (partial)
    assert int(rec.read_plane(idx).max()) == 0       # frame 5 pending -> fill (caches handle)

    _grow_recording(root, "A1", 0, first=3, last=6, value=1234)

    m.refresh()                                       # must drop the cached store handle
    assert rec.filled_extent("frame") == 6            # the grown extent is now visible
    assert int(rec.read_plane(idx).max()) == 1234     # REAL data, not stale fill 0


def test_refresh_reopened_fov_reads_grown_data(tmp_path):
    """A FOV re-opened after refresh() (the live-UI flow) also sees the new data —
    the fresh source is built from a re-opened handle, not a stale cached one."""
    root = generate_experiment(
        tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
        recording_frames=6, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2,
        rich=False, complete=False)

    m = ExperimentModel(root, default_context())
    idx = {"acq_t": 0, "frame": 5, "z": 0}
    assert int(m.open_fov("A1", 0).recording.read_plane(idx).max()) == 0   # pending

    _grow_recording(root, "A1", 0, first=3, last=6, value=1234)
    m.refresh()

    rec2 = m.open_fov("A1", 0).recording              # re-open, as the UI does on_change
    assert rec2.filled_extent("frame") == 6
    assert int(rec2.read_plane(idx).max()) == 1234


# --- step 2: coordinates.csv (translations stripped, csv present) ------------

def test_model_positions_from_coordinates_csv(tmp_path):
    root = generate_experiment(
        tmp_path / "exp", wells=["A1"], fovs_per_well=2, nt=1,
        recording_frames=3, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2, rich=False)
    _strip_translations(root)                        # step 1 unavailable
    # write coords far from the A1 region center (10, 10) so csv is distinguishable
    # from a geometry reconstruction
    (root / "coordinates.csv").write_text(
        "region,fov,z_level,x (mm),y (mm),z (um)\n"
        "A1,0,0,42.000,7.000,3000.0\n"
        "A1,1,0,43.000,7.000,3000.0\n")
    m = ExperimentModel(root, default_context())
    assert m.fovs("A1")[0].xy_mm == pytest.approx((42.0, 7.0))
    assert m.fovs("A1")[1].xy_mm == pytest.approx((43.0, 7.0))


# --- step 3: geometry reconstruction (translations stripped, no coords.csv) ---

def test_model_reconstructs_from_geometry(tmp_path):
    root = generate_experiment(
        tmp_path / "exp", wells=["A1"], fovs_per_well=6, nt=1,
        recording_frames=3, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2, rich=False)
    _strip_translations(root)
    _assert_positions_absent(root)                   # step 1 unavailable
    assert not (root / "coordinates.csv").exists()   # step 2 unavailable (rich=False)

    m = ExperimentModel(root, default_context())
    fovs = m.fovs("A1")
    assert len(fovs) == 6
    xy = fovs[0].xy_mm
    assert xy is not None                            # reconstructed via geometry
    # A1 region center is (10, 10) mm; a reconstructed grid clusters tightly there
    assert abs(xy[0] - 10.0) < 1.0 and abs(xy[1] - 10.0) < 1.0
    # every FOV gets a distinct reconstructed position (no two map to one cell)
    assert len({f.xy_mm for f in fovs}) == 6


def test_reconstruction_uses_raster_fov_order(tmp_path):
    """Reconstruction (step 3) must place FOVs in squid's DEFAULT intra-region order,
    Unidirectional (raster — every row left->right), NOT serpentine.

    Pins the reconstructed FOV order for a multi-row grid (2 rows x 3 cols): the
    second (odd) row must NOT be column-reversed. Fails on a serpentine default.
    """
    root = generate_experiment(
        tmp_path / "exp", wells=["A1"], fovs_per_well=6, nt=1,
        recording_frames=3, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2, rich=False)
    _strip_translations(root)
    _assert_positions_absent(root)                   # step 1 unavailable
    assert not (root / "coordinates.csv").exists()   # step 2 unavailable -> forces recon

    m = ExperimentModel(root, default_context())
    xy = {f.fov: f.xy_mm for f in m.fovs("A1")}
    assert len(xy) == 6 and all(v is not None for v in xy.values())

    # reconciled grid is 3 cols x 2 rows: fov 0,1,2 = row 0; fov 3,4,5 = row 1.
    assert len({round(v[1], 6) for v in xy.values()}) == 2      # two distinct rows (y)
    assert len({round(v[0], 6) for v in xy.values()}) == 3      # three distinct cols (x)
    assert xy[3][1] > xy[0][1]                                  # row 1 sits below row 0

    # RASTER: both rows run left->right. The odd row is NOT reversed, so its x
    # increases with fov index and aligns column-for-column with the first row.
    assert xy[0][0] < xy[1][0] < xy[2][0]                       # row 0 left->right
    assert xy[3][0] < xy[4][0] < xy[5][0]                       # row 1 left->right (serpentine -> reversed, FAILS)
    assert xy[3][0] == pytest.approx(xy[0][0])                  # fov 3 back at col 0 (serpentine puts it at col 2)
    assert xy[5][0] == pytest.approx(xy[2][0])                  # fov 5 at col 2 (serpentine puts it at col 0)


# --- step 4: no wellplate_scan -> flat list, xy_mm None ----------------------

def test_current_position_flat_list_no_positions(tmp_path):
    root = generate_experiment(
        tmp_path / "exp", wells=["Manual"], fovs_per_well=3, nt=1,
        recording_frames=3, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2, rich=False)
    _strip_translations(root)
    # a "Current Position" run has no wellplate_scan block to reconstruct from
    (root / "acquisition.yaml").write_text("acquisition_mode: Current Position\n")

    m = ExperimentModel(root, default_context())
    assert m.wells() == ["Manual"]                   # non-well region id tolerated
    fovs = m.fovs("Manual")
    assert len(fovs) == 3
    assert all(f.xy_mm is None for f in fovs)        # nothing to reconstruct from
    assert m.plate_format() == "unknown"
    assert m.fov_count_expected("Manual") is None
