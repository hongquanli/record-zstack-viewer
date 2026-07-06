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
from pathlib import Path

import pytest

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
