# tests/test_contract.py
def test_contract_version_is_semver():
    from record_zstack_viewer.contract.version import CONTRACT_VERSION
    parts = CONTRACT_VERSION.split(".")
    assert len(parts) == 2 and all(p.isdigit() for p in parts)


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


def test_validator_warns_without_translations(tiny_experiment_poor):
    import json
    from record_zstack_viewer.contract.validate import validate_experiment
    # after the amendment even the poor fixture emits translations (rich=False only drops
    # sidecars), so no translation warning -- and its single-plane recording store sits at z_ref
    rec_zj = tiny_experiment_poor / "recording" / "t0" / "A1" / "fov_0.ome.zarr" / "zarr.json"
    ct = json.loads(rec_zj.read_text())["attributes"]["ome"]["multiscales"][0][
        "datasets"][0]["coordinateTransformations"]
    assert ct[1]["translation"][2] == 3000.0  # single plane: z_stage_um = z_ref_um
    assert not any("no per-FOV translation" in w
                   for w in validate_experiment(tiny_experiment_poor).warnings)
    # simulate old data: strip coordinateTransformations from every store's OME metadata
    for zj in tiny_experiment_poor.rglob("zarr.json"):
        meta = json.loads(zj.read_text())
        try:
            meta["attributes"]["ome"]["multiscales"][0]["datasets"][0].pop(
                "coordinateTransformations")
        except (KeyError, IndexError):
            continue  # array node under <zstack>/0 carries no OME attributes
        zj.write_text(json.dumps(meta))
    r = validate_experiment(tiny_experiment_poor)
    assert r.ok  # old data lacking translations is legitimate: warning, NOT error
    tw = [w for w in r.warnings if "no per-FOV translation" in w]
    assert any("recording" in w for w in tw) and any(f"{tiny_experiment_poor}/zarr/" in w for w in tw)


def test_validator_flags_violations(tmp_path):
    from record_zstack_viewer.contract.validate import validate_experiment
    from tests.fixtures.generate import generate_experiment, _open_create
    root = generate_experiment(tmp_path / "exp", wells=["A1"], fovs_per_well=1, nt=1,
        recording_frames=2, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=2)
    # 1. required config sidecar missing
    (root / "acquisition.yaml").unlink()
    # 2. unopenable store: strip its zarr.json
    broken = root / "recording" / "t0" / "A1" / "fov_0.ome.zarr"
    (broken / "zarr.json").unlink()
    # 3. wrong-rank z-stack array (3-D instead of 5-D TCZYX)
    _open_create(root / "zarr" / "BAD" / "fov_0.ome.zarr" / "0", (2, 3, 4), (2, 3, 4))
    r = validate_experiment(root)
    assert r.ok is False
    assert any("acquisition.yaml" in e for e in r.errors)
    assert any(str(broken) in e for e in r.errors)
    assert any("ndim" in e for e in r.errors)
