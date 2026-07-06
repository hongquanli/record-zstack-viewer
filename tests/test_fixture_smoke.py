import json

import tensorstore as ts


def test_fixture_has_recording_and_zstack(tiny_experiment):
    # tiny_experiment has recording_planes=3 -> multi-plane naming fov_{k}_z{j}.ome.zarr (CONTRACT.md)
    rec = tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr"
    zs = tiny_experiment / "zarr" / "A1" / "fov_0.ome.zarr" / "0"
    assert rec.exists() and (zs).exists()
    a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(rec)}}).result()
    assert a.shape[0] >= 1  # frames on T axis


def test_fixture_conforms_to_contract(tiny_experiment):
    # tiny_experiment: nt=2, wells [A1,B2], fovs_per_well=1, complete=True, rich=True

    # channels live at attributes.ome.omero.channels[], NOT ome.channels (CONTRACT.md)
    rec_meta = json.loads((tiny_experiment / "recording" / "t0" / "A1"
                           / "fov_0_z0.ome.zarr" / "zarr.json").read_text())
    zs_meta = json.loads((tiny_experiment / "zarr" / "A1" / "fov_0.ome.zarr"
                          / "zarr.json").read_text())
    for meta in (rec_meta, zs_meta):
        ome = meta["attributes"]["ome"]
        assert "channels" not in ome
        assert ome["omero"]["channels"]
    assert [c["label"] for c in zs_meta["attributes"]["ome"]["omero"]["channels"]] == ["DAPI", "GFP"]

    events = [json.loads(l) for l in
              (tiny_experiment / "events.jsonl").read_text().splitlines()]
    # exactly one fov_started per (t, region, fov, phase) -- both phases
    starts = sorted((e["t"], e["region"], e["fov"], e["phase"])
                    for e in events if e["type"] == "fov_started")
    assert starts == sorted((t, wl, 0, ph) for t in (0, 1) for wl in ("A1", "B2")
                            for ph in ("recording", "zstack"))
    # z-stack phase emits store_finalized too (one per (t, region, fov) when complete)
    zfin = [e for e in events if e["type"] == "store_finalized" and e["phase"] == "zstack"]
    assert len(zfin) == 4 and all(e["acquisition_complete"] and "path" in e for e in zfin)
    # complete run ends with run_done
    assert events[-1]["type"] == "run_done"

    # coordinates.csv: nt=2 (time-lapse) -> time column, one distinct row per (region, fov, t)
    lines = (tiny_experiment / "coordinates.csv").read_text().splitlines()
    assert lines[0] == "region,fov,z_level,x (mm),y (mm),z (um),time"
    assert len(lines) - 1 == 2 * 2 * 1 and len(set(lines[1:])) == len(lines) - 1
