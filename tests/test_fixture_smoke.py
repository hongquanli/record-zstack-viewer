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


def _ct(store):  # datasets[0].coordinateTransformations of a store's OME metadata
    meta = json.loads((store / "zarr.json").read_text())
    return meta["attributes"]["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]


def test_fixture_has_translations(tiny_experiment):
    # every recording + z-stack store: [scale, translation], scale FIRST (OME-NGFF rule)
    stores = (sorted(tiny_experiment.glob("recording/t*/*/*.ome.zarr"))
              + sorted(tiny_experiment.glob("zarr/*/*.ome.zarr")))
    assert len(stores) == 2*2*3 + 2  # nt*wells*planes recording stores + 2 z-stack stores
    for s in stores:
        assert [x["type"] for x in _ct(s)] == ["scale", "translation"], s

    # axes are enriched with type/unit (squid style), still t,c,z,y,x
    axes = json.loads((tiny_experiment / "zarr" / "A1" / "fov_0.ome.zarr" / "zarr.json")
                      .read_text())["attributes"]["ome"]["multiscales"][0]["axes"]
    assert [a["name"] for a in axes] == list("tczyx")
    assert axes[0] == {"name": "t", "type": "time", "unit": "second"}
    assert axes[1] == {"name": "c", "type": "channel"}
    assert axes[2] == {"name": "z", "type": "space", "unit": "micrometer"}

    # XY translation = the SAME stage coords the fixture writes to coordinates.csv (mm -> um)
    rows = {(r[0], int(r[1])): (float(r[3]), float(r[4]))
            for r in (l.split(",") for l in
                      (tiny_experiment / "coordinates.csv").read_text().splitlines()[1:])}
    for wl in ("A1", "B2"):
        x_um, y_um = (v * 1000 for v in rows[(wl, 0)])
        zct = _ct(tiny_experiment / "zarr" / wl / "fov_0.ome.zarr")
        assert zct[0]["scale"] == [60.0, 1.0, 1.0, 0.5, 0.5]  # zstack _squid t/z_step/px
        assert zct[1]["translation"] == [0.0, 0.0, 3000.0, y_um, x_um]  # Z = FOV ref (3 mm)
        # multi-plane recording siblings: share XY with each other AND the z-stack store,
        # Z = z_ref_um + plane_z_offset_um (-2/0/+2)
        for j, dz in enumerate((-2.0, 0.0, 2.0)):
            rct = _ct(tiny_experiment / "recording" / "t0" / wl / f"fov_0_z{j}.ome.zarr")
            assert rct[0]["scale"] == [1/15.0, 1.0, 2.0, 0.5, 0.5]  # recording _squid values
            assert rct[1]["translation"] == [0.0, 0.0, 3000.0 + dz, y_um, x_um]
