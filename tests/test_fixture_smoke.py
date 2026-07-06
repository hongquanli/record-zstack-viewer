import tensorstore as ts


def test_fixture_has_recording_and_zstack(tiny_experiment):
    # tiny_experiment has recording_planes=3 -> multi-plane naming fov_{k}_z{j}.ome.zarr (CONTRACT.md)
    rec = tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr"
    zs = tiny_experiment / "zarr" / "A1" / "fov_0.ome.zarr" / "0"
    assert rec.exists() and (zs).exists()
    a = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(rec)}}).result()
    assert a.shape[0] >= 1  # frames on T axis
