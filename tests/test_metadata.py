"""Task 1.1 — reader metadata parsing (omero channels, _squid, yaml sidecars,
per-FOV placement translation).

Fixture geometry (tests/conftest.py + tests/fixtures/generate.py): A1 fov0 sits at
x=10 mm, y=10 mm, z_ref=3 mm, so the OME translation (um, axis order t,c,z,y,x) is
[0, 0, 3000, 10000, 10000]; recording plane j sits at z = 3000 + (-2 + 2*j).
"""


def test_reads_channels_and_squid(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    grp = tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr"
    attrs = read_store_zarr_json(grp)["attributes"]
    m = fov_metadata_from_attrs(attrs)
    assert [c.label for c in m.channels] == ["DAPI", "GFP"]
    assert m.channels[0].color == "#4a9eff"
    assert m.acquisition_complete is True


def test_squid_scalars_window_and_flags(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    grp = tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr"
    m = fov_metadata_from_attrs(read_store_zarr_json(grp)["attributes"])
    assert m.pixel_size_um == 0.5
    assert m.z_step_um == 1.0
    assert m.time_increment_s == 60.0
    assert m.aborted is False
    assert m.channels[0].window == (0.0, 65535.0)  # full-dtype clamp bounds
    assert m.channels[0].active is True
    assert m.plane_z_offsets is None  # collected across sibling stores by the caller


def test_single_plane_defaults(tiny_experiment_poor):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    st = tiny_experiment_poor/"recording"/"t0"/"A1"/"fov_0.ome.zarr"
    m = fov_metadata_from_attrs(read_store_zarr_json(st)["attributes"])
    assert m.n_planes == 1


def test_multi_plane_recording_n_planes(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    st = tiny_experiment/"recording"/"t0"/"A1"/"fov_0_z1.ome.zarr"
    m = fov_metadata_from_attrs(read_store_zarr_json(st)["attributes"])
    assert m.n_planes == 3
    assert m.acquisition_complete is True


def test_position_um_from_translation(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    grp = tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr"
    m = fov_metadata_from_attrs(read_store_zarr_json(grp)["attributes"])
    assert m.position_um == (10000.0, 10000.0, 3000.0)  # (x, y, z) in um


def test_position_um_xy_not_swapped(tiny_experiment):
    # B2 fov 0 sits at x=19 mm, y=10 mm, so translation x != y. This discriminates the
    # (translation[4]=x, translation[3]=y) mapping from a swapped one — unlike A1 where x==y.
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    attrs = read_store_zarr_json(tiny_experiment/"zarr"/"B2"/"fov_0.ome.zarr")["attributes"]
    ct = attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    translation = next(x["translation"] for x in ct if x["type"] == "translation")
    assert translation[4] != translation[3]  # x != y, so an X/Y swap would be detectable here
    m = fov_metadata_from_attrs(attrs)
    assert m.position_um == (translation[4], translation[3], translation[2])  # (x, y, z) um


def test_recording_plane_shares_xy_with_zstack(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    zst = fov_metadata_from_attrs(read_store_zarr_json(
        tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr")["attributes"])
    rec = fov_metadata_from_attrs(read_store_zarr_json(
        tiny_experiment/"recording"/"t0"/"A1"/"fov_0_z0.ome.zarr")["attributes"])
    assert rec.position_um[:2] == zst.position_um[:2] == (10000.0, 10000.0)
    assert rec.position_um[2] == 2998.0  # plane 0: z_ref + (-2 + 0*2)


def test_position_none_when_translation_stripped(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    attrs = read_store_zarr_json(tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr")["attributes"]
    ct = attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    ct[:] = [x for x in ct if x["type"] != "translation"]  # simulate old data
    m = fov_metadata_from_attrs(attrs)
    assert m.position_um is None
    assert m.pixel_size_um == 0.5  # the rest of the metadata still parses


def test_scale_fallback_when_squid_missing(tiny_experiment):
    from record_zstack_viewer.reader.metadata import read_store_zarr_json, fov_metadata_from_attrs
    attrs = read_store_zarr_json(tiny_experiment/"zarr"/"A1"/"fov_0.ome.zarr")["attributes"]
    del attrs["_squid"]
    m = fov_metadata_from_attrs(attrs)
    assert m.pixel_size_um == 0.5  # scale[4] (x)
    assert m.z_step_um == 1.0      # scale[2] (z)
    assert m.time_increment_s == 60.0  # scale[0] (t)
    assert m.n_planes == 1
    assert m.acquisition_complete is False  # no evidence of completion


def test_empty_attrs_tolerated():
    from record_zstack_viewer.reader.metadata import fov_metadata_from_attrs
    m = fov_metadata_from_attrs({})
    assert m.channels == []
    assert m.n_planes == 1
    assert m.position_um is None
    assert m.plane_z_offsets is None
    assert m.acquisition_complete is False and m.aborted is False
    assert m.pixel_size_um == m.z_step_um == m.time_increment_s == 1.0  # neutral fallback


def test_parse_acquisition_yaml(tiny_experiment):
    from record_zstack_viewer.reader.metadata import parse_acquisition_yaml
    d = parse_acquisition_yaml(tiny_experiment/"acquisition.yaml")
    ws = d["wellplate_scan"]
    assert ws["xy_mode"] == "Select Wells"
    assert ws["scan_size_mm"] == 1.0
    assert ws["overlap_percent"] == 10
    assert [r["name"] for r in ws["regions"]] == ["A1", "B2"]
    assert ws["regions"][0]["center_mm"] == [10.0, 10.0]


def test_parse_channels_yaml(tiny_experiment):
    from record_zstack_viewer.reader.metadata import ChannelMeta, parse_channels_yaml
    chans = parse_channels_yaml(tiny_experiment/"acquisition_channels.yaml")
    assert [c.label for c in chans] == ["DAPI", "GFP"]
    assert all(isinstance(c, ChannelMeta) and c.active for c in chans)


def test_reader_package_reexports():
    from record_zstack_viewer import reader
    for name in ("ChannelMeta", "FovMetadata", "read_store_zarr_json",
                 "fov_metadata_from_attrs", "parse_acquisition_yaml", "parse_channels_yaml"):
        assert hasattr(reader, name)
