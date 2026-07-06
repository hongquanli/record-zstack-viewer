"""Task 1.3 — ZStackFovSource: the z-stack read interface, dims (acq_t,c,z,y,x).

A z-stack recording is a single store per (well, fov): the group node at
`zarr/{well}/fov_{k}.ome.zarr`, its pixel array at `/0`, shape (Nt,C,NZ,Y,X)
where the store T axis is the acquisition-timepoint axis (`acq_t`). These tests
open the Phase-0 fixture and assert the source exposes the uniform 5-D FovSource
surface, reads only the one visible plane (never the whole array), and returns a
numpy dtype (not a tensorstore dtype) so it conforms to the protocol.
"""
import numpy as np
import tensorstore as ts

from record_zstack_viewer.cache.memory import default_context, open_array
from record_zstack_viewer.reader.fov_source import FovSource
from record_zstack_viewer.reader.metadata import FovMetadata
from record_zstack_viewer.reader.zarr_v3 import ZStackFovSource


def _source(tiny_experiment):
    return ZStackFovSource(tiny_experiment / "zarr" / "A1" / "fov_0.ome.zarr",
                           default_context())


def test_zstack_source_reads_plane(tiny_experiment):
    s = _source(tiny_experiment)
    assert s.dims == ("acq_t", "c", "z", "y", "x")
    assert s.shape == (2, 2, 4, 64, 64)
    plane = s.read_plane({"acq_t": 0, "c": 1, "z": 2})
    assert plane.shape == (64, 64)


def test_zstack_source_conforms_to_protocol(tiny_experiment):
    assert isinstance(_source(tiny_experiment), FovSource)


def test_dtype_is_numpy_not_tensorstore(tiny_experiment):
    s = _source(tiny_experiment)
    # FovSource protocol + downstream expect a numpy dtype, not ts.dtype
    assert isinstance(s.dtype, np.dtype)
    assert s.dtype == np.uint16


def test_read_plane_returns_the_selected_plane(tiny_experiment):
    s = _source(tiny_experiment)
    plane = s.read_plane({"acq_t": 0, "c": 1, "z": 2})
    assert isinstance(plane, np.ndarray) and plane.dtype == np.uint16
    # matches a direct per-plane tensorstore read of the same index (no whole-array read)
    arr = open_array(tiny_experiment / "zarr" / "A1" / "fov_0.ome.zarr" / "0",
                     default_context())
    np.testing.assert_array_equal(plane, arr[0, 1, 2].read().result())


def test_read_tile_slices_yx(tiny_experiment):
    s = _source(tiny_experiment)
    idx = {"acq_t": 1, "c": 0, "z": 3}
    tile = s.read_tile(idx, 8, 24, 4, 40)
    assert tile.shape == (16, 36)
    np.testing.assert_array_equal(tile, s.read_plane(idx)[8:24, 4:40])


def test_metadata_from_group_zarr_json(tiny_experiment):
    s = _source(tiny_experiment)
    assert isinstance(s.metadata, FovMetadata)
    assert [c.label for c in s.metadata.channels] == ["DAPI", "GFP"]
    assert s.metadata.channels[0].color == "#4a9eff"
    assert s.metadata.acquisition_complete is True


def test_filled_extent_stub_reports_declared_count(tiny_experiment):
    # minimal stub for Task 1.3 (real chunk-presence probing is Task 1.5):
    # a complete store's acq_t extent equals its declared T size.
    s = _source(tiny_experiment)
    assert s.filled_extent("acq_t") == 2
