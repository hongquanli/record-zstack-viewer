"""Task 1.4 — RecordingFovSource multi-plane compositor: dims (acq_t,frame,z,y,x).

A multi-plane recording is one store PER (well, fov, plane): sibling
`fov_{k}_z{j}.ome.zarr` dirs under each `recording/t{t}/{well}/`, each an array
at the store root with shape (T,1,1,Y,X) whose T axis is the `frame` axis. The
compositor stacks the sibling planes (ordered by `_squid.plane_index`) into the
z axis and the `recording/t{t}` dirs into the acq_t axis, presenting the uniform
5-D FovSource surface. A lone `fov_{k}.ome.zarr` (no `_z`) is a single z plane.

The recording fixture seeds each frame with `t*1000 + wi*100 + k*10 + f`, so
distinct (acq_t, frame) yield distinct pixels. The dispatch tests below assert
`read_plane` returns the *ground-truth* store's frame, so a wrong acq_t/frame/z
dispatch fails. (Sibling planes share pixels by design in this fixture — cross-z
uniqueness is the z-stack fixture's job — so z is checked structurally, via the
ground-truth store path plus plane_z_offsets/shape.)
"""
import numpy as np

from record_zstack_viewer.cache.memory import default_context, open_array
from record_zstack_viewer.reader.zarr_v3 import build_recording_source


def _source(exp, well="A1", fov=0):
    return build_recording_source(exp, well=well, fov=fov, context=default_context())


def test_read_plane_dispatches_to_ground_truth_store(tiny_experiment):
    s = _source(tiny_experiment)
    # ground truth: the exact per-(t,plane) store, read at the requested frame
    gt = open_array(
        tiny_experiment / "recording" / "t1" / "A1" / "fov_0_z2.ome.zarr",
        default_context())
    np.testing.assert_array_equal(
        s.read_plane({"acq_t": 1, "frame": 4, "z": 2}), gt[4, 0, 0].read().result())


def test_frame_and_acq_t_are_discriminating(tiny_experiment):
    s = _source(tiny_experiment)
    base = s.read_plane({"acq_t": 1, "frame": 4, "z": 2})
    # different acq_t -> different seed -> different pixels (acq_t dispatch is real)
    assert not np.array_equal(base, s.read_plane({"acq_t": 0, "frame": 4, "z": 2}))
    # different frame -> different seed -> different pixels (frame -> store T axis)
    assert not np.array_equal(base, s.read_plane({"acq_t": 1, "frame": 0, "z": 2}))


def test_plane_z_offsets_ordered_by_plane_index(tiny_experiment):
    s = _source(tiny_experiment)
    # fixture per-plane offset is -2 + j*2 for plane_index j in 0..2
    assert s.metadata.plane_z_offsets == [-2.0, 0.0, 2.0]
    assert s.metadata.n_planes == 3


def test_position_um_from_translation(tiny_experiment):
    s = _source(tiny_experiment)
    assert s.metadata.position_um is not None
    x_um, y_um, _z_um = s.metadata.position_um
    assert (x_um, y_um) == (10000.0, 10000.0)  # A1 fov0 stage XY in um


def test_read_tile_matches_plane_slice(tiny_experiment):
    s = _source(tiny_experiment)
    idx = {"acq_t": 0, "frame": 2, "z": 1}
    tile = s.read_tile(idx, 8, 24, 4, 40)
    assert tile.shape == (16, 36)
    np.testing.assert_array_equal(tile, s.read_plane(idx)[8:24, 4:40])


def test_single_plane_metadata(tiny_experiment_poor):
    s = _source(tiny_experiment_poor)
    assert s.shape[2] == 1 and s.metadata.n_planes == 1
    assert s.read_plane({"acq_t": 0, "frame": 0, "z": 0}).shape == (64, 64)


# ---- Task 3.1: reads reuse one store handle (no per-read re-open) ------------

def test_reads_reuse_a_single_handle_per_store(tiny_experiment):
    """The per-read re-open flagged in 1.4/1.5 is gone: the target store is opened
    once via the source's HandleCache and reused across reads of the same store."""
    from record_zstack_viewer.cache.memory import HandleCache
    s = _source(tiny_experiment)
    assert isinstance(s._handles, HandleCache)
    store = s._store[(1, 2)]                            # the (acq_t=1, z=2) store
    s.read_plane({"acq_t": 1, "frame": 4, "z": 2})     # opens it
    handle = s._handles.get(store)
    n = len(s._handles)
    s.read_plane({"acq_t": 1, "frame": 0, "z": 2})     # same store, new frame
    s.read_plane({"acq_t": 1, "frame": 2, "z": 2})     # same store again
    assert len(s._handles) == n                          # not re-opened per read
    assert s._handles.get(store) is handle              # same cached handle reused
