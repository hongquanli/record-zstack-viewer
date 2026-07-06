"""Task 1.2 — FovSource protocol + TensorStore cache-pool context.

`open_array` opens a zarr3 store lazily through a shared `cache_pool` Context
(Global Constraints: memory is O(1) in dataset size via a single bounded
cache_pool shared by every read). This file starts the recording-reader tests;
the concrete RecordingFovSource lands in Task 1.4.

Sharing is asserted via `spec(retain_context=True)`, which surfaces the pool
resource the array is bound to — two opens with the same Context bind the
identical configured pool, a context-less open binds the empty default.
"""
import numpy as np
import tensorstore as ts

from record_zstack_viewer.cache.memory import default_context, open_array
from record_zstack_viewer.reader.fov_source import FovSource
from record_zstack_viewer.reader.metadata import FovMetadata
from tests.fixtures.generate import generate_experiment

_DEFAULT_LIMIT = 192 * 1024 * 1024


def _bound_pool(arr) -> dict:
    """The cache_pool resource `arr` is actually bound to."""
    return arr.spec(retain_context=True).to_json()["context"]["cache_pool"]


def test_open_array_shares_cache_context(tiny_experiment):
    ctx = default_context()
    st = tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr"
    a = open_array(st, ctx)
    assert a.ndim == 5 and a.shape[1:3] == (1, 1)


def test_default_context_is_a_context():
    assert isinstance(default_context(), ts.Context)


def test_default_context_configures_byte_limit(tiny_experiment):
    limit = 64 * 1024 * 1024
    a = open_array(tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr",
                   default_context(byte_limit=limit))
    assert _bound_pool(a)["total_bytes_limit"] == limit


def test_default_context_default_limit_is_192mb(tiny_experiment):
    a = open_array(tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr",
                   default_context())
    assert _bound_pool(a)["total_bytes_limit"] == _DEFAULT_LIMIT


def test_two_opens_share_the_same_pool(tiny_experiment):
    ctx = default_context()
    st = tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr"
    a, b = open_array(st, ctx), open_array(st, ctx)
    # both bind the one configured pool; a context-less open would bind {}
    assert _bound_pool(a) == _bound_pool(b) == {"total_bytes_limit": _DEFAULT_LIMIT}


def test_open_array_is_lazy_and_reads_per_plane(tiny_experiment):
    ctx = default_context()
    a = open_array(tiny_experiment / "recording" / "t0" / "A1" / "fov_0_z0.ome.zarr", ctx)
    # opening yields a lazy TensorStore array, not a materialized ndarray
    assert not isinstance(a, np.ndarray)
    plane = a[0, 0, 0].read().result()  # explicit per-plane read
    assert plane.shape == (64, 64) and plane.dtype == np.uint16


class _Stub:
    """Minimal structurally-conforming FovSource."""
    dims = ("acq_t", "frame", "z", "y", "x")
    shape = (2, 1, 1, 64, 64)
    dtype = np.dtype("uint16")
    metadata = FovMetadata(0.5, 2.0, 1.0, [], True, False, 1)

    def read_plane(self, index):
        return np.zeros((64, 64), "uint16")

    def read_tile(self, index, y0, y1, x0, x1):
        return np.zeros((y1 - y0, x1 - x0), "uint16")

    def filled_extent(self, axis):
        return 2

    def is_extent_exact(self):
        return True


def test_fov_source_protocol_accepts_conforming_object():
    assert isinstance(_Stub(), FovSource)


def test_fov_source_protocol_rejects_incomplete_object():
    class Missing:  # only one attr, no methods
        dims = ()

    assert not isinstance(Missing(), FovSource)


# ---- Task 1.4: RecordingFovSource + multi-plane compositor ------------------

def test_recording_source_dims_and_planes(tiny_experiment):
    from record_zstack_viewer.reader.zarr_v3 import build_recording_source
    s = build_recording_source(tiny_experiment, well="A1", fov=0, context=default_context())
    assert s.dims == ("acq_t", "frame", "z", "y", "x")
    assert s.shape == (2, 5, 3, 64, 64)                    # 2 timepoints, 5 frames, 3 planes
    assert s.read_plane({"acq_t": 1, "frame": 4, "z": 2}).shape == (64, 64)


def test_single_plane_recording_has_z1(tiny_experiment_poor):
    from record_zstack_viewer.reader.zarr_v3 import build_recording_source
    s = build_recording_source(tiny_experiment_poor, well="A1", fov=0, context=default_context())
    assert s.shape[2] == 1


def test_recording_source_conforms_to_protocol(tiny_experiment):
    from record_zstack_viewer.reader.zarr_v3 import build_recording_source
    s = build_recording_source(tiny_experiment, well="A1", fov=0, context=default_context())
    assert isinstance(s, FovSource)


def test_recording_dtype_is_numpy_not_tensorstore(tiny_experiment):
    from record_zstack_viewer.reader.zarr_v3 import build_recording_source
    s = build_recording_source(tiny_experiment, well="A1", fov=0, context=default_context())
    # FovSource protocol + downstream expect a numpy dtype, not ts.dtype
    assert isinstance(s.dtype, np.dtype) and s.dtype == np.uint16


# ---- Task 1.6: FormatAdapter — open_fov -------------------------------------

def test_open_fov_returns_both_phase_sources(tiny_experiment):
    # A FOV acquired in both phases → open_fov yields a working recording AND
    # z-stack source, each on its own uniform FovSource surface.
    from record_zstack_viewer.reader.adapter import open_fov
    ds = open_fov(tiny_experiment, well="A1", fov=0, context=default_context())
    assert isinstance(ds.recording, FovSource) and isinstance(ds.zstack, FovSource)
    assert ds.recording.dims == ("acq_t", "frame", "z", "y", "x")
    assert ds.zstack.dims == ("acq_t", "c", "z", "y", "x")
    # each source actually reads its visible plane (per-plane, never whole-array)
    assert ds.recording.read_plane({"acq_t": 0, "frame": 0, "z": 0}).shape == (64, 64)
    assert ds.zstack.read_plane({"acq_t": 0, "c": 0, "z": 0}).shape == (64, 64)


def test_open_fov_recording_only(tmp_path):
    # Recording-only FOV (z-stack tree removed): recording is a working source,
    # zstack is None.
    import shutil
    from record_zstack_viewer.reader.adapter import open_fov
    root = generate_experiment(tmp_path / "rec_only", wells=["A1"], fovs_per_well=1, nt=1,
        recording_frames=4, recording_planes=2,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=3, rich=False)
    shutil.rmtree(root / "zarr")  # drop the z-stack phase entirely
    ds = open_fov(root, well="A1", fov=0, context=default_context())
    assert isinstance(ds.recording, FovSource)
    assert ds.zstack is None
    assert ds.recording.read_plane({"acq_t": 0, "frame": 0, "z": 0}).shape == (64, 64)


def test_open_fov_zstack_only(tmp_path):
    # Z-stack-only FOV (recording_planes=0 → no recording stores): zstack is a
    # working source, recording is None.
    from record_zstack_viewer.reader.adapter import open_fov
    root = generate_experiment(tmp_path / "zs_only", wells=["A1"], fovs_per_well=1, nt=1,
        recording_frames=4, recording_planes=0,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=3, rich=False)
    ds = open_fov(root, well="A1", fov=0, context=default_context())
    assert ds.recording is None
    assert isinstance(ds.zstack, FovSource)
    assert ds.zstack.read_plane({"acq_t": 0, "c": 0, "z": 0}).shape == (64, 64)


def test_registry_seam_registers_and_detects_zarr_v3(tiny_experiment):
    # The seam is real: a Zarr-v3 adapter is registered by format and `detect`
    # recognizes the experiment layout (legacy adapters plug in the same way).
    from record_zstack_viewer.reader.adapter import ZarrV3Adapter, detect_adapter
    adapter = detect_adapter(tiny_experiment)
    assert isinstance(adapter, ZarrV3Adapter) and adapter.format == "ZARR_V3"
