"""Task 1.5 — `filled_extent` live written-count (FAST chunk-presence; sharded fallback).

A live/partial store has real pixels only up to the acquisition frontier; the rest
of the declared shape reads back as `fill_value` (0). `filled_extent(axis)` reports
how far real data actually reaches so callers can bound sliders / status without
mistaking pending fill for data, and `is_extent_exact()` says whether that count is
trustworthy (FAST chunk-per-file → exact; sharded → degrade to declared size).

Both sources probe *on-disk chunk presence* (one `scandir` pass, never a stat per
frame) and must be robust to ragged stores — the recording FOV here has one complete
timepoint and one half-written one, and the z-stack's last timepoint has only z=0.
"""
import json

import numpy as np
import pytest
import tensorstore as ts

from tests.fixtures.generate import generate_experiment
from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.reader.zarr_v3 import ZStackFovSource, build_recording_source


@pytest.fixture
def partial_experiment(tmp_path):
    """Last FOV (t1/B2/fov0) half-written: recording 3 of 6 frames, z-stack only z=0.

    Ragged on purpose — B2/fov0 spans acq_t0 (complete, 6 frames) and acq_t1 (the
    growing frontier, 3 frames), so `filled_extent` must probe actual presence per
    store rather than assume every timepoint has equal filled counts.
    """
    return generate_experiment(
        tmp_path / "exp", wells=["A1", "B2"], fovs_per_well=1, nt=2,
        recording_frames=6, recording_planes=3,
        zstack_channels=[("DAPI", "#4a9eff"), ("GFP", "#34d058")], zstack_nz=4,
        complete=False)


def _recording(exp, well="B2"):
    return build_recording_source(exp, well=well, fov=0, context=default_context())


def _zstack(exp, well="B2"):
    return ZStackFovSource(exp / "zarr" / well / "fov_0.ome.zarr", default_context())


# ---- recording: frame axis is the live-growing one ---------------------------

def test_recording_filled_frame_is_written_count_not_declared(partial_experiment):
    s = _recording(partial_experiment)
    declared = s.shape[s.dims.index("frame")]
    assert declared == 6                                # full declared frame count
    # acq_t0 wrote 6 frames, acq_t1 (the frontier) only 3 → real data guaranteed to 3
    assert s.filled_extent("frame") == 3
    assert s.filled_extent("frame") < declared


def test_recording_filled_acq_t_counts_present_timepoint_dirs(partial_experiment):
    s = _recording(partial_experiment)
    # both t{t} dirs exist for this FOV (a timepoint dir appears as soon as it starts)
    assert s.filled_extent("acq_t") == s.shape[0] == 2


def test_read_plane_on_unwritten_frame_returns_fill_value(partial_experiment):
    s = _recording(partial_experiment)
    # frame 5 is within the declared shape but never written in the partial acq_t=1
    plane = s.read_plane({"acq_t": 1, "frame": 5, "z": 0})
    assert plane.shape == (64, 64) and plane.dtype == np.uint16
    assert np.array_equal(plane, np.zeros_like(plane))  # fill_value → pending, not an error
    # contrast: a written frame in the same store holds real (non-zero) pixels
    assert s.read_plane({"acq_t": 1, "frame": 0, "z": 0}).any()


# ---- z-stack: acq_t axis is the live-growing one -----------------------------

def test_zstack_partial_extent_detected(partial_experiment):
    s = _zstack(partial_experiment)
    declared = s.shape[s.dims.index("acq_t")]
    assert declared == 2
    # t0's full (c,z) chunk set is present; t1 has only z=0 → 1 complete timepoint
    assert s.filled_extent("acq_t") == 1
    assert s.filled_extent("acq_t") < declared


# ---- FAST store (no sharding) → the count is exact ---------------------------

def test_fast_store_extent_is_exact(partial_experiment):
    assert _recording(partial_experiment).is_extent_exact() is True
    assert _zstack(partial_experiment).is_extent_exact() is True


def test_complete_store_reports_declared_and_exact(tiny_experiment):
    r = _recording(tiny_experiment)                     # every frame present in every timepoint
    assert r.filled_extent("frame") == r.shape[r.dims.index("frame")] == 5
    assert r.is_extent_exact() is True
    z = _zstack(tiny_experiment)                         # both timepoints fully acquired
    assert z.filled_extent("acq_t") == z.shape[0] == 2


# ---- sharded store → presence != per-frame, degrade to declared + inexact ----

def _make_sharded_zstack(root):
    """A z-stack group whose `/0` uses the `sharding_indexed` codec (BALANCED/BEST).

    One shard file aggregates many inner chunks, so on-disk chunk presence can no
    longer be read per-frame; the source must fall back to the declared extent.
    """
    grp = root / "zarr" / "A1" / "fov_0.ome.zarr"
    arr = grp / "0"
    a = ts.open({
        "driver": "zarr3", "kvstore": {"driver": "file", "path": str(arr)},
        "metadata": {
            "shape": [3, 1, 1, 8, 8],
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3, 1, 1, 8, 8]}},
            "data_type": "uint16",
            "codecs": [{"name": "sharding_indexed", "configuration": {
                "chunk_shape": [1, 1, 1, 8, 8],
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                 {"name": "crc32c"}]}}],
            "fill_value": 0},
        "create": True, "delete_existing": True}).result()
    a[0, 0, 0].write(np.ones((8, 8), "uint16")).result()  # only 1 of 3 timepoints written
    grp.mkdir(parents=True, exist_ok=True)
    (grp / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "group",
        "attributes": {"_squid": {"compression": "BALANCED"}}}))
    return grp


def test_sharded_store_degrades_to_declared_and_inexact(tmp_path):
    s = ZStackFovSource(_make_sharded_zstack(tmp_path), default_context())
    assert s.is_extent_exact() is False
    # cannot probe per-frame presence under sharding → report the declared size
    assert s.filled_extent("acq_t") == s.shape[0] == 3
