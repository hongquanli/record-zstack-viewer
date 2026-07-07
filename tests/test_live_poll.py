"""Task 4.2b — `LiveEngine`: events-primary / poll-fallback change detection.

`LiveEngine.tick()` is the unit of work (a Qt timer or a test drives it). It

  - reads `events.jsonl` via `EventLogTailer` when that log exists (**primary**),
    mapping new events to a changed `{(well, fov)}` set and a status dict;
  - else computes a cheap per-store on-disk signature (dir/chunk/zarr.json mtimes
    + `_squid.acquisition_complete`) and diffs it against the previous snapshot
    (**fallback**);

and then, in this ORDER: `model.refresh()` first (drops the handle cache so a
re-open sees fresh pixels), `thumbnail_cache.invalidate(well, fov)` for each
changed FOV, then `on_change(changed)` and `on_status(status)`.

The z-stack visibility test locks the *consumer contract* the 4.2a review
flagged: a held `ZStackFovSource` binds its array handle at construction and
stays stale across `refresh()`, so a consumer MUST re-open the FOV on `on_change`
to see grown data. No Qt/ndv imports here — pure filesystem + callbacks.
"""
import json
import threading
from pathlib import Path

import numpy as np
import tensorstore as ts

from record_zstack_viewer.cache.memory import default_context
from record_zstack_viewer.live.engine import LiveEngine
from record_zstack_viewer.model.experiment import ExperimentModel
from tests.fixtures.generate import generate_experiment


# --- spies / stubs -----------------------------------------------------------

class _SpyModel:
    """A stand-in for `ExperimentModel` in the events-path unit tests.

    The events path only ever touches `model.root` (to find `events.jsonl`) and
    `model.refresh()`, so a full model isn't needed — this records refresh calls
    and their order relative to the callbacks.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.refresh_calls = 0

    def refresh(self):
        self.refresh_calls += 1


class _SpyThumbs:
    def __init__(self):
        self.invalidated = []

    def invalidate(self, well, fov):
        self.invalidated.append((well, fov))


def _append(log: Path, *events: dict) -> None:
    with log.open("a") as fh:
        for ev in events:
            fh.write(json.dumps({**ev, "ts": 0.0, "format_version": "1.0"}) + "\n")


# --- events path (primary) ---------------------------------------------------

def test_events_path_changed_status_refresh_invalidate(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    log = root / "events.jsonl"
    log.write_text("")                                   # exists -> events path chosen
    model = _SpyModel(root)
    thumbs = _SpyThumbs()

    changes, statuses = [], []

    def on_change(s):
        # record the refresh count AT FIRE TIME to lock the refresh-before-on_change order
        changes.append((set(s), model.refresh_calls))

    eng = LiveEngine(model, on_change=on_change, on_status=statuses.append,
                     thumbnail_cache=thumbs)

    _append(log,
            {"type": "run_started", "expected_fovs_per_well": {"A1": 1}, "wells": ["A1"]},
            {"type": "fov_started", "region": "A1", "fov": 0, "phase": "recording"},
            {"type": "heartbeat", "region": "A1", "fov": 0, "phase": "recording", "written": 3},
            {"type": "store_finalized", "region": "A1", "fov": 0, "phase": "recording",
             "acquisition_complete": True, "aborted": False},
            {"type": "store_finalized", "region": "A1", "fov": 0, "phase": "zstack",
             "acquisition_complete": True, "aborted": False},
            {"type": "run_done"})
    eng.tick()

    # on_change carries the right (well, fov), and refresh() ran BEFORE it fired
    assert len(changes) == 1
    changed_set, refresh_at_fire = changes[0]
    assert changed_set == {("A1", 0)}
    assert refresh_at_fire == 1                          # refresh() precedes on_change
    assert model.refresh_calls == 1

    # thumbnail cache invalidated for each changed FOV
    assert thumbs.invalidated == [("A1", 0)]

    # status reflects current / coverage / done
    st = statuses[-1]
    assert (st["current_well"], st["current_fov"]) == ("A1", 0)
    assert st["coverage_done"] == 1 and st["coverage_total"] == 1
    assert st["done"] is True and st["acquiring"] is False and st["aborted"] is False


def test_events_path_no_new_events_is_noop(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    log = root / "events.jsonl"
    log.write_text("")
    model = _SpyModel(root)
    changes, statuses = [], []
    eng = LiveEngine(model, on_change=changes.append, on_status=statuses.append)

    _append(log, {"type": "fov_started", "region": "A1", "fov": 0, "phase": "recording"})
    eng.tick()
    assert changes == [{("A1", 0)}] and model.refresh_calls == 1

    eng.tick()                                           # nothing appended since
    assert changes == [{("A1", 0)}]                      # no extra on_change
    assert model.refresh_calls == 1                      # no needless refresh
    assert len(statuses) == 1                            # no extra on_status


def test_events_path_abort_sets_status(tmp_path):
    root = tmp_path / "exp"
    root.mkdir()
    log = root / "events.jsonl"
    log.write_text("")
    model = _SpyModel(root)
    statuses = []
    eng = LiveEngine(model, on_change=lambda s: None, on_status=statuses.append)

    _append(log,
            {"type": "run_started", "expected_fovs_per_well": {"A1": 2}},
            {"type": "fov_started", "region": "A1", "fov": 0, "phase": "recording"},
            {"type": "aborted", "reason": "user"})
    eng.tick()
    st = statuses[-1]
    assert st["aborted"] is True and st["acquiring"] is False and st["done"] is False


# --- poll fallback -----------------------------------------------------------

def _grow_recording(root: Path, well: str, fov: int, *, first: int, last: int,
                    value: int) -> None:
    """Write frames [first, last) into (well, fov)'s single-plane recording store at
    t0 and bump its `_squid`, as a live acquisition grows the store on disk."""
    store = root / "recording" / "t0" / well / f"fov_{fov}.ome.zarr"
    arr = ts.open({"driver": "zarr3",
                   "kvstore": {"driver": "file", "path": str(store)}}).result()
    h, w = arr.shape[-2:]
    for f in range(first, last):
        arr[f, 0, 0].write(np.full((h, w), value, dtype="uint16")).result()
    zj = store / "zarr.json"
    meta = json.loads(zj.read_text())
    meta["attributes"]["_squid"]["captured_frames"] = last
    meta["attributes"]["_squid"]["acquisition_complete"] = True
    zj.write_text(json.dumps(meta, indent=2))


def _partial_no_events(tmp_path):
    """A partial experiment with NO events.jsonl (rich=False) -> forces the poll path.

    The last FOV (B2/fov0 at t0) is left half-written; nt=1 keeps it at t0 so the
    `_grow_recording` helper (which targets t0) grows the right store.
    """
    root = generate_experiment(
        tmp_path / "exp", wells=["A1", "B2"], fovs_per_well=1, nt=1,
        recording_frames=6, recording_planes=1,
        zstack_channels=[("DAPI", "#4a9eff")], zstack_nz=4,
        rich=False, complete=False)
    assert not (root / "events.jsonl").exists()          # precondition: poll path
    return root


def test_poll_fallback_detects_grown_store(tmp_path):
    root = _partial_no_events(tmp_path)
    model = ExperimentModel(root, default_context())
    changes, statuses = [], []
    eng = LiveEngine(model, on_change=changes.append, on_status=statuses.append)

    eng.tick()                                           # baseline snapshot
    assert changes == []                                 # baseline reports nothing changed

    _grow_recording(root, "B2", 0, first=3, last=6, value=1234)
    eng.tick()

    assert len(changes) == 1
    assert ("B2", 0) in changes[0]                       # the grown FOV is reported
    assert ("A1", 0) not in changes[0]                   # an untouched FOV is not


def test_poll_fallback_idle_tick_is_noop(tmp_path):
    root = _partial_no_events(tmp_path)
    model = ExperimentModel(root, default_context())
    changes = []
    eng = LiveEngine(model, on_change=changes.append, on_status=lambda s: None)
    eng.tick()                                           # baseline
    eng.tick()                                           # nothing changed on disk
    assert changes == []


# --- live z-stack visibility: the 4.2a coverage gap --------------------------

def _grow_zstack(root: Path, well: str, fov: int, *, value: int) -> None:
    """Fill in every pending z of (well, fov)'s z-stack at t0 and finalize it, as a
    live z-stack acquisition completes its planes on disk."""
    arr_dir = root / "zarr" / well / f"fov_{fov}.ome.zarr" / "0"
    a = ts.open({"driver": "zarr3",
                 "kvstore": {"driver": "file", "path": str(arr_dir)}}).result()
    _nt, nc, nz, h, w = a.shape
    for c in range(nc):
        for z in range(1, nz):                           # z=0 was already written (partial)
            a[0, c, z].write(np.full((h, w), value, dtype="uint16")).result()
    grp = root / "zarr" / well / f"fov_{fov}.ome.zarr"
    zj = grp / "zarr.json"
    meta = json.loads(zj.read_text())
    meta["attributes"]["_squid"]["acquisition_complete"] = True
    zj.write_text(json.dumps(meta, indent=2))


def test_live_zstack_visible_only_after_reopen(tmp_path):
    """The consumer contract: a HELD z-stack source stays stale across a live tick;
    RE-OPENING the FOV after the engine's refresh() surfaces the grown pixels.

    Locks the 4.2a gap for z-stack — `ZStackFovSource` caches its array handle at
    construction, so only a re-open (not the held source) reflects the growth.
    """
    root = _partial_no_events(tmp_path)
    model = ExperimentModel(root, default_context())
    well, fov = "B2", 0

    held = model.open_fov(well, fov).zstack               # a displayed FOV's held source
    pending = {"acq_t": 0, "c": 0, "z": held.shape[2] - 1}   # last z: unwritten in the partial
    assert int(held.read_plane(pending).max()) == 0       # pending -> fill 0 (stale baseline)

    changes = []
    eng = LiveEngine(model, on_change=changes.append, on_status=lambda s: None)
    eng.tick()                                            # baseline snapshot (pre-grow)

    _grow_zstack(root, well, fov, value=4242)
    eng.tick()                                            # detects change -> model.refresh()

    assert changes and (well, fov) in changes[-1]         # on_change flags the FOV

    # held source is still stale (binds its handle at construction) ...
    assert int(held.read_plane(pending).max()) == 0
    # ... but RE-OPENING after refresh() sees the real grown data
    reopened = model.open_fov(well, fov).zstack
    assert int(reopened.read_plane(pending).max()) == 4242


# --- start()/stop() lifecycle (no Qt) ----------------------------------------

def test_start_stop_drives_tick_without_qt(tmp_path):
    """`start()` schedules periodic `tick()` off a daemon thread; `stop()` halts it.

    Uses a real thread (the only way to exercise the driver) with a short interval
    and a generous timeout — it asserts the driver fires at least once and stops
    cleanly, not exact timing. The behavioral assertions live in the tick() tests.
    """
    root = tmp_path / "exp"
    root.mkdir()
    (root / "events.jsonl").write_text("")
    model = _SpyModel(root)
    _append(root / "events.jsonl",
            {"type": "fov_started", "region": "A1", "fov": 0, "phase": "recording"})

    fired = threading.Event()
    eng = LiveEngine(model, on_change=lambda s: fired.set(), on_status=lambda s: None,
                     interval=0.01)
    eng.start()
    try:
        assert fired.wait(timeout=3.0), "start() never drove a tick()"
    finally:
        eng.stop()
    assert eng.stop() is None                              # idempotent, safe to re-stop
