"""The live channel: an `events.jsonl` tailer plus the `LiveEngine` that drives
a growing acquisition's UI updates.

`EventLogTailer` is the primary source — an offset-based tail of squid's
append-only `events.jsonl`. `LiveEngine` sits on top: each `tick()` derives the
changed `{(well, fov)}` set and a status dict (events if the log exists, else a
cheap on-disk poll), then refreshes the model, invalidates thumbnails, and fires
the consumer callbacks. `LiveEngine` imports **no Qt** — production wires
`tick()` to a Qt timer on the UI thread; tests drive `tick()` directly.

`events.jsonl` is an append-only JSON-lines log at the experiment root (one JSON
object per line — see CONTRACT.md). `EventLogTailer` reads it incrementally: it
remembers a byte offset and, on each `poll()`, reads only the bytes appended
since the previous call, so cost is O(new bytes) rather than a full re-read.

Two things make it robust against a log being written *concurrently* by another
process:

- **Complete lines only.** A line is emitted only once its terminating ``\n``
  has been written. Bytes after the last newline are held in a pending buffer
  (kept as raw bytes, so a multi-byte UTF-8 character split across a read
  boundary is never decoded half-way) and re-examined on the next poll.
- **Tolerant of a shrinking/absent file.** If the file does not exist yet the
  tailer simply reports no events until it appears; if it is truncated or
  rewritten shorter than the tracked offset, the offset is reset so tailing
  restarts from the new beginning instead of reading past EOF.

Malformed-but-complete lines are skipped rather than raised, so one bad line
never stops the tail. No Qt import — this is pure filesystem tailing.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Callable

from record_zstack_viewer.reader.metadata import read_store_zarr_json

logger = logging.getLogger(__name__)


class EventLogTailer:
    """Incrementally tail an append-only `events.jsonl`, one poll at a time.

    Construct with the log path (which need not exist yet). Call `poll()`
    repeatedly; each call returns the list of event dicts appended since the
    previous call. `has_log()` reports whether the file currently exists.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._offset = 0           # bytes of the file already consumed
        self._pending = b""        # bytes read past the last newline, not yet a line

    def has_log(self) -> bool:
        """Whether the log file currently exists on disk."""
        return self._path.is_file()

    def poll(self) -> list[dict]:
        """Return event dicts appended since the previous `poll()`.

        Only complete ``\n``-terminated lines are parsed; a half-written
        trailing line is buffered until its newline arrives. Malformed complete
        lines are skipped. Returns ``[]`` when the file is missing, unreadable,
        or has grown no new complete line since the last call.
        """
        try:
            size = self._path.stat().st_size
        except OSError:
            return []              # file missing or inaccessible — nothing to tail

        if size < self._offset:
            # truncated or rewritten shorter than what we've read: restart cleanly
            self._offset = 0
            self._pending = b""
        if size == self._offset:
            return []              # no new bytes; any pending partial line stays buffered

        try:
            with self._path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read()
        except OSError:
            return []
        self._offset += len(chunk)

        data = self._pending + chunk
        parts = data.split(b"\n")
        self._pending = parts[-1]  # trailing element has no newline yet (may be b"")

        events: list[dict] = []
        for raw in parts[:-1]:
            line = raw.strip()
            if not line:
                continue           # skip blank lines
            try:
                obj = json.loads(line)
            except ValueError:
                continue           # malformed JSON (incl. bad UTF-8) — skip, don't crash
            if isinstance(obj, dict):
                events.append(obj)
        return events


# --- LiveEngine (Task 4.2b) ---------------------------------------------------

# `fov_{k}.ome.zarr` or `fov_{k}_z{j}.ome.zarr` — the store dir name for a FOV.
_FOV_STORE_RE = re.compile(r"^fov_(\d+)(?:_z\d+)?\.ome\.zarr$")

# events that carry a `region`/`fov` and imply that FOV's store appeared/grew/finalized
_FOV_EVENTS = frozenset({"fov_started", "heartbeat", "store_finalized"})


class LiveEngine:
    """Drives UI updates for a growing acquisition: one `tick()` per interval.

    Each `tick()` detects what changed since the last one — the `events.jsonl`
    tailer when that log exists (**primary**), else a cheap per-store on-disk
    signature diff (**fallback**) — and produces a changed `{(well, fov)}` set
    plus a status dict. It then, in this exact ORDER:

      1. calls ``model.refresh()`` **first** (drops the model's store-handle
         cache so a re-opened FOV re-reads grown pixels rather than stale fill);
      2. calls ``thumbnail_cache.invalidate(well, fov)`` for each changed FOV
         (when a cache was provided);
      3. fires ``on_change(changed_set)`` (only when non-empty) and
         ``on_status(status)``.

    The refresh-before-`on_change` order matters: a consumer re-opening the FOV
    inside `on_change` must see fresh data.

    **Consumer contract.** On `on_change` for a *displayed* FOV, the consumer MUST
    **re-open** it (``model.open_fov`` → viewer) rather than reuse a held source.
    A held `ZStackFovSource` binds its array handle at construction and stays
    stale even after `refresh()`; only a re-open (built from the just-cleared
    handle cache, which `refresh()` already ran) surfaces the grown pixels.
    Non-displayed FOVs need no action beyond the thumbnail invalidation done here.

    **No Qt import.** Production wires `tick()` to a Qt timer on the UI thread
    (``QTimer.timeout`` → ``engine.tick()``); tests call `tick()` directly.

    **Threading invariant.** `HandleCache` is thread-safe, but
    `ExperimentModel.refresh()` is NOT — so `refresh()` and `on_change` must run
    single-threaded with respect to model access. `tick()` is therefore designed
    to be called on ONE thread. The recommended production wiring (Qt timer on the
    UI thread) keeps everything on that thread. The convenience `start()` driver
    runs `tick()` on a background daemon thread and thus **assumes single-consumer
    model access**: the consumer must NOT read the model (`open_fov`, `wells`,
    `fovs`, or a held source) from another thread while a background tick may be
    running. If you cannot guarantee that, do not use `start()` — drive `tick()`
    from your UI thread's timer instead.
    """

    def __init__(self, model, on_change: Callable[[set[tuple[str, int]]], None],
                 on_status: Callable[[dict], None], thumbnail_cache=None,
                 interval: float = 1.0):
        self._model = model
        self._on_change = on_change
        self._on_status = on_status
        self._thumbnail_cache = thumbnail_cache
        self._interval = float(interval)
        self._tailer = EventLogTailer(Path(model.root) / "events.jsonl")

        # accumulated events-path status (advanced across ticks)
        self._status: dict = {
            "current_well": None, "current_fov": None,
            "coverage_done": 0, "coverage_total": None,
            "acquiring": False, "aborted": False, "done": False,
        }
        self._finalized: set[tuple[str, int]] = set()   # distinct completed FOVs (coverage k)

        # poll-path state: previous per-FOV signature snapshot (None until baseline)
        self._poll_snapshot: dict[tuple[str, int], tuple] | None = None

        # start()/stop() lifecycle (background daemon driver)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lifecycle_lock = threading.Lock()

    # --- the unit of work ----------------------------------------------------

    def tick(self) -> None:
        """Detect change, refresh the model, invalidate thumbnails, fire callbacks.

        Uses the `events.jsonl` tailer when the log exists, else the on-disk poll.
        A tick with nothing new is a no-op (no refresh, no callbacks). See the
        class docstring for the strict refresh → invalidate → on_change/on_status
        order and the threading invariant (call this on a single thread).
        """
        if self._tailer.has_log():
            events = self._tailer.poll()
            if not events:
                return                                   # idle: nothing to surface
            changed = self._apply_events(events)
            status = dict(self._status)
            active = True
        else:
            changed, status, active = self._poll_tick()

        if not active:
            return

        self._model.refresh()                            # FIRST — before any on_change re-open
        if self._thumbnail_cache is not None:
            for well, fov in changed:
                self._thumbnail_cache.invalidate(well, fov)
        if changed:
            self._on_change(changed)
        self._on_status(status)

    # --- events path (primary) -----------------------------------------------

    def _apply_events(self, events: list[dict]) -> set[tuple[str, int]]:
        """Fold new events into `self._status`; return the changed `{(well, fov)}`.

        `fov_started`/`heartbeat`/`store_finalized` each mark their `(region, fov)`
        changed (its store appeared / grew / finalized). `run_started` seeds the
        expected coverage total; a completed `store_finalized` advances the done
        count; `aborted`/`run_done` set the terminal status flags.
        """
        changed: set[tuple[str, int]] = set()
        for ev in events:
            typ = ev.get("type")
            wf = _well_fov(ev)
            if typ == "run_started":
                total = _expected_total(ev.get("expected_fovs_per_well"))
                self._status["coverage_total"] = total
                self._status.update(acquiring=True, done=False, aborted=False)
            elif typ in ("fov_started", "heartbeat"):
                if wf is not None:
                    changed.add(wf)
                    self._status["current_well"], self._status["current_fov"] = wf
                    self._status["acquiring"] = True
            elif typ == "store_finalized":
                if wf is not None:
                    changed.add(wf)
                    if ev.get("aborted"):
                        self._status["aborted"] = True
                    if ev.get("acquisition_complete"):
                        self._finalized.add(wf)
                self._status["coverage_done"] = len(self._finalized)
            elif typ == "aborted":
                self._status.update(aborted=True, acquiring=False)
            elif typ == "run_done":
                self._status.update(done=True, acquiring=False)
            # timepoint_done / unknown types: no per-FOV change, no status flip
        return changed

    # --- poll fallback --------------------------------------------------------

    def _poll_tick(self) -> tuple[set[tuple[str, int]], dict, bool]:
        """Diff a fresh on-disk signature snapshot against the previous one.

        The first call only seeds the baseline (returns `active=False`, so no
        refresh/callbacks — the model is already current from construction).
        Later calls return the FOVs whose signature changed (grown/finalized/new).
        """
        snapshot = self._scan_signatures()
        prev, self._poll_snapshot = self._poll_snapshot, snapshot
        if prev is None:
            return set(), {}, False                      # baseline only
        changed = {k for k, sig in snapshot.items() if sig != prev.get(k)}
        if not changed:
            return set(), {}, False
        return changed, self._poll_status(snapshot, changed), True

    def _scan_signatures(self) -> dict[tuple[str, int], tuple]:
        """`{(well, fov): signature}` over the `zarr/` and `recording/` trees.

        Each FOV's signature aggregates its stores' per-store signatures (see
        `_store_sig`), so growth or finalization of ANY backing store changes it.
        Walking both trees directly (not the model) lets a brand-new FOV be
        detected on the first tick after it appears.
        """
        root = Path(self._model.root)
        by_fov: dict[tuple[str, int], list[tuple]] = {}

        def add(well: str, name: str, store: Path, chunk_dir: Path) -> None:
            m = _FOV_STORE_RE.match(name)
            if m and store.is_dir():
                by_fov.setdefault((well, int(m.group(1))), []).append(
                    _store_sig(store, chunk_dir))

        zroot = root / "zarr"
        if zroot.is_dir():
            for well_dir in zroot.iterdir():
                if well_dir.is_dir():
                    for store in well_dir.iterdir():
                        add(well_dir.name, store.name, store, store / "0" / "c")
        rec = root / "recording"
        if rec.is_dir():
            for t_dir in rec.iterdir():
                if not (t_dir.is_dir() and t_dir.name.startswith("t")):
                    continue
                for well_dir in t_dir.iterdir():
                    if well_dir.is_dir():
                        for store in well_dir.iterdir():
                            add(well_dir.name, store.name, store, store / "c")
        return {k: tuple(sorted(v)) for k, v in by_fov.items()}

    def _poll_status(self, snapshot: dict[tuple[str, int], tuple],
                     changed: set[tuple[str, int]]) -> dict:
        """Coarse status from the on-disk snapshot (no events to read a 'current'
        FOV from, so the most-recently-changed FOV stands in)."""
        total = len(snapshot)
        # a FOV is 'done' when every one of its stores reports acquisition_complete
        done = sum(1 for sigs in snapshot.values() if sigs and all(s[-1] for s in sigs))
        run_done = (Path(self._model.root) / ".done").exists()
        current = max(changed) if changed else (None, None)
        return {
            "current_well": current[0], "current_fov": current[1],
            "coverage_done": done, "coverage_total": total,
            "acquiring": (not run_done) and done < total,
            "aborted": False, "done": run_done,
        }

    # --- start()/stop(): background periodic driver (see threading invariant) --

    def start(self) -> None:
        """Schedule periodic `tick()` on a background daemon thread.

        A convenience driver for non-Qt use. It runs `tick()` — and therefore
        `model.refresh()` — off the UI thread, so it **assumes single-consumer
        model access** (see the class docstring's threading invariant): the
        consumer must not touch the model from another thread while a tick may be
        running. In production, prefer wiring `tick()` to a Qt timer on the UI
        thread instead of `start()`. Idempotent: a second call while running is a
        no-op.
        """
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="LiveEngine",
                                            daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Cancel the periodic driver and wait for the in-flight tick to finish.

        Safe to call when not started and safe to call repeatedly (idempotent).
        """
        self._stop_event.set()
        with self._lifecycle_lock:
            thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._interval + 5.0)

    def _loop(self) -> None:
        """Tick every `interval` until stopped. `Event.wait` returns True when
        `stop()` sets it (exit) and False on timeout (tick). A tick error is logged
        but must not kill the loop — a transient FS hiccup shouldn't stop live
        following."""
        while not self._stop_event.wait(self._interval):
            try:
                self.tick()
            except Exception:                            # noqa: BLE001 — keep the driver alive
                logger.exception("LiveEngine.tick() failed; continuing")


# --- module helpers ----------------------------------------------------------

def _well_fov(ev: dict) -> tuple[str, int] | None:
    """`(region, fov)` from an event, or None when either is absent/malformed."""
    region, fov = ev.get("region"), ev.get("fov")
    if region is None or fov is None:
        return None
    try:
        return (str(region), int(fov))
    except (TypeError, ValueError):
        return None


def _expected_total(expected: object) -> int | None:
    """Sum of `expected_fovs_per_well` values (the coverage N), or None if empty/bad."""
    if not isinstance(expected, dict):
        return None
    try:
        total = sum(int(v) for v in expected.values())
    except (TypeError, ValueError):
        return None
    return total or None


def _store_sig(store: Path, chunk_dir: Path) -> tuple:
    """A cheap change signature for one store: mtimes of the store dir, its chunk
    container, and `zarr.json`, plus `_squid.acquisition_complete` / `captured_frames`.

    The chunk-dir mtime catches live growth (a new chunk key), the `zarr.json`
    mtime and the `_squid` flags catch finalization / a rewritten frame count. All
    fields are plain ints/bools (missing → 0 / False / -1) so signatures compare
    and sort without None-vs-int errors. Ends with `acquisition_complete` so
    `_poll_status` can read completeness off `sig[-1]`.
    """
    def mtime(p: Path) -> int:
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return 0

    try:
        squid = (read_store_zarr_json(store).get("attributes") or {}).get("_squid") or {}
    except (OSError, ValueError):
        squid = {}
    captured = squid.get("captured_frames")
    return (
        str(store),
        mtime(store), mtime(chunk_dir), mtime(store / "zarr.json"),
        int(captured) if captured is not None else -1,
        bool(squid.get("acquisition_complete", False)),
    )
