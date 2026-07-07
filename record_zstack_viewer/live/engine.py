"""The primary live channel: an offset-based tailer of squid's `events.jsonl`.

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
from pathlib import Path


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
