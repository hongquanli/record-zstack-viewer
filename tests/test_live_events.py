"""Task 4.1 — `EventLogTailer`: the primary live channel over `events.jsonl`.

The tailer follows an append-only JSON-lines log written by squid. `poll()`
returns only the events appended since the previous call (tracked by a byte
offset, so no full re-read), and only ever yields **complete** `\n`-terminated
lines: a partially-written trailing line is buffered until its newline arrives.
It tolerates the log not existing yet at construction and the file being
truncated/rewritten, and it skips malformed-but-complete lines without crashing.
No Qt import — this is pure file tailing.
"""
from record_zstack_viewer.live.engine import EventLogTailer


def test_tailer_reads_incrementally(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"type":"run_started"}\n')
    t = EventLogTailer(p)
    assert [e["type"] for e in t.poll()] == ["run_started"]
    assert t.poll() == []                                    # nothing new appended
    with p.open("a") as f:
        f.write('{"type":"fov_started","region":"A1"}\n')
    assert [e["type"] for e in t.poll()] == ["fov_started"]  # only the new one
    with p.open("a") as f:
        f.write('{"type":"partial')                          # half line, no newline
    assert t.poll() == []                                    # ignored until complete
    with p.open("a") as f:
        f.write('"}\n')                                       # completes the line
    assert [e["type"] for e in t.poll()] == ["partial"]      # now returned


def test_tailer_tolerates_missing_then_appearing(tmp_path):
    p = tmp_path / "events.jsonl"
    t = EventLogTailer(p)                                    # file does not exist yet
    assert t.has_log() is False
    assert t.poll() == []
    p.write_text('{"type":"run_started"}\n')                # log appears later
    assert t.has_log() is True
    assert [e["type"] for e in t.poll()] == ["run_started"]


def test_tailer_skips_malformed_lines(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text(
        '{"type":"run_started"}\n'
        "not json at all\n"
        "\n"                                                 # blank line
        '{"type":"timepoint_done","t":0}\n'
    )
    t = EventLogTailer(p)
    # malformed and blank complete lines are skipped; valid events still returned
    assert [e["type"] for e in t.poll()] == ["run_started", "timepoint_done"]


def test_tailer_handles_truncation(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"type":"run_started"}\n{"type":"fov_started","region":"A1"}\n')
    t = EventLogTailer(p)
    assert len(t.poll()) == 2
    # file rewritten from scratch — new size is shorter than the tracked offset
    p.write_text('{"type":"run_done"}\n')
    assert [e["type"] for e in t.poll()] == ["run_done"]
