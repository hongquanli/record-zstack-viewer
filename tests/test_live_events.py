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


def test_tailer_reconstitutes_multibyte_char_split_across_polls(tmp_path):
    """A multi-byte UTF-8 char split across a read boundary must be buffered as raw
    bytes, never decoded half-way: the poll that sees only the first byte returns []
    (no newline yet), and the next poll — after the rest of the char and the newline
    arrive — reconstitutes and decodes the full line."""
    mu = "µ"                                        # 'µ' — UTF-8 bytes 0xC2 0xB5
    p = tmp_path / "events.jsonl"
    t = EventLogTailer(p)
    p.write_bytes(b'{"type":"note","msg":"' + b"\xc2")   # line up to only the 1st byte of µ
    assert t.poll() == []                                # split mid-char, no newline yet
    with p.open("ab") as f:
        f.write(b"\xb5" + b'"}\n')                        # 2nd byte of µ + close quote + newline
    assert t.poll() == [{"type": "note", "msg": mu}]      # reconstituted and decoded


def test_tailer_skips_invalid_utf8_line(tmp_path):
    """A complete line of raw invalid-UTF-8 / non-JSON bytes is skipped without
    raising and without stalling the valid line that follows it."""
    p = tmp_path / "events.jsonl"
    p.write_bytes(b"\xff\xff not valid utf-8 or json\n"   # bad bytes, complete line
                  b'{"type":"run_started"}\n')            # valid line immediately after
    t = EventLogTailer(p)
    assert [e["type"] for e in t.poll()] == ["run_started"]
