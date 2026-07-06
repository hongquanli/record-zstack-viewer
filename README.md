# record-zstack-viewer

Standalone, memory-frugal desktop viewer for Squid **Record + Z-Stack** well-plate scans
(OME-Zarr v3 from squid PR #564/#579): plate navigation, per-FOV recording playback and
z-stack viewing, and live updates during an ongoing acquisition.

**Status:** design complete; implementation not started (planned for Opus).

## Design docs
- Design spec: [`docs/specs/2026-07-05-record-zstack-viewer-design.md`](docs/specs/2026-07-05-record-zstack-viewer-design.md)
- Implementation plan (contract-first, TDD): [`docs/plans/2026-07-06-record-zstack-viewer.md`](docs/plans/2026-07-06-record-zstack-viewer.md)

## Approach (summary)
- **Stack:** Python 3.11 · TensorStore (Zarr v3 reads) · ndv/vispy (GPU render, single path) · qtpy/PyQt5 + pyqtgraph (UI).
- **Memory:** O(1) in dataset size — per-plane / per-tile reads bounded by a shared TensorStore `cache_pool`; no whole-array loads.
- **Live:** tails an `events.jsonl` when present, else file-watch + poll; reads partially-written stores safely.
- **Contract-first:** a versioned `CONTRACT.md` plus a fixture generator and validator come first (Phase 0); the viewer is built and tested entirely against fixtures, decoupling it from the squid-side additions.

## Related (acquisition-side, squid repo)
Three additive sidecars the viewer will adopt are tracked in `AI-docs/Squid/to-do/`:
per-FOV `coordinates.csv`, `events.jsonl` live status, and per-FOV thumbnails. The viewer
works without them (fallbacks) and gets cheaper/mm-exact when they land.
