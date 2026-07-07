"""Qt UI layer (Phase 5).

Submodules import Qt (via qtpy) and — in `fov_viewer` only — ndv. Import them
explicitly; `import record_zstack_viewer` alone must stay Qt-free (Global
Constraints: non-UI modules import no Qt, ndv lives behind `ui.fov_viewer`).
"""
