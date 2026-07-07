"""CLI entry: ``python -m record_zstack_viewer [path]``.

With no path, opens the drag-and-drop launcher (`LauncherWindow`); with an
experiment-root path, opens that experiment directly (`MainWindow`, Layout A).
Either follows a growing acquisition live via each window's UI-thread QTimer
(~1 s `LiveEngine.tick()`; never `LiveEngine.start()`). All windows share one
`ViewerController` (one bounded `cache_pool`).

Qt (and, transitively, ndv) load only inside `main()` after argument parsing,
so ``--help``/bad-path errors stay instant and importing this module stays
UI-free. Headless / fully GL-less hosts: run with ``LIBGL_ALWAYS_SOFTWARE=1``
(Mesa llvmpipe) per the CPU/headless note.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build(argv: list[str] | None = None):
    """Parse args and build the app; return ``(app, controller, launcher_or_None)``.

    Split from :func:`main` (which adds ``app.exec()``) so the wiring is testable
    without entering the Qt event loop. No path -> the drag-and-drop launcher;
    a valid experiment path -> that experiment opens directly (no launcher);
    an invalid path -> ``parser.error`` (exit 2).
    """
    parser = argparse.ArgumentParser(
        prog="record_zstack_viewer",
        description="Viewer for squid record + z-stack well-plate experiments "
                    "(OME-Zarr v3), with live-acquisition following.")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="experiment root; omit to open the drag-and-drop launcher")
    args = parser.parse_args(argv)

    from qtpy.QtWidgets import QApplication          # Qt only past this point

    from record_zstack_viewer.ui.controller import ViewerController, is_experiment
    from record_zstack_viewer.ui.launcher import LauncherWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    controller = ViewerController()
    launcher = None
    if args.path is None:
        launcher = LauncherWindow(controller)
        controller.launcher = launcher
        launcher.show()
    else:
        root = Path(args.path).expanduser()
        if not is_experiment(root):
            parser.error(f"{root} is not a record+zstack experiment "
                         "(no recording/ or zarr/ folder)")
        controller.open_experiment(root)
    return app, controller, launcher


def main(argv: list[str] | None = None) -> int:
    app, _controller, _launcher = _build(argv)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
