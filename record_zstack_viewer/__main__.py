"""CLI entry: ``python -m record_zstack_viewer [path]``.

Opens the viewer (`MainWindow`, Layout A) on an experiment root — default the
current directory — and follows a growing acquisition live via the window's
UI-thread QTimer (~1 s `LiveEngine.tick()`; never `LiveEngine.start()`).

Qt (and, transitively, ndv) load only inside `main()` after argument parsing,
so ``--help``/bad-path errors stay instant and importing this module stays
UI-free. Headless / fully GL-less hosts: run with ``LIBGL_ALWAYS_SOFTWARE=1``
(Mesa llvmpipe) per the CPU/headless note.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="record_zstack_viewer",
        description="Viewer for squid record + z-stack well-plate experiments "
                    "(OME-Zarr v3), with live-acquisition following.")
    parser.add_argument(
        "path", nargs="?", default=".",
        help="experiment root (default: the current directory)")
    args = parser.parse_args(argv)
    root = Path(args.path).expanduser()
    if not root.is_dir():
        parser.error(f"experiment root {root} is not a directory")

    from qtpy.QtWidgets import QApplication          # Qt only past this point

    from record_zstack_viewer.ui.app import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
