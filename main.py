#!/usr/bin/env python3
import sys
import argparse

if sys.version_info < (3, 10):
    print(f"Error: Python 3.10 or newer is required (you have {sys.version.split()[0]})")
    sys.exit(1)

import logging

from PyQt6.QtWidgets import QApplication
from gui.scope_window import ScopeWindow


def main():
    parser = argparse.ArgumentParser(description="Hanscope — Hantek 1008C oscilloscope GUI")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="enable debug logging (-v: INFO, -vv: DEBUG)")
    args, qt_args = parser.parse_known_args()

    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("usb").setLevel(logging.WARNING)

    app = QApplication([sys.argv[0], *qt_args])
    window = ScopeWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
