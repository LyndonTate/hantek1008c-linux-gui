#!/usr/bin/env python3
import sys

if sys.version_info < (3, 10):
    print(f"Error: Python 3.10 or newer is required (you have {sys.version.split()[0]})")
    sys.exit(1)

from PyQt6.QtWidgets import QApplication
from gui.scope_window import ScopeWindow


def main():
    app = QApplication(sys.argv)
    window = ScopeWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
