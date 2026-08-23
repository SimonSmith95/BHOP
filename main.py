"""
Phase 12 — Entry Point
Launch the BHOP PySide6 application.

Usage
-----
    python main.py
"""
import sys

from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("BHOP")
    app.setOrganizationName("Lab")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
