"""
Openrobot Terminal entry point.
"""

import sys
import os
import ctypes
from pathlib import Path

from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt


def resource_path(relative_path: str) -> str:
    """Resolve resource path for both dev and PyInstaller onefile."""
    base = getattr(sys, "_MEIPASS", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / relative_path)


def main():
    # Windows taskbar icon: set AppUserModelID so Windows shows our icon
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "openrobot.motor.analyzer.1")

    # Load .env from project root
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    app = QApplication(sys.argv)

    # Set icon — search multiple locations, prefer .png
    app_icon = QIcon()
    for name in ("openrobot_term.png", "openrobot_term.ico"):
        # Try resource_path (handles PyInstaller _MEIPASS)
        p = resource_path(name)
        if os.path.isfile(p):
            app_icon = QIcon(p)
            break
        # Also try relative to this file directly
        p2 = str(Path(__file__).resolve().parent.parent / name)
        if os.path.isfile(p2):
            app_icon = QIcon(p2)
            break
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    from .ui.plot_style import apply_global_theme
    apply_global_theme()

    from .ui.main_window import MainWindow

    w = MainWindow()
    if not app_icon.isNull():
        w.setWindowIcon(app_icon)
    w.resize(1680, 980)
    w.show()

    w.resizeDocks([w.motor_dock], [320], Qt.Orientation.Horizontal)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
