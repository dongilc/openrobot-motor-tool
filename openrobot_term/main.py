"""
Openrobot Terminal entry point.
"""

import sys
import os
import ctypes
import logging
from pathlib import Path

from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

log = logging.getLogger(__name__)


def resource_path(relative_path: str) -> str:
    """Resolve resource path for both dev and PyInstaller onefile."""
    base = getattr(sys, "_MEIPASS", str(Path(__file__).resolve().parent.parent))
    return str(Path(base) / relative_path)


def main():
    # Windows taskbar icon: set AppUserModelID so Windows shows our icon
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "openrobot.motor.analyzer.1")

    # Load .env — exe: next to exe file, dev: project root
    if getattr(sys, 'frozen', False):
        env_path = Path(sys.executable).parent / ".env"
    else:
        env_path = Path(__file__).resolve().parent.parent / ".env"

    _env_missing_path = None
    if env_path.exists():
        load_dotenv(env_path, override=True)
        log.info(".env loaded from %s", env_path)
    else:
        log.warning(".env not found at %s", env_path)
        _env_missing_path = str(env_path)

    app = QApplication(sys.argv)

    # Warn user if .env was not found (AI features won't work)
    if _env_missing_path:
        QMessageBox.warning(
            None, ".env not found",
            f".env file not found. AI features disabled.\n\n"
            f"Expected location:\n{_env_missing_path}\n\n"
            f"Create a .env file with:\nOPENAI_API_KEY=sk-..."
        )

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
