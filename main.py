from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    from PySide6.QtGui import QIcon, QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from blankforge.ui.main_window import BlankForgeWindow

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setApplicationName("BlankForge")
    app.setOrganizationName("BlankForge")
    app.setStyle("Fusion")

    icon_path = Path(__file__).parent / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = BlankForgeWindow()

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.suffix == ".surfboard" and path.exists():
            window.load_file(path)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
