from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QMainWindow, QMessageBox,
    QStatusBar, QTabWidget, QWidget,
)

from blankforge.data.model import BoardModel
from blankforge.data.serializer import SurfboardSerializer
from blankforge.geometry.board import BoardGeometryBuilder, BoardMesh, BoardStats
from blankforge.ui.tabs.tab_export import ExportTab
from blankforge.ui.tabs.tab_parameters import ParametersTab
from blankforge.ui.tabs.tab_profile_view import ProfileViewTab
from blankforge.ui.tabs.tab_quad_view import QuadViewTab
from blankforge.ui.tabs.tab_rendered_view import RenderedViewTab
from blankforge.ui.tabs.tab_side_view import SideViewTab
from blankforge.ui.tabs.tab_statistics import StatisticsTab
from blankforge.ui.tabs.tab_template import TemplateTab
from blankforge.ui.tabs.tab_top_view import TopViewTab


class _WorkerSignals(QObject):
    finished = Signal(object, object)  # (BoardMesh, BoardStats)
    error = Signal(str)


class _GeometryWorker(QRunnable):
    def __init__(self, model: BoardModel, builder: BoardGeometryBuilder, signals: _WorkerSignals) -> None:
        super().__init__()
        self.signals = signals
        self._model = model.model_copy(deep=True)
        self._builder = builder

    @Slot()
    def run(self) -> None:
        try:
            mesh, stats = self._builder.build(self._model)
            self.signals.finished.emit(mesh, stats)
        except RuntimeError:
            pass  # Window may have closed before thread finished
        except Exception as e:
            try:
                self.signals.error.emit(str(e))
            except RuntimeError:
                pass


class BlankForgeWindow(QMainWindow):
    model_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BlankForge — Parametric Surfboard Designer")
        self.resize(1400, 900)

        from pathlib import Path as _Path
        from PySide6.QtGui import QIcon as _QIcon
        _icon = _Path(__file__).parent.parent.parent / "icon.png"
        if _icon.exists():
            self.setWindowIcon(_QIcon(str(_icon)))

        self.model = BoardModel.from_template("shortboard")
        self._builder = BoardGeometryBuilder(use_occt=False)
        self._current_file: Path | None = None
        self._build_pending = False
        self._active_workers: list[_GeometryWorker] = []

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._trigger_geometry_build()

    def _build_ui(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.setCentralWidget(self._tabs)

        self._tab_template = TemplateTab(self.model, self.model_changed)
        self._tab_parameters = ParametersTab(self.model, self.model_changed)
        self._tab_top = TopViewTab(self.model, self.model_changed)
        self._tab_side = SideViewTab(self.model, self.model_changed)
        self._tab_profile = ProfileViewTab(self.model, self.model_changed)
        self._tab_rendered = RenderedViewTab(self.model, self.model_changed)
        self._tab_quad = QuadViewTab(self.model, self.model_changed)
        self._tab_stats = StatisticsTab(self.model, self.model_changed)
        self._tab_export = ExportTab(self.model, self.model_changed)

        self._tabs.addTab(self._tab_template, "1 Template")
        self._tabs.addTab(self._tab_parameters, "2 Parameters")
        self._tabs.addTab(self._tab_top, "3 Top View")
        self._tabs.addTab(self._tab_side, "4 Side View")
        self._tabs.addTab(self._tab_profile, "5 Profile View")
        self._tabs.addTab(self._tab_rendered, "6 Rendered View")
        self._tabs.addTab(self._tab_quad, "7 Quad View")
        self._tabs.addTab(self._tab_stats, "8 Statistics")
        self._tabs.addTab(self._tab_export, "9 Export")

        self._status_label = QLabel("Ready")
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_label)
        self.setStatusBar(status_bar)

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        file_menu.addAction("&New", self._new_file, "Ctrl+N")
        file_menu.addAction("&Open…", self._open_file, "Ctrl+O")
        file_menu.addSeparator()
        file_menu.addAction("&Save", self._save_file, "Ctrl+S")
        file_menu.addAction("Save &As…", self._save_file_as, "Ctrl+Shift+S")
        file_menu.addSeparator()
        file_menu.addAction("Export ST&L…", self._export_stl)
        file_menu.addAction("Export &OBJ…", self._export_obj)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        help_menu = menu.addMenu("&Help")
        help_menu.addAction("About BlankForge", self._about)

    def _connect_signals(self) -> None:
        self.model_changed.connect(self._on_model_changed)

    def _on_model_changed(self) -> None:
        self._tab_parameters.refresh_from_model()
        self._tab_top.refresh_from_model()
        self._tab_side.refresh_from_model()
        self._tab_profile.refresh_from_model()
        self._tab_quad.refresh_from_model()
        self._tab_template.refresh_from_model()
        self._trigger_geometry_build()

    def _trigger_geometry_build(self) -> None:
        self._tab_stats.set_building()
        self._status_label.setText("Building geometry…")
        signals = _WorkerSignals(self)
        signals.finished.connect(self._on_geometry_ready)
        signals.error.connect(self._on_geometry_error)
        worker = _GeometryWorker(self.model, self._builder, signals)
        self._active_workers.append(worker)
        QThreadPool.globalInstance().start(worker)

    def _on_geometry_ready(self, mesh: BoardMesh, stats: BoardStats) -> None:
        self._tab_rendered.update_mesh(mesh)
        self._tab_quad.update_mesh(mesh)
        self._tab_stats.update_stats(stats)
        self._status_label.setText(
            f"Volume: {stats.volume_cm3 / 1000:.1f} L  |  "
            f"Length: {stats.length_mm:.0f} mm  |  "
            f"Width: {stats.width_mm:.0f} mm  |  "
            f"Thickness: {stats.thickness_mm:.1f} mm"
        )

    def _on_geometry_error(self, msg: str) -> None:
        self._status_label.setText(f"Geometry error: {msg}")

    def load_file(self, path: Path) -> None:
        try:
            loaded = SurfboardSerializer.load(path)
            self.model.meta = loaded.meta
            self.model.parameters = loaded.parameters
            self.model.tail = loaded.tail
            self.model.curves = loaded.curves
            self._current_file = path
            self.setWindowTitle(f"BlankForge — {path.name}")
            self.model_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def _new_file(self) -> None:
        new_model = BoardModel.from_template("shortboard")
        self.model.meta = new_model.meta
        self.model.parameters = new_model.parameters
        self.model.tail = new_model.tail
        self.model.curves = new_model.curves
        self._current_file = None
        self.setWindowTitle("BlankForge — Untitled")
        self.model_changed.emit()

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Surfboard", "", "Surfboard Files (*.surfboard);;All Files (*)")
        if path:
            self.load_file(Path(path))

    def _save_file(self) -> None:
        if self._current_file:
            SurfboardSerializer.save(self.model, self._current_file)
            self._status_label.setText(f"Saved: {self._current_file}")
        else:
            self._save_file_as()

    def _save_file_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save As", "", "Surfboard Files (*.surfboard)")
        if path:
            self._current_file = Path(path)
            SurfboardSerializer.save(self.model, self._current_file)
            self.setWindowTitle(f"BlankForge — {self._current_file.name}")
            self._status_label.setText(f"Saved: {self._current_file}")

    def _export_stl(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export STL", "", "STL Files (*.stl)")
        if path:
            try:
                SurfboardSerializer.export_stl(self.model, Path(path))
                self._status_label.setText(f"Exported STL: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _export_obj(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "", "OBJ Files (*.obj)")
        if path:
            try:
                SurfboardSerializer.export_obj(self.model, Path(path))
                self._status_label.setText(f"Exported OBJ: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About BlankForge",
            "<h3>BlankForge</h3>"
            "<p>Parametric Surfboard Designer v0.1.0</p>"
            "<p>Design surfboard shapes parametrically using Bezier/NURBS curves.<br>"
            "Export to .surfboard (JSON), STL, or OBJ.</p>",
        )
