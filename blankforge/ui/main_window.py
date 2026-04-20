from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QLabel, QMainWindow, QMessageBox,
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
from blankforge.ui.tabs.tab_fins import FinsTab
from blankforge.ui.tabs.tab_template import NewBoardDialog
from blankforge.ui.tabs.tab_top_view import TopViewTab


class _WorkerSignals(QObject):
    finished = Signal(object, object)  # (BoardMesh, BoardStats)
    error = Signal(str)


class _GeometryWorker(QRunnable):
    def __init__(self, model: BoardModel, builder: BoardGeometryBuilder,
                 signals: _WorkerSignals, resolution: int = 50, n_contour: int = 32) -> None:
        super().__init__()
        self.signals = signals
        self._model = model.model_copy(deep=True)
        self._builder = builder
        self._resolution = resolution
        self._n_contour = n_contour

    @Slot()
    def run(self) -> None:
        try:
            mesh, stats = self._builder.build(self._model, self._resolution, self._n_contour)
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

        # Load samples/default.surfboard if it exists; otherwise fall back to shortboard template
        default_path = Path(__file__).parent.parent.parent / "samples" / "default.surfboard"
        self.model: BoardModel
        self._current_file: Path | None = None
        if default_path.exists():
            try:
                self.model = SurfboardSerializer.load(default_path)
            except Exception:
                self.model = BoardModel.from_template("shortboard")
        else:
            self.model = BoardModel.from_template("shortboard")

        self._builder = BoardGeometryBuilder(use_occt=False)
        self._build_pending = False
        self._active_workers: list[_GeometryWorker] = []
        self._mesh_resolution = 50  # stations along length
        self._mesh_contour = 32     # points per cross-section side

        self._build_ui()
        self._build_menu()
        self._connect_signals()
        self._trigger_geometry_build()

    def _build_ui(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.setCentralWidget(self._tabs)

        self._tab_parameters = ParametersTab(self.model, self.model_changed)
        self._tab_top = TopViewTab(self.model, self.model_changed)
        self._tab_side = SideViewTab(self.model, self.model_changed)
        self._tab_profile = ProfileViewTab(self.model, self.model_changed)
        self._tab_rendered = RenderedViewTab(self.model, self.model_changed)
        self._tab_quad = QuadViewTab(self.model, self.model_changed)
        self._tab_stats = StatisticsTab(self.model, self.model_changed)
        self._tab_export = ExportTab(self.model, self.model_changed)
        self._tab_fins = FinsTab(self.model, self.model_changed)

        self._tabs.addTab(self._tab_parameters, "Parameters")
        self._tabs.addTab(self._tab_top, "Top View")
        self._tabs.addTab(self._tab_side, "Side View")
        self._tabs.addTab(self._tab_profile, "Rails")
        self._tabs.addTab(self._tab_fins, "Fins")
        self._tabs.addTab(self._tab_rendered, "Rendered View")
        self._tabs.addTab(self._tab_quad, "Quad View")
        self._tabs.addTab(self._tab_stats, "Statistics")
        self._tabs.addTab(self._tab_export, "Export")

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

        view_menu = menu.addMenu("&View")
        self._fullscreen_action = view_menu.addAction("&Full Screen", self._toggle_fullscreen, "F11")
        self._fullscreen_action.setCheckable(True)

        help_menu = menu.addMenu("&Help")
        help_menu.addAction("About BlankForge", self._about)

    def _connect_signals(self) -> None:
        self.model_changed.connect(self._on_model_changed)
        self._tab_rendered.mesh_quality_changed.connect(self.set_mesh_quality)

    def _on_model_changed(self) -> None:
        self._tab_parameters.refresh_from_model()
        self._tab_top.refresh_from_model()
        self._tab_side.refresh_from_model()
        self._tab_profile.refresh_from_model()
        self._tab_quad.refresh_from_model()
        self._tab_fins.refresh_from_model()
        self._trigger_geometry_build()

    def _trigger_geometry_build(self) -> None:
        self._tab_stats.set_building()
        self._status_label.setText("Building geometry…")
        signals = _WorkerSignals(self)
        signals.finished.connect(self._on_geometry_ready)
        signals.error.connect(self._on_geometry_error)
        worker = _GeometryWorker(self.model, self._builder, signals,
                                 resolution=self._mesh_resolution,
                                 n_contour=self._mesh_contour)
        self._active_workers.append(worker)
        QThreadPool.globalInstance().start(worker)

    def set_mesh_quality(self, resolution: int, n_contour: int) -> None:
        self._mesh_resolution = resolution
        self._mesh_contour = n_contour
        self._trigger_geometry_build()

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
            self.model.curves = loaded.curves
            self._current_file = path
            self.setWindowTitle(f"BlankForge — {path.name}")
            self.model_changed.emit()
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def _new_file(self) -> None:
        dlg = NewBoardDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_model = BoardModel.from_template(dlg.selected_template())
        self.model.meta = new_model.meta
        self.model.parameters = new_model.parameters
        self.model.curves = new_model.curves
        self._current_file = None
        self.setWindowTitle("BlankForge — Untitled")
        self.model_changed.emit()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self._fullscreen_action.setChecked(False)
        else:
            self.showFullScreen()
            self._fullscreen_action.setChecked(True)

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
            if not path.lower().endswith(".surfboard"):
                path += ".surfboard"
            self._current_file = Path(path)
            SurfboardSerializer.save(self.model, self._current_file)
            self.setWindowTitle(f"BlankForge — {self._current_file.name}")
            self._status_label.setText(f"Saved: {self._current_file}")

    def _export_stl(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export STL", "", "STL Files (*.stl)")
        if path:
            if not path.lower().endswith(".stl"):
                path += ".stl"
            try:
                SurfboardSerializer.export_stl(self.model, Path(path))
                self._status_label.setText(f"Exported STL: {path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _export_obj(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "", "OBJ Files (*.obj)")
        if path:
            if not path.lower().endswith(".obj"):
                path += ".obj"
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
