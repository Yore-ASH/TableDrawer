"""绘图页：数据序列配置 + matplotlib 画布 + 图片/源码导出。"""

from __future__ import annotations

import json
import os
from typing import Any

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core import plotting
from core.dataset import DataSetStore
from core.spec import (
    COLORMAPS,
    KINDS_2D,
    KINDS_3D,
    LINESTYLES,
    MARKERS,
    SCALES,
    THEMES,
    PlotSpec,
    SeriesSpec,
    kind_label,
    kinds_for,
)

from .dialogs import ExportSourceDialog, TextInfoDialog
from .widgets import MplCanvas

__all__ = ["PlotTab"]

#: 各图表类型需要显示的字段（``dataset`` / ``kind`` / ``label`` / ``color`` 始终显示）
_KIND_FIELDS: dict[str, set[str]] = {
    "line": {"x", "y", "expr", "marker", "linestyle", "linewidth", "markersize", "alpha", "smooth"},
    "scatter": {"x", "y", "expr", "marker", "markersize", "alpha"},
    "bar": {"x", "y", "expr", "bar_width", "alpha"},
    "barh": {"x", "y", "expr", "bar_width", "alpha"},
    "stem": {"x", "y", "expr", "marker", "linestyle", "linewidth", "markersize"},
    "step": {"x", "y", "expr", "linestyle", "linewidth"},
    "fill": {"x", "y", "expr", "alpha", "filled"},
    "errorbar": {"x", "y", "yerr", "marker", "linestyle", "linewidth", "markersize"},
    "hist": {"y", "bins", "density", "cumulative", "alpha"},
    "box": {"y"},
    "pie": {"x", "y", "alpha"},
    "quiver": {"x", "y", "u", "v", "scale", "normalize"},
    "contour": {"x", "y", "z", "levels", "cmap"},
    "heatmap": {"x", "y", "z", "cmap"},
    "line3d": {"x", "y", "z", "marker", "linestyle", "linewidth", "markersize"},
    "scatter3d": {"x", "y", "z", "marker", "markersize", "alpha"},
    "surface": {"x", "y", "z", "cmap"},
    "wireframe": {"x", "y", "z", "cmap", "linewidth"},
    "contour3d": {"x", "y", "z", "levels", "cmap"},
    "trisurf": {"x", "y", "z", "cmap"},
    "bar3d": {"x", "y", "z", "bar_width", "cmap", "alpha"},
    "quiver3d": {"x", "y", "z", "u", "v", "w", "scale", "normalize"},
}

_ALWAYS_FIELDS = {"dataset", "kind", "label", "color"}

#: 字段 -> 中文标签
_FIELD_LABELS: dict[str, str] = {
    "dataset": "数据集",
    "kind": "图表类型",
    "x": "X 列",
    "y": "Y 列",
    "z": "Z 列",
    "u": "U 分量",
    "v": "V 分量",
    "w": "W 分量",
    "yerr": "误差列",
    "expr": "Y 表达式",
    "label": "图例标签",
    "color": "颜色",
    "cmap": "色图",
    "marker": "标记",
    "linestyle": "线型",
    "linewidth": "线宽",
    "markersize": "标记大小",
    "alpha": "透明度",
    "bar_width": "柱宽",
    "bins": "分组数",
    "levels": "层数",
    "scale": "箭头缩放",
    "normalize": "单位化箭头",
    "density": "归一化密度",
    "cumulative": "累计分布",
    "filled": "填充",
    "smooth": "平滑曲线",
}

#: 不需要「（行号）」选项的字段
_NO_INDEX_FIELDS = {"y", "z", "u", "v", "w", "yerr", "expr"}


class PlotTab(QWidget):
    """绘图工作区。"""

    statusMessage = Signal(str)

    def __init__(self, store: DataSetStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self.spec = PlotSpec()
        self._series: list[SeriesSpec] = []
        self._current = -1
        self._loading = False
        self._palette: dict[str, str] = {}
        self._color_value = ""

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(260)
        self._timer.timeout.connect(self.render)

        self._build_ui()
        self.store.subscribe(self.refresh_datasets)

    # ================================================================== #
    # 界面构建
    # ================================================================== #
    def _build_ui(self) -> None:
        # ---------------- 序列列表 ---------------- #
        self.series_list = QListWidget()
        self.series_list.setMinimumHeight(120)
        self.series_list.currentRowChanged.connect(self._on_series_selected)

        add_btn = QPushButton("添加")
        add_btn.clicked.connect(self.add_series)
        dup_btn = QPushButton("复制")
        dup_btn.clicked.connect(self.duplicate_series)
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(self.remove_series)
        up_btn = QPushButton("↑")
        up_btn.setFixedWidth(34)
        up_btn.clicked.connect(lambda: self.move_series(-1))
        down_btn = QPushButton("↓")
        down_btn.setFixedWidth(34)
        down_btn.clicked.connect(lambda: self.move_series(1))

        list_buttons = QHBoxLayout()
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(dup_btn)
        list_buttons.addWidget(del_btn)
        list_buttons.addStretch(1)
        list_buttons.addWidget(up_btn)
        list_buttons.addWidget(down_btn)

        series_box = QGroupBox("数据序列")
        series_layout = QVBoxLayout(series_box)
        series_layout.addWidget(self.series_list, 1)
        series_layout.addLayout(list_buttons)

        # ---------------- 序列详细设置 ---------------- #
        self._widgets: dict[str, QWidget] = {}
        self.series_form = QFormLayout()
        self.series_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.series_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self._on_series_dataset_changed)
        self._widgets["dataset"] = self.dataset_combo

        self.kind_combo = QComboBox()
        self.kind_combo.currentIndexChanged.connect(self._on_kind_changed)
        self._widgets["kind"] = self.kind_combo

        self.x_combo = self._make_column_combo()
        self.y_combo = self._make_column_combo(editable=True)
        self.y_combo.setToolTip("可以填写多列（英文逗号分隔），每列画一条序列")
        self.z_combo = self._make_column_combo()
        self.u_combo = self._make_column_combo()
        self.v_combo = self._make_column_combo()
        self.w_combo = self._make_column_combo()
        self.yerr_combo = self._make_column_combo()

        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("可留空；例如 sin(x)*exp(-x/5)")
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("留空则自动生成")

        self.color_button = QPushButton("自动")
        self.color_button.clicked.connect(self._pick_color)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.marker_combo = QComboBox()
        for key, text in MARKERS.items():
            self.marker_combo.addItem(text, key)
        self.linestyle_combo = QComboBox()
        for key, text in LINESTYLES.items():
            self.linestyle_combo.addItem(text, key)

        self.linewidth_spin = self._make_double(0.0, 30.0, 2.0, 0.5)
        self.markersize_spin = self._make_double(0.5, 60.0, 6.0, 0.5)
        self.alpha_spin = self._make_double(0.05, 1.0, 1.0, 0.05)
        self.barwidth_spin = self._make_double(0.05, 2.0, 0.8, 0.05)
        self.levels_spin = QSpinBox()
        self.levels_spin.setRange(2, 200)
        self.levels_spin.setValue(12)
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(1, 500)
        self.bins_spin.setValue(20)
        self.scale_spin = self._make_double(0.01, 1000.0, 1.0, 0.1)
        self.normalize_check = QCheckBox("单位长度")
        self.density_check = QCheckBox("概率密度")
        self.cumulative_check = QCheckBox("累计")
        self.filled_check = QCheckBox("填充曲线下方")
        self.filled_check.setChecked(True)
        self.smooth_check = QCheckBox("平滑")

        self._widgets.update(
            {
                "x": self.x_combo,
                "y": self.y_combo,
                "z": self.z_combo,
                "u": self.u_combo,
                "v": self.v_combo,
                "w": self.w_combo,
                "yerr": self.yerr_combo,
                "expr": self.expr_edit,
                "label": self.label_edit,
                "color": self.color_button,
                "cmap": self.cmap_combo,
                "marker": self.marker_combo,
                "linestyle": self.linestyle_combo,
                "linewidth": self.linewidth_spin,
                "markersize": self.markersize_spin,
                "alpha": self.alpha_spin,
                "bar_width": self.barwidth_spin,
                "bins": self.bins_spin,
                "levels": self.levels_spin,
                "scale": self.scale_spin,
                "normalize": self.normalize_check,
                "density": self.density_check,
                "cumulative": self.cumulative_check,
                "filled": self.filled_check,
                "smooth": self.smooth_check,
            }
        )

        for key in (
            "dataset", "kind", "x", "y", "z", "u", "v", "w", "yerr", "expr",
            "label", "color", "cmap", "marker", "linestyle", "linewidth",
            "markersize", "alpha", "bar_width", "bins", "levels", "scale",
            "normalize", "density", "cumulative", "filled", "smooth",
        ):
            self.series_form.addRow(_FIELD_LABELS[key], self._widgets[key])

        self.kind_help = QLabel("")
        self.kind_help.setWordWrap(True)
        self.kind_help.setProperty("muted", "true")

        series_detail = QGroupBox("序列设置")
        detail_layout = QVBoxLayout(series_detail)
        detail_layout.addLayout(self.series_form)
        detail_layout.addWidget(self.kind_help)

        # ---------------- 坐标轴 / 画布 ---------------- #
        self.projection_combo = QComboBox()
        self.projection_combo.addItem("二维 (2D)", "2d")
        self.projection_combo.addItem("三维 (3D)", "3d")
        self.projection_combo.currentIndexChanged.connect(self._on_projection_changed)

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("支持 $\\LaTeX$ 公式，如 $y=e^{-x}$")
        self.xlabel_edit = QLineEdit()
        self.ylabel_edit = QLineEdit()
        self.zlabel_edit = QLineEdit()
        self.theme_combo = QComboBox()
        for key, text in THEMES.items():
            self.theme_combo.addItem(text, key)

        self.figwidth_spin = self._make_double(2.0, 60.0, 8.0, 0.5)
        self.figheight_spin = self._make_double(2.0, 60.0, 5.0, 0.5)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 600)
        self.dpi_spin.setValue(110)
        self.grid_check = QCheckBox("显示网格")
        self.grid_check.setChecked(True)
        self.legend_check = QCheckBox("显示图例")
        self.legend_check.setChecked(True)
        self.legend_loc_combo = QComboBox()
        for key, text in (
            ("best", "自动"), ("upper right", "右上"), ("upper left", "左上"),
            ("lower right", "右下"), ("lower left", "左下"), ("center", "居中"),
            ("right", "右侧"), ("upper center", "上中"), ("lower center", "下中"),
        ):
            self.legend_loc_combo.addItem(text, key)
        self.aspect_check = QCheckBox("等比例坐标轴")
        self.xscale_combo = self._make_scale_combo()
        self.yscale_combo = self._make_scale_combo()
        self.zscale_combo = self._make_scale_combo()
        self.xlim_edit = QLineEdit()
        self.xlim_edit.setPlaceholderText("自动，或 0,10")
        self.ylim_edit = QLineEdit()
        self.ylim_edit.setPlaceholderText("自动，或 -1,1")
        self.zlim_edit = QLineEdit()
        self.zlim_edit.setPlaceholderText("自动")
        self.colorbar_check = QCheckBox("显示色条")
        self.colorbar_check.setChecked(True)
        self.elev_spin = self._make_double(-90.0, 90.0, 30.0, 5.0)
        self.azim_spin = self._make_double(-180.0, 180.0, -60.0, 5.0)

        axis_form = QFormLayout()
        axis_form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        axis_form.addRow("投影", self.projection_combo)
        axis_form.addRow("标题", self.title_edit)
        axis_form.addRow("X 轴标签", self.xlabel_edit)
        axis_form.addRow("Y 轴标签", self.ylabel_edit)
        axis_form.addRow("Z 轴标签", self.zlabel_edit)
        axis_form.addRow("X 范围", self.xlim_edit)
        axis_form.addRow("Y 范围", self.ylim_edit)
        axis_form.addRow("Z 范围", self.zlim_edit)
        axis_form.addRow("X 刻度", self.xscale_combo)
        axis_form.addRow("Y 刻度", self.yscale_combo)
        axis_form.addRow("Z 刻度", self.zscale_combo)
        axis_form.addRow("", self.grid_check)
        axis_form.addRow("", self.legend_check)
        axis_form.addRow("图例位置", self.legend_loc_combo)
        axis_form.addRow("", self.aspect_check)
        axis_form.addRow("", self.colorbar_check)
        axis_form.addRow("俯仰角", self.elev_spin)
        axis_form.addRow("方位角", self.azim_spin)
        axis_form.addRow("主题", self.theme_combo)
        fig_row = QHBoxLayout()
        fig_row.addWidget(self.figwidth_spin)
        fig_row.addWidget(QLabel("×"))
        fig_row.addWidget(self.figheight_spin)
        fig_row.addWidget(QLabel("in @"))
        fig_row.addWidget(self.dpi_spin)
        fig_row.addWidget(QLabel("dpi"))
        fig_row.addStretch(1)
        fig_row_widget = QWidget()
        fig_row_widget.setLayout(fig_row)
        axis_form.addRow("画布", fig_row_widget)

        axis_box = QGroupBox("坐标轴与画布")
        axis_box.setLayout(axis_form)
        self._axis_form = axis_form

        #: 仅三维模式可用的行（表单标签与控件一起隐藏）
        self._z_extra_widgets = [
            self.zlabel_edit,
            self.zscale_combo,
            self.zlim_edit,
            self.elev_spin,
            self.azim_spin,
            self.colorbar_check,
        ]

        # ---------------- 左侧面板 ---------------- #
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(2, 2, 8, 2)
        left_layout.addWidget(series_box)
        left_layout.addWidget(series_detail)
        left_layout.addWidget(axis_box)
        left_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_content)
        # 关掉横向滚动条：表单宽度跟随视口，不会出现需要左右拖的情况
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 只给最小宽度，不设上限：硬上限会让分隔条拖到一半就「顶住」，
        # 手感上像是往反方向跳
        scroll.setMinimumWidth(340)

        # ---------------- 右侧画布 ---------------- #
        self.canvas = MplCanvas(figsize=(8, 5), dpi=110)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setProperty("muted", "true")
        self.warning_label.setMaximumHeight(48)

        self.auto_check = QCheckBox("自动重绘")
        self.auto_check.setChecked(True)

        # 画布工具条同样用 QToolBar：窄窗口时自动收进「»」溢出菜单，不会被裁掉
        self.canvas_bar = QToolBar()
        self.canvas_bar.setIconSize(QSize(16, 16))
        self.canvas_bar.setMovable(False)
        self.canvas_bar.setStyleSheet(
            "QToolBar { background: transparent; border: none; padding: 0; spacing: 2px; }"
        )
        render_action = QAction("重绘", self)
        render_action.setToolTip("按当前设置重新绘制（F5）")
        render_action.triggered.connect(self.render)
        self.canvas_bar.addAction(render_action)
        self.canvas_bar.addWidget(self.auto_check)
        self.canvas_bar.addSeparator()
        for text, slot, tip in (
            ("复制图片", self.copy_image, "把当前图复制到剪贴板"),
            ("导出图片…", self.export_image, "导出 SVG / PNG / PDF / PGF"),
            ("导出源码…", self.export_source, "导出可独立运行的 matplotlib 源码"),
            ("保存配置", self.save_config, "把绘图设置保存为 JSON"),
            ("载入配置", self.load_config, "从 JSON 载入绘图设置"),
            ("公式语法", self.show_syntax_help, "查看标签支持的 LaTeX 语法"),
        ):
            action = QAction(text, self)
            action.setToolTip(tip)
            action.triggered.connect(slot)
            self.canvas_bar.addAction(action)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas_bar)
        right_layout.addWidget(self.canvas, 1)
        right_layout.addWidget(self.warning_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        # 禁止把面板拖成 0 宽（塌掉再弹回来会像乱跳）；要逐个设置才生效
        splitter.setChildrenCollapsible(False)
        for index in range(splitter.count()):
            splitter.setCollapsible(index, False)
        right.setMinimumWidth(320)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 900])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)

        # ---------------- 连接信号 ---------------- #
        for key, widget in self._widgets.items():
            if key in ("dataset", "kind"):
                continue
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._on_series_edited)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._on_series_edited)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._on_series_edited)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._on_series_edited)

        for widget in (
            self.title_edit, self.xlabel_edit, self.ylabel_edit, self.zlabel_edit,
            self.xlim_edit, self.ylim_edit, self.zlim_edit,
        ):
            widget.textChanged.connect(self._on_spec_edited)
        for widget in (
            self.theme_combo, self.legend_loc_combo, self.xscale_combo,
            self.yscale_combo, self.zscale_combo,
        ):
            widget.currentIndexChanged.connect(self._on_spec_edited)
        for widget in (
            self.grid_check, self.legend_check, self.aspect_check,
            self.colorbar_check,
        ):
            widget.toggled.connect(self._on_spec_edited)
        for widget in (
            self.figwidth_spin, self.figheight_spin, self.elev_spin,
            self.azim_spin, self.dpi_spin,
        ):
            widget.valueChanged.connect(self._on_spec_edited)

        self._on_projection_changed()

    # ------------------------------------------------------------------ #
    # 小组件工厂
    # ------------------------------------------------------------------ #
    def _make_column_combo(self, *, editable: bool = False) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(editable)
        if editable:
            combo.setInsertPolicy(QComboBox.NoInsert)
        return combo

    @staticmethod
    def _make_double(low: float, high: float, value: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        return spin

    @staticmethod
    def _make_scale_combo() -> QComboBox:
        combo = QComboBox()
        for key, text in SCALES.items():
            combo.addItem(text, key)
        return combo

    # ================================================================== #
    # 数据集
    # ================================================================== #
    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = dict(palette)

    def refresh_datasets(self) -> None:
        names = self.store.names()
        current = self.dataset_combo.currentText()
        self._loading = True
        try:
            self.dataset_combo.clear()
            self.dataset_combo.addItems(names)
            if current in names:
                self.dataset_combo.setCurrentText(current)
            elif names:
                self.dataset_combo.setCurrentIndex(0)
        finally:
            self._loading = False
        self._reload_columns()
        if not self._series and names:
            self.new_default_plot(names[0])
        elif self._series:
            self._on_series_selected(self._current)

    def _reload_columns(self) -> None:
        """按当前数据集刷新所有列下拉框。"""
        name = self.dataset_combo.currentText()
        dataset = self.store.get(name)
        columns = dataset.columns if dataset else []
        numeric = dataset.numeric_columns if dataset else []

        for key in ("x", "z", "u", "v", "w", "yerr", "y"):
            combo = self._widgets[key]
            previous = combo.currentText()
            self._loading = True
            try:
                combo.clear()
                if key == "x":
                    combo.addItem("（行号）", "")
                options = numeric if (key in {"u", "v", "w", "yerr", "z"} and numeric) else columns
                for column in options:
                    combo.addItem(str(column), str(column))
                if key == "y" and not numeric and columns:
                    combo.clear()
                    for column in columns:
                        combo.addItem(str(column), str(column))
                if previous:
                    combo.setCurrentText(previous)
            finally:
                self._loading = False

    # ================================================================== #
    # 序列管理
    # ================================================================== #
    def _kinds(self) -> dict[str, str]:
        projection = self.projection_combo.currentData() or "2d"
        return kinds_for(projection)

    def add_series(self) -> None:
        datasets = self.store.names()
        if not datasets:
            QMessageBox.information(self, "没有数据", "请先在「数据」页导入或添加数据集。")
            return
        name = self.dataset_combo.currentText() or datasets[0]
        dataset = self.store.get(name)
        kind = self.kind_combo.currentData() or ("line3d" if self._is_3d() else "line")
        series = plotting.default_series(name, dataset.frame, kind) if dataset else SeriesSpec(dataset=name, kind=kind)
        self._series.append(series)
        self._refresh_series_list(select=len(self._series) - 1)
        self._on_series_selected(len(self._series) - 1)

    def duplicate_series(self) -> None:
        if not (0 <= self._current < len(self._series)):
            return
        clone = SeriesSpec.from_dict(self._series[self._current].to_dict())
        self._series.insert(self._current + 1, clone)
        self._refresh_series_list(select=self._current + 1)
        self._on_series_selected(self._current + 1)

    def remove_series(self) -> None:
        if not (0 <= self._current < len(self._series)):
            return
        self._series.pop(self._current)
        new_index = min(self._current, len(self._series) - 1)
        self._refresh_series_list(select=new_index)
        self._on_series_selected(new_index)

    def move_series(self, delta: int) -> None:
        index = self._current
        target = index + delta
        if not (0 <= index < len(self._series)) or not (0 <= target < len(self._series)):
            return
        self._series[index], self._series[target] = self._series[target], self._series[index]
        self._refresh_series_list(select=target)
        self._on_series_selected(target)

    def _refresh_series_list(self, *, select: int | None = None) -> None:
        self._loading = True
        try:
            self.series_list.clear()
            for index, series in enumerate(self._series):
                item = QListWidgetItem(self._series_caption(index, series))
                self.series_list.addItem(item)
            if select is not None and 0 <= select < self.series_list.count():
                self.series_list.setCurrentRow(select)
        finally:
            self._loading = False
        self.schedule_render()

    @staticmethod
    def _series_caption(index: int, series: SeriesSpec) -> str:
        target = series.y or series.z or series.v or ""
        return f"{index + 1}. {kind_label(series.kind)}　{series.dataset}［{series.x or '行号'} → {target}］"

    def _on_series_selected(self, row: int) -> None:
        self._current = row
        if not (0 <= row < len(self._series)):
            return
        self._apply_series(self._series[row])

    def _on_series_dataset_changed(self, name: str) -> None:
        if self._loading:
            return
        self._reload_columns()
        self._on_series_edited()

    def _on_kind_changed(self, *_: Any) -> None:
        if self._loading:
            return
        self._update_field_visibility()
        help_text = plotting.KIND_HELP.get(self.kind_combo.currentData() or "", "")
        self.kind_help.setText(help_text)
        self._on_series_edited()

    def _on_series_edited(self, *_: Any) -> None:
        if self._loading or not (0 <= self._current < len(self._series)):
            return
        series = self._read_series()
        self._series[self._current] = series
        self._loading = True
        try:
            item = self.series_list.item(self._current)
            if item is not None:
                item.setText(self._series_caption(self._current, series))
        finally:
            self._loading = False
        self.schedule_render()

    def _on_spec_edited(self, *_: Any) -> None:
        if self._loading:
            return
        self.schedule_render()

    # ------------------------------------------------------------------ #
    # 表单 <-> 数据类
    # ------------------------------------------------------------------ #
    def _apply_series(self, series: SeriesSpec) -> None:
        """把序列写入表单。"""
        self._loading = True
        try:
            index = self.dataset_combo.findText(series.dataset)
            if index < 0 and self.dataset_combo.count():
                index = 0
            self.dataset_combo.setCurrentIndex(max(index, 0))
            self._reload_columns_silent()

            kinds = self._kinds()
            kind_index = self.kind_combo.findData(series.kind)
            if kind_index < 0:
                kind_index = 0
            self.kind_combo.setCurrentIndex(kind_index)
            self._update_field_visibility()

            self.x_combo.setCurrentText(series.x)
            self.y_combo.setCurrentText(series.y)
            self.z_combo.setCurrentText(series.z)
            self.u_combo.setCurrentText(series.u)
            self.v_combo.setCurrentText(series.v)
            self.w_combo.setCurrentText(series.w)
            self.yerr_combo.setCurrentText(series.yerr)
            self.expr_edit.setText(series.expr)
            self.label_edit.setText(series.label)

            self._color_value = series.color
            self._update_color_button()

            cmap_index = self.cmap_combo.findText(series.cmap)
            self.cmap_combo.setCurrentIndex(max(cmap_index, 0))
            marker_index = self.marker_combo.findData(series.marker)
            self.marker_combo.setCurrentIndex(max(marker_index, 0))
            style_index = self.linestyle_combo.findData(series.linestyle)
            self.linestyle_combo.setCurrentIndex(max(style_index, 0))

            self.linewidth_spin.setValue(series.linewidth)
            self.markersize_spin.setValue(series.markersize)
            self.alpha_spin.setValue(series.alpha)
            self.barwidth_spin.setValue(series.bar_width)
            self.bins_spin.setValue(series.bins)
            self.levels_spin.setValue(series.levels)
            self.scale_spin.setValue(series.scale)
            self.normalize_check.setChecked(series.normalize)
            self.density_check.setChecked(series.density)
            self.cumulative_check.setChecked(series.cumulative)
            self.filled_check.setChecked(series.filled)
            self.smooth_check.setChecked(series.smooth)

            self.kind_help.setText(plotting.KIND_HELP.get(series.kind, ""))
        finally:
            self._loading = False

    def _reload_columns_silent(self) -> None:
        was_loading = self._loading
        self._loading = True
        try:
            self._reload_columns()
        finally:
            self._loading = was_loading

    def _read_series(self) -> SeriesSpec:
        series = self._series[self._current] if 0 <= self._current < len(self._series) else SeriesSpec()
        series.dataset = self.dataset_combo.currentText()
        series.kind = self.kind_combo.currentData() or series.kind
        series.x = self.x_combo.currentText().strip()
        series.y = self.y_combo.currentText().strip()
        series.z = self.z_combo.currentText().strip()
        series.u = self.u_combo.currentText().strip()
        series.v = self.v_combo.currentText().strip()
        series.w = self.w_combo.currentText().strip()
        series.yerr = self.yerr_combo.currentText().strip()
        series.expr = self.expr_edit.text().strip()
        series.label = self.label_edit.text().strip()
        series.color = self._color_value
        series.cmap = self.cmap_combo.currentText()
        series.marker = self.marker_combo.currentData() or ""
        series.linestyle = self.linestyle_combo.currentData() or ""
        series.linewidth = float(self.linewidth_spin.value())
        series.markersize = float(self.markersize_spin.value())
        series.alpha = float(self.alpha_spin.value())
        series.bar_width = float(self.barwidth_spin.value())
        series.bins = int(self.bins_spin.value())
        series.levels = int(self.levels_spin.value())
        series.scale = float(self.scale_spin.value())
        series.normalize = self.normalize_check.isChecked()
        series.density = self.density_check.isChecked()
        series.cumulative = self.cumulative_check.isChecked()
        series.filled = self.filled_check.isChecked()
        series.smooth = self.smooth_check.isChecked()
        return series

    def _update_field_visibility(self) -> None:
        kind = self.kind_combo.currentData() or "line"
        visible = _ALWAYS_FIELDS | _KIND_FIELDS.get(kind, set())
        if not self._is_3d():
            visible.discard("z")
        for key, widget in self._widgets.items():
            show = key in visible
            widget.setVisible(show)
            label = self.series_form.labelForField(widget)
            if label is not None:
                label.setVisible(show)
        self._update_3d_rows()

    def _update_3d_rows(self) -> None:
        """显示 / 隐藏只在三维模式下有意义的设置行。"""
        is_3d = self._is_3d()
        for widget in self._z_extra_widgets:
            widget.setVisible(is_3d)
            label = self._axis_form.labelForField(widget)
            if label is not None:
                label.setVisible(is_3d)

    def _is_3d(self) -> bool:
        return (self.projection_combo.currentData() or "2d") == "3d"

    # ------------------------------------------------------------------ #
    # 颜色
    # ------------------------------------------------------------------ #
    def _update_color_button(self) -> None:
        if self._color_value:
            self.color_button.setText(self._color_value)
            self.color_button.setStyleSheet(
                f"background-color: {self._color_value}; color: "
                f"{'#000000' if QColor(self._color_value).lightness() > 140 else '#ffffff'};"
            )
        else:
            self.color_button.setText("自动（按色环循环）")
            self.color_button.setStyleSheet("")

    def _pick_color(self) -> None:
        initial = QColor(self._color_value) if self._color_value else QColor("#4c8dff")
        color = QColorDialog.getColor(initial, self, "选择颜色", QColorDialog.ShowAlphaChannel)
        if color.isValid():
            self._color_value = color.name()
        else:
            return
        self._update_color_button()
        self._on_series_edited()

    # ================================================================== #
    # 渲染
    # ================================================================== #
    def schedule_render(self) -> None:
        """按需触发重绘（自动重绘关闭时只在点「重绘」时渲染）。"""
        if getattr(self, "auto_check", None) is None or self.auto_check.isChecked():
            self._timer.start()

    def _collect_spec(self) -> PlotSpec:
        spec = PlotSpec()
        spec.projection = self.projection_combo.currentData() or "2d"
        spec.title = self.title_edit.text()
        spec.xlabel = self.xlabel_edit.text()
        spec.ylabel = self.ylabel_edit.text()
        spec.zlabel = self.zlabel_edit.text()
        spec.theme = self.theme_combo.currentData() or "light"
        spec.figsize = (float(self.figwidth_spin.value()), float(self.figheight_spin.value()))
        spec.dpi = int(self.dpi_spin.value())
        spec.grid = self.grid_check.isChecked()
        spec.legend = self.legend_check.isChecked()
        spec.legend_loc = self.legend_loc_combo.currentData() or "best"
        spec.equal_aspect = self.aspect_check.isChecked()
        spec.xscale = self.xscale_combo.currentData() or "linear"
        spec.yscale = self.yscale_combo.currentData() or "linear"
        spec.zscale = self.zscale_combo.currentData() or "linear"
        spec.xlim = _parse_limits(self.xlim_edit.text())
        spec.ylim = _parse_limits(self.ylim_edit.text())
        spec.zlim = _parse_limits(self.zlim_edit.text())
        spec.colorbar = self.colorbar_check.isChecked()
        spec.view_elev = float(self.elev_spin.value())
        spec.view_azim = float(self.azim_spin.value())
        spec.series = list(self._series)
        return spec

    def render(self) -> None:
        self._timer.stop()
        self.spec = self._collect_spec()
        datasets = self.store.frames()
        # 始终复用同一个 Figure 重绘：不重建 Qt 画布控件，避免界面抖动
        try:
            figure, warnings = plotting.build_figure(
                self.spec, datasets, figure=self.canvas.figure
            )
        except Exception as exc:
            self.warning_label.setText(f"绘图失败：{exc}")
            return
        self._apply_canvas_theme(figure)
        self.canvas.refresh()
        if not self.spec.series:
            self.warning_label.setText("还没有数据序列：请在左侧「数据序列」中点「添加」。")
            return
        messages = list(warnings)
        if not messages:
            total = sum(len(s.y_columns()) for s in self.spec.series) or len(self.spec.series)
            messages.append(f"已绘制 {len(self.spec.series)} 条序列（共 {total} 组数据）")
        self.warning_label.setText("　|　".join(messages[:3]))

    def _apply_canvas_theme(self, figure) -> None:
        """让画布底色与界面主题一致（``light`` / ``dark_background``）。"""
        theme = self.spec.theme
        if theme not in ("light", "dark_background", "dark"):
            return
        background = self._palette.get("canvas_bg")
        if not background:
            return
        figure.patch.set_facecolor(background)
        for axes in figure.axes:
            axes.set_facecolor(background)

    # ================================================================== #
    # 投影 / 默认图
    # ================================================================== #
    def _on_projection_changed(self, *_: Any) -> None:
        if self._loading:
            return
        self._loading = True
        try:
            kinds = self._kinds()
            current = self.kind_combo.currentData()
            self.kind_combo.clear()
            for key, text in kinds.items():
                self.kind_combo.addItem(text, key)
            if current in kinds:
                self.kind_combo.setCurrentIndex(self.kind_combo.findData(current))
            fallback = "line3d" if self._is_3d() else "line"
            for series in self._series:
                if series.kind not in kinds:
                    series.kind = fallback
            self._update_field_visibility()
        finally:
            self._loading = False
        self._refresh_series_list(select=self._current)

    def new_default_plot(self, dataset_name: str) -> None:
        """为一个数据集生成开箱即用的默认图。"""
        dataset = self.store.get(dataset_name)
        if dataset is None:
            return
        projection = self.projection_combo.currentData() or "2d"
        spec = plotting.default_spec(dataset_name, dataset.frame, projection)
        self._series = list(spec.series) or []
        self._loading = True
        try:
            self.title_edit.setText(spec.title)
            self.xlabel_edit.setText(spec.xlabel)
            self.ylabel_edit.setText(spec.ylabel)
            self.zlabel_edit.setText(spec.zlabel)
        finally:
            self._loading = False
        self._current = 0 if self._series else -1
        self._refresh_series_list(select=self._current if self._current >= 0 else None)
        if self._series:
            self._on_series_selected(0)
        else:
            self.schedule_render()

    # ================================================================== #
    # 导出
    # ================================================================== #
    def copy_image(self) -> None:
        if self.canvas.copy_to_clipboard():
            self.statusMessage.emit("图片已复制到剪贴板")
        else:
            self.statusMessage.emit("复制失败")

    def export_image(self) -> None:
        base = (self.spec.title or "figure").strip()
        safe = "".join(ch for ch in base if ch.isalnum() or ch in "-_ ") or "figure"
        path, selected = QFileDialog.getSaveFileName(
            self,
            "导出图片",
            f"{safe}.svg",
            "SVG 矢量图 (*.svg);;PNG 位图 (*.png);;PDF 文档 (*.pdf);;PGF/TikZ (*.pgf);;JPEG 图片 (*.jpg)",
        )
        if not path:
            return
        try:
            # 单独渲染一张**按「画布尺寸」设置**的 Figure（屏幕上的图会跟随窗口大小，
            # 导出时则应严格按用户设置的尺寸/DPI），而不是直接存画布上那张
            spec = self._collect_spec()
            figure, _warnings = plotting.build_figure(spec, self.store.frames())
            self._apply_canvas_theme(figure)
            kwargs: dict[str, Any] = {}
            if self._is_3d():
                kwargs["bbox_inches"] = "tight"
            plotting.save_figure(figure, path, **kwargs)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.statusMessage.emit(f"已导出图片：{path}")

    def export_source(self) -> None:
        self.spec = self._collect_spec()
        csv_paths = {
            name: dataset.path
            for name in self.spec.used_datasets()
            if (dataset := self.store.get(name)) is not None and dataset.path
        }
        dialog = ExportSourceDialog(
            self, spec=self.spec, datasets=self.store.frames(), csv_paths=csv_paths
        )
        dialog.exec()

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存绘图配置", "plot.json", "JSON 配置 (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._collect_spec().to_dict(), handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.statusMessage.emit(f"配置已保存到 {path}")

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "载入绘图配置", "", "JSON 配置 (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            spec = PlotSpec.from_dict(data)
        except Exception as exc:
            QMessageBox.warning(self, "载入失败", str(exc))
            return
        self.apply_spec(spec)
        self.statusMessage.emit(f"已载入配置：{path}")

    def apply_spec(self, spec: PlotSpec) -> None:
        """把一份 PlotSpec 应用到界面。"""
        self._loading = True
        try:
            index = self.projection_combo.findData(spec.projection)
            self.projection_combo.setCurrentIndex(max(index, 0))
            self._on_projection_changed()
            self.title_edit.setText(spec.title)
            self.xlabel_edit.setText(spec.xlabel)
            self.ylabel_edit.setText(spec.ylabel)
            self.zlabel_edit.setText(spec.zlabel)
            self.theme_combo.setCurrentIndex(max(self.theme_combo.findData(spec.theme), 0))
            self.figwidth_spin.setValue(float(spec.figsize[0]))
            self.figheight_spin.setValue(float(spec.figsize[1]))
            self.dpi_spin.setValue(int(spec.dpi))
            self.grid_check.setChecked(spec.grid)
            self.legend_check.setChecked(spec.legend)
            self.legend_loc_combo.setCurrentIndex(max(self.legend_loc_combo.findData(spec.legend_loc), 0))
            self.aspect_check.setChecked(spec.equal_aspect)
            self.xscale_combo.setCurrentIndex(max(self.xscale_combo.findData(spec.xscale), 0))
            self.yscale_combo.setCurrentIndex(max(self.yscale_combo.findData(spec.yscale), 0))
            self.zscale_combo.setCurrentIndex(max(self.zscale_combo.findData(spec.zscale), 0))
            self.xlim_edit.setText(_format_limits(spec.xlim))
            self.ylim_edit.setText(_format_limits(spec.ylim))
            self.zlim_edit.setText(_format_limits(spec.zlim))
            self.colorbar_check.setChecked(spec.colorbar)
            self.elev_spin.setValue(float(spec.view_elev))
            self.azim_spin.setValue(float(spec.view_azim))
            self._series = list(spec.series)
        finally:
            self._loading = False
        self._current = 0 if self._series else -1
        self._refresh_series_list(select=self._current if self._current >= 0 else None)
        if self._series:
            self._on_series_selected(0)
        else:
            self.render()

    def show_syntax_help(self) -> None:
        text = (
            "图表中的标题、轴标签都支持 LaTeX 数学公式，用 $ 包裹即可。\n"
            "底层使用 matplotlib 的 mathtext，支持常见数学命令。\n\n"
            "示例：\n"
            "    $y = e^{-x^2}$\n"
            "    $\\frac{\\partial u}{\\partial t}$\n"
            "    $\\sum_{i=1}^{n} x_i^2$\n"
            "    $\\alpha, \\beta, \\lambda, \\sigma, \\mu$\n"
            "    $\\int_0^\\infty e^{-x^2}\\,dx = \\frac{\\sqrt{\\pi}}{2}$\n"
            "    $\\left\\|A\\right\\|_F = \\sqrt{\\sum a_{ij}^2}$\n\n"
            "注意：\n"
            "  • 反斜杠在 Python 字符串里要写成 \\\\，在界面输入框里直接写 \\ 即可\n"
            "  • 中文直接写在 $ 外面\n"
            "  • 矩阵环境（\\begin{pmatrix}）只支持在「矩阵」页的结果区使用\n\n"
            "常用转义：\\alpha \\beta \\gamma \\theta \\lambda \\mu \\pi \\sigma \\omega\n"
            "          \\times \\cdot \\pm \\leq \\geq \\neq \\approx \\infty\n"
            "          \\frac{}{} \\sqrt{} \\sum \\int \\partial \\nabla"
        )
        TextInfoDialog(self, title="公式语法说明", text=text).exec()


def _parse_limits(text: str) -> tuple[float, float] | None:
    """解析 ``"0,10"`` 形式的上限下限；无法解析时返回 ``None``（自动）。"""
    text = (text or "").strip().replace("，", ",")
    if not text:
        return None
    parts = [p for p in text.replace("~", ",").replace(":", ",").split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        low, high = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if low == high:
        return None
    return (low, high)


def _format_limits(limits: Any) -> str:
    if not limits:
        return ""
    try:
        low, high = limits
        return f"{low:g},{high:g}"
    except Exception:
        return ""
