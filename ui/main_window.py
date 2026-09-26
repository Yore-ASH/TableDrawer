"""主窗口：菜单、工具栏、数据集侧栏与三个功能页。"""

from __future__ import annotations

import os
import sys
from typing import Any

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core import __version__
from core.dataset import DataSetStore, load_csv
from core.latex import __name__ as _latex_pkg  # noqa: F401  （确保引擎可导入）
from core.mplsetup import FONT_PRESETS, apply_font_preset, font_config

from . import theme as theme_module
from .data_tab import DataTab
from .dialogs import TextInfoDialog
from .matrix_tab import MatrixTab
from .plot_tab import PlotTab
from .widgets import latex_pixmaps

__all__ = ["MainWindow"]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(_PROJECT_ROOT, "samples")


class MainWindow(QMainWindow):
    """应用主窗口。"""

    def __init__(self, parent: QWidget | None = None, *, theme: str = "dark"):
        super().__init__(parent)
        self.store = DataSetStore()
        self._theme_name = theme
        self._palette = theme_module.palette(theme)

        self.setWindowTitle("TableDrawer · 图表绘制与矩阵计算工作台")
        self.resize(1560, 980)
        self.setAcceptDrops(True)
        self.setDockNestingEnabled(False)

        self.data_tab = DataTab(self.store)
        self.plot_tab = PlotTab(self.store)
        self.matrix_tab = MatrixTab(self.store)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.data_tab, "数据")
        self.tabs.addTab(self.plot_tab, "绘图")
        self.tabs.addTab(self.matrix_tab, "矩阵")
        self.setCentralWidget(self.tabs)

        self._build_dock()
        self._build_actions()
        self._build_statusbar()
        self._connect()

        self.apply_theme(theme)
        self._font_preset = font_config()["preset"]
        for preset_key, action in self._font_actions.items():
            action.setChecked(preset_key == self._font_preset)
        self.refresh_dataset_dock()
        self.statusBar().showMessage("就绪。可从「文件 → 载入示例数据」开始，或直接拖入 CSV 文件。")

    # ================================================================== #
    # 构建
    # ================================================================== #
    def _build_dock(self) -> None:
        self.dataset_list = QListWidget()
        self.dataset_list.itemDoubleClicked.connect(self._on_dock_activated)

        import_btn = QPushButton("导入 CSV…")
        import_btn.clicked.connect(lambda: self.data_tab.open_csv_dialog())
        manual_btn = QPushButton("手动添加…")
        manual_btn.clicked.connect(self.data_tab.open_manual_dialog)
        samples_btn = QPushButton("载入示例数据")
        samples_btn.clicked.connect(self.load_samples)
        plot_btn = QPushButton("绘制当前数据集")
        plot_btn.setProperty("accent", "true")
        plot_btn.clicked.connect(self._plot_current)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(QLabel("数据集"))
        layout.addWidget(self.dataset_list, 1)
        layout.addWidget(import_btn)
        layout.addWidget(manual_btn)
        layout.addWidget(samples_btn)
        layout.addWidget(plot_btn)

        dock = QDockWidget("数据集", self)
        dock.setObjectName("datasetDock")
        dock.setWidget(container)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setMinimumWidth(230)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.dataset_dock = dock

    def _build_actions(self) -> None:
        menubar = self.menuBar()

        # ---- 文件 -------------------------------------------------------- #
        file_menu = menubar.addMenu("文件(&F)")
        self.act_import = QAction("导入 CSV…", self)
        self.act_import.setShortcut(QKeySequence.Open)
        self.act_import.triggered.connect(lambda: self.data_tab.open_csv_dialog())
        self.act_manual = QAction("手动添加数据组…", self)
        self.act_manual.setShortcut("Ctrl+N")
        self.act_manual.triggered.connect(self.data_tab.open_manual_dialog)
        self.act_samples = QAction("载入示例数据", self)
        self.act_samples.triggered.connect(self.load_samples)

        self.act_export_image = QAction("导出图表图片…", self)
        self.act_export_image.setShortcut("Ctrl+E")
        self.act_export_image.triggered.connect(self.plot_tab.export_image)
        self.act_export_source = QAction("导出 Python 源码…", self)
        self.act_export_source.setShortcut("Ctrl+Shift+E")
        self.act_export_source.triggered.connect(self.plot_tab.export_source)
        self.act_save_cfg = QAction("保存绘图配置…", self)
        self.act_save_cfg.triggered.connect(self.plot_tab.save_config)
        self.act_load_cfg = QAction("载入绘图配置…", self)
        self.act_load_cfg.triggered.connect(self.plot_tab.load_config)

        self.act_quit = QAction("退出", self)
        self.act_quit.setShortcut(QKeySequence.Quit)
        self.act_quit.triggered.connect(self.close)

        for action in (
            self.act_import, self.act_manual, self.act_samples, None,
            self.act_export_image, self.act_export_source, None,
            self.act_save_cfg, self.act_load_cfg, None, self.act_quit,
        ):
            if action is None:
                file_menu.addSeparator()
            else:
                file_menu.addAction(action)

        # ---- 数据 -------------------------------------------------------- #
        data_menu = menubar.addMenu("数据(&D)")
        act_expr = QAction("表达式生成数据…", self)
        act_expr.triggered.connect(self.data_tab.open_expression_dialog)
        act_summary = QAction("统计摘要（LaTeX）", self)
        act_summary.triggered.connect(self.data_tab.show_summary)
        act_export_csv = QAction("导出当前数据集为 CSV…", self)
        act_export_csv.triggered.connect(self.data_tab.export_csv)
        act_copy = QAction("复制表格到剪贴板", self)
        act_copy.triggered.connect(self.data_tab.copy_table)
        act_remove = QAction("删除当前数据集", self)
        act_remove.triggered.connect(self.data_tab.remove_dataset)
        for action in (act_expr, act_summary, None, act_export_csv, act_copy, None, act_remove):
            if action is None:
                data_menu.addSeparator()
            else:
                data_menu.addAction(action)

        # ---- 视图 -------------------------------------------------------- #
        view_menu = menubar.addMenu("视图(&V)")
        self.act_theme_dark = QAction("深色主题", self, checkable=True)
        self.act_theme_light = QAction("浅色主题", self, checkable=True)
        self.act_theme_dark.triggered.connect(lambda: self.apply_theme("dark"))
        self.act_theme_light.triggered.connect(lambda: self.apply_theme("light"))
        view_menu.addAction(self.act_theme_dark)
        view_menu.addAction(self.act_theme_light)
        view_menu.addSeparator()

        # 字体预设（影响界面、图表与 LaTeX 渲染）
        font_menu = view_menu.addMenu("字体")
        self._font_actions: dict[str, QAction] = {}
        for key, spec in FONT_PRESETS.items():
            action = QAction(spec["name"], self, checkable=True)
            action.triggered.connect(lambda _=False, k=key: self.set_font_preset(k))
            font_menu.addAction(action)
            self._font_actions[key] = action
        view_menu.addSeparator()
        for index, name in enumerate(("数据", "绘图", "矩阵")):
            action = QAction(f"{name}页", self)
            action.setShortcut(f"Ctrl+{index + 1}")
            action.triggered.connect(lambda _=False, i=index: self.tabs.setCurrentIndex(i))
            view_menu.addAction(action)
        view_menu.addSeparator()
        act_reset = QAction("重置侧栏布局", self)
        act_reset.triggered.connect(lambda: self.dataset_dock.setVisible(True))
        view_menu.addAction(act_reset)

        # ---- 帮助 -------------------------------------------------------- #
        help_menu = menubar.addMenu("帮助(&H)")
        act_help = QAction("使用说明", self)
        act_help.triggered.connect(self.show_help)
        act_syntax = QAction("公式语法", self)
        act_syntax.triggered.connect(self.plot_tab.show_syntax_help)
        act_about = QAction("关于 TableDrawer", self)
        act_about.triggered.connect(self.show_about)
        for action in (act_help, act_syntax, None, act_about):
            if action is None:
                help_menu.addSeparator()
            else:
                help_menu.addAction(action)

        # ---- 工具栏 ------------------------------------------------------ #
        toolbar = self.addToolBar("主工具栏")
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(False)
        toolbar.addAction(self.act_import)
        toolbar.addAction(self.act_manual)
        toolbar.addAction(self.act_samples)
        toolbar.addSeparator()
        toolbar.addAction(self.act_export_image)
        toolbar.addAction(self.act_export_source)
        toolbar.addSeparator()
        act_render = QAction("重绘当前图", self)
        act_render.setShortcut("F5")
        act_render.triggered.connect(self.plot_tab.render)
        toolbar.addAction(act_render)
        toolbar.addSeparator()
        act_toggle_theme = QAction("切换主题", self)
        act_toggle_theme.triggered.connect(
            lambda: self.apply_theme("light" if self._theme_name == "dark" else "dark")
        )
        toolbar.addAction(act_toggle_theme)

    def _build_statusbar(self) -> None:
        self.status_count = QLabel("")
        self.statusBar().addPermanentWidget(self.status_count)

    def _connect(self) -> None:
        self.data_tab.statusMessage.connect(self.statusBar().showMessage)
        self.plot_tab.statusMessage.connect(self.statusBar().showMessage)
        self.matrix_tab.statusMessage.connect(self.statusBar().showMessage)
        self.data_tab.datasetChanged.connect(self._on_dataset_changed)
        self.data_tab.plotRequested.connect(self._plot_dataset)
        self.store.subscribe(self.refresh_dataset_dock)

    # ================================================================== #
    # 主题
    # ================================================================== #
    def apply_theme(self, name: str) -> None:
        palette = theme_module.palette(name)
        self._theme_name = palette.get("name", name)
        self._palette = palette

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme_module.stylesheet(palette, theme_module.ui_fonts()))

        self.data_tab.set_palette(palette)
        self.plot_tab.set_palette(palette)
        self.matrix_tab.set_palette(palette)

        self.act_theme_dark.setChecked(self._theme_name == "dark")
        self.act_theme_light.setChecked(self._theme_name == "light")
        self.plot_tab.schedule_render()
        self.statusBar().showMessage(f"已切换到{'深色' if self._theme_name == 'dark' else '浅色'}主题")

    def set_font_preset(self, key: str) -> None:
        """切换字体预设（界面 + 图表 + LaTeX 渲染同时生效）。"""
        if key not in FONT_PRESETS:
            return
        apply_font_preset(key)
        # 旧的 LaTeX 位图是用旧字体排版出来的，必须整体丢弃重排
        latex_pixmaps().clear()
        self._font_preset = key
        self.apply_theme(self._theme_name)
        self.matrix_tab._refresh_latex_buttons()
        self.matrix_tab._rerender()
        for preset_key, action in self._font_actions.items():
            action.setChecked(preset_key == key)
        config = font_config()
        self.statusBar().showMessage(
            f"字体已切换：{FONT_PRESETS[key]['name']}"
            f"（西文 {config['latin'][0]}／中文 {config['cjk'][0]}／等宽 {config['mono'][0]}）"
        )

    # ================================================================== #
    # 数据集
    # ================================================================== #
    def refresh_dataset_dock(self) -> None:
        current = self.data_tab.current_name()
        self.dataset_list.clear()
        for dataset in self.store.items():
            item = QListWidgetItem(f"{dataset.name}\n{dataset.shape_text}　·　{_source_label(dataset.source)}")
            item.setData(Qt.UserRole, dataset.name)
            self.dataset_list.addItem(item)
            if dataset.name == current:
                self.dataset_list.setCurrentItem(item)
        self.status_count.setText(f"数据集：{len(self.store)} 个")

    def _on_dataset_changed(self, name: str) -> None:
        self.refresh_dataset_dock()
        if name:
            self.plot_tab.refresh_datasets()

    def _on_dock_activated(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.UserRole)
        self.tabs.setCurrentIndex(0)
        self.data_tab.select_dataset(str(name))

    def _plot_current(self) -> None:
        self._plot_dataset(self.data_tab.current_name())

    def _plot_dataset(self, name: str) -> None:
        if not name or name not in self.store:
            QMessageBox.information(self, "没有数据集", "请先导入或添加一个数据集。")
            return
        self.tabs.setCurrentIndex(1)
        self.plot_tab.new_default_plot(name)
        self.plot_tab.render()
        self.statusBar().showMessage(f"已为「{name}」生成默认图表")

    def load_samples(self) -> None:
        """载入 ``samples/`` 目录下的示例 CSV。"""
        files: list[str] = []
        if os.path.isdir(SAMPLES_DIR):
            files = sorted(
                os.path.join(SAMPLES_DIR, name)
                for name in os.listdir(SAMPLES_DIR)
                if name.lower().endswith(".csv")
            )
        if not files:
            dataset = _builtin_sample()
            self.store.add(dataset)
            self.data_tab.select_dataset(dataset.name)
            self.statusBar().showMessage("已载入内置示例数据（samples 目录为空）")
            return

        added: list[str] = []
        problems: list[str] = []
        for path in files:
            try:
                dataset, warnings = load_csv(path)
                self.store.add(dataset, replace=False)
                added.append(dataset.name)
                problems.extend(f"{dataset.name}：{w}" for w in warnings)
            except Exception as exc:
                problems.append(f"{os.path.basename(path)}：{exc}")
        if added:
            self.data_tab.select_dataset(added[0])
            self.statusBar().showMessage(f"已载入 {len(added)} 个示例数据集：{', '.join(added)}")
        if problems:
            QMessageBox.information(self, "部分文件有提示", "\n".join(problems[:12]))

    def _import_paths(self, paths: list[str]) -> None:
        added: list[str] = []
        problems: list[str] = []
        for path in paths:
            try:
                dataset, warnings = load_csv(path)
                self.store.add(dataset)
                added.append(dataset.name)
                problems.extend(f"{dataset.name}：{w}" for w in warnings)
            except Exception as exc:
                problems.append(f"{os.path.basename(path)}：{exc}")
        if added:
            self.data_tab.select_dataset(added[0])
            self.statusBar().showMessage(f"已导入 {len(added)} 个数据集")
        if problems:
            QMessageBox.information(self, "导入提示", "\n".join(problems[:12]))

    # ================================================================== #
    # 拖放
    # ================================================================== #
    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".csv", ".tsv", ".txt", ".dat")):
                    event.acceptProposedAction()
                    return
        super().dragEnterEvent(event)

    def dropEvent(self, event):  # noqa: N802
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if url.toLocalFile().lower().endswith((".csv", ".tsv", ".txt", ".dat"))
        ]
        if paths:
            self._import_paths(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    # ================================================================== #
    # 帮助
    # ================================================================== #
    def show_help(self) -> None:
        TextInfoDialog(self, title="使用说明", text=HELP_TEXT, width=840, height=700).exec()

    def show_about(self) -> None:
        import matplotlib

        config = font_config()
        preset_name = FONT_PRESETS.get(config["preset"], {}).get("name", config["preset"])
        text = (
            f"TableDrawer  {__version__}\n"
            "图表绘制与矩阵计算工作台\n\n"
            "技术栈\n"
            f"  Python      {sys.version.split()[0]}\n"
            f"  PySide6     {_version('PySide6')}\n"
            f"  matplotlib  {matplotlib.__version__}\n"
            f"  pandas      {pd.__version__}\n"
            f"  NumPy       {np.__version__}\n\n"
            "功能\n"
            "  · 读取 CSV / 手动录入 / 表达式生成数据\n"
            "  · 二维与三维图表：折线、散点、柱形、条形、面积、误差棒、\n"
            "    直方图、箱线、饼图、等高线、热力图、向量图（quiver）、\n"
            "    曲面、线框、三维柱形、三维向量场、三角曲面\n"
            "  · 导出 SVG / PNG / PDF / PGF-TikZ 与可独立运行的 Python 源码\n"
            "  · 矩阵输入支持放大/全屏编辑、表格自动扩展、有效区域识别、\n"
            "    空白单元格填充策略，并实时预览识别结果\n"
            "  · 矩阵与向量组计算：转置、逆、行列式、特征值、秩、分解、范数等\n"
            "  · 界面符号与数学结果全部以 LaTeX 排版渲染\n"
            "    （自研迷你 TeX 引擎，支持矩阵环境与中英文混排）\n\n"
            "字体\n"
            f"  预设     {preset_name}\n"
            f"  西文     {config['latin'][0] if config['latin'] else '-'}\n"
            f"  中文     {config['cjk'][0] if config['cjk'] else '-'}\n"
            f"  等宽     {config['mono'][0] if config['mono'] else '-'}\n"
            f"  数学     {config['mathtext']}（mathtext.fontset）\n"
            "  可在「视图 → 字体」中切换\n"
        )
        TextInfoDialog(self, title="关于 TableDrawer", text=text, width=700, height=660).exec()


def _version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "?")
    except Exception:
        return "?"


def _source_label(source: str) -> str:
    return {"csv": "CSV", "manual": "手动", "expression": "表达式", "sample": "示例"}.get(source, source)


def _builtin_sample():
    """``samples`` 目录缺失时的内置示例数据。"""
    from core.dataset import DataSet

    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 10.0, 200)
    frame = pd.DataFrame(
        {
            "时间 t": np.round(x, 3),
            "信号 u": np.round(np.sin(x) * np.exp(-x / 6.0), 6),
            "信号 v": np.round(np.cos(x) * np.exp(-x / 8.0), 6),
        }
    )
    return DataSet(name="衰减振荡（内置）", frame=frame, source="sample")


HELP_TEXT = """\
TableDrawer 使用说明
====================

一、准备数据
------------
1. 「文件 → 导入 CSV…」或直接把手上的 .csv 文件拖进窗口。
   导入对话框会自动探测编码（UTF-8 / GBK 等）、分隔符与表头，
   并在右侧给出预览，可随时手动调整任何一项。
2. 「文件 → 手动添加数据组…」用于手工敲数据：
   · 「表格录入」页可直接双击编辑，也可以从 Excel 复制后 Ctrl+V 粘贴。
   · 「文本粘贴」页支持整段粘贴 CSV / TSV 文本，自动解析成表格。
3. 「数据 → 表达式生成数据…」用 NumPy 表达式派生新列或从零生成数据集，
   例如新建 x = linspace(0, 2*pi, n)、y = sin(x)、z = cos(x)*exp(-x/5)。
4. 「文件 → 载入示例数据」会一次性载入 samples 目录下的示例 CSV。

二、绘图
--------
1. 在左侧「数据集」侧栏点「绘制当前数据集」，会自动生成一张默认图。
2. 「绘图」页左侧是序列列表与序列设置：
   · 「添加 / 复制 / 删除 / ↑↓」管理多条序列，可把多个数据集画在同一张图上。
   · 图表类型涵盖折线、散点、柱形、条形、阶梯、面积、误差棒、直方图、
     箱线、饼图、等高线、热力图，以及向量图（需要 X、Y 为起点，U、V 为分量）。
   · 切换「投影」到三维后可使用三维折线、散点、曲面、线框、三维柱形、
     三维向量场与三角曲面（曲面类需要 X、Y、Z 三个网格列）。
   · Y 列可以填多列（英文逗号分隔），一次画多条。
3. 「坐标轴与画布」里可设置标题、轴标签、范围、刻度、图例、主题与画布尺寸；
   标题与轴标签都支持 $LaTeX$ 公式，例如 $y=e^{-x^2}$。
4. 画布下方的状态行会提示缺列等警告；右上角工具条支持缩放、平移与保存。

三、导出
--------
· 「导出图片…」：SVG（矢量，推荐）、PNG、PDF、PGF/TikZ、JPEG。
· 「导出源码…」：生成一段**可独立运行**的 matplotlib Python 源码，
  数据可以内联进源码（自包含）或改为从 CSV 读取；支持中文注释与
  make_figure() 包装形式，可直接复制或保存为 .py 运行。
· 「保存配置 / 载入配置」把当前绘图设置存成 JSON，方便复用。

四、矩阵与向量组计算
--------------------
1. 在「矩阵」页输入矩阵 A（以及二元运算需要的矩阵 B、标量 k）。
   · 单元格支持从 Excel/文本粘贴；也用「单位阵 / 零阵 / 全 1 / 随机 / 转置」快速填充。
   · 「从数据集列导入…」可以把数据集的若干数值列直接变成矩阵。
2. 中间按钮区按类别列出全部运算：生成矩阵、基本运算、矩阵乘法、
   行列式与秩、逆与广义逆、特征值与特征向量、矩阵分解、范数与条件数、
   线性方程组、向量组。可用上方搜索框快速过滤。
3. 结果在右侧用 LaTeX 排版渲染，支持滚动与缩放，可复制图片、
   复制 LaTeX 源码、导出 SVG 或导出文本结果。
4. 「一键常用量」一次给出行列式、秩、迹、逆、特征值、范数；
   「一键向量组分析」给出向量组的秩、相关性、施密特正交化与范数。

五、小技巧
----------
· 快捷键：Ctrl+O 导入 CSV、Ctrl+N 手动添加、Ctrl+E 导出图片、
  Ctrl+Shift+E 导出源码、F5 重绘、Ctrl+1/2/3 切换页签。
· 表格与矩阵单元格都支持 Ctrl+V 直接粘贴二维数据。
· 统计摘要会把描述性统计与相关系数矩阵用 LaTeX 表格渲染出来。
· 矩阵太大不好填？点「放大编辑…」在全屏窗口里填，表格会自动加行加列，
  填完自动识别有效区域、裁掉右下空白，空白格子可以选填充方式。
· 界面上的数学符号（Aᵀ、A⁻¹、λᵢ、‖A‖）都是 LaTeX 排版出来的，不依赖系统字体。
· 「视图 → 字体」可以切换字体预设，界面、图表与公式会一起变。
· 标题与轴标签支持中文和公式写在同一个字符串里，例如
  「温度 $T(t)$ / ℃」，程序会自动拆开分别渲染。

六、用 manim 渲染计算过程
--------------------------
矩阵页的「用 manim 渲染计算过程…」可以把运算过程做成动画，6 种场景：
高斯消元、转置、求逆、行列式展开、矩阵乘法、特征值。
对话框有三个页签：
· 分镜预览 —— 用本程序的 LaTeX 引擎展示动画会依次播放的每一步，
  **不用安装 manim 也能看到动画效果**；
· manim 源码 —— 完整可运行的 manim 场景文件，可复制或另存为 .py；
· 渲染视频 —— 装了 manim 就能一键渲染成 MP4（后台执行，实时显示日志）。
生成的脚本会自行探测 LaTeX：有就用 Matrix + MathTex，没有就降级成
Text 拼的等价格子，因此两种环境都能渲染。
"""
