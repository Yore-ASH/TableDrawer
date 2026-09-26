"""矩阵页：矩阵输入、向量组计算、LaTeX 结果展示。"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core import matrixops
from core.dataset import DataSetStore, numeric_columns
from core.latex import LatexDocument, TextBlock, layout_document
from core.numberfmt import matrix_to_text
from core.spec import ResultItem, ResultSheet

from .dialogs import TextInfoDialog
from .widgets import FILL_POLICIES, LatexButton, LatexView, MatrixEditor

__all__ = ["MatrixTab"]

#: 运算按钮的 LaTeX 排版字号（点）
_BUTTON_FONT_SIZE = 10.5

#: 运算按钮的 LaTeX 文案：``运算 key -> 含 $...$ 的标签``。
#: 界面上的这些符号（Aᵀ、A⁻¹、λᵢ、‖A‖ …）由 :class:`ui.widgets.LatexButton`
#: 交给自研 LaTeX 引擎排版成矢量位图，因此不依赖系统字体里是否含有这些字符。
_LATEX_LABELS: dict[str, str] = {
    # 生成矩阵
    "identity": r"单位矩阵 $I_n$",
    "zeros": r"零矩阵 $O$",
    "ones": r"全 $1$ 矩阵 $J$",
    "random": r"随机矩阵",
    "hilbert": r"希尔伯特矩阵 $H$",
    "magic": r"幻方矩阵",
    # 基本运算
    "transpose": r"转置 $A^{\mathrm{T}}$",
    "ctranspose": r"共轭转置 $A^{\mathrm{H}}$",
    "trace": r"迹 $\operatorname{tr}(A)$",
    "scalar_multiply": r"数乘 $kA$",
    "add": r"加法 $A+B$",
    "subtract": r"减法 $A-B$",
    "hadamard": r"哈达玛积 $A\circ B$",
    "power": r"矩阵幂 $A^{k}$",
    "elementwise_power": r"逐元素幂 $A^{\circ k}$",
    "negate": r"取负 $-A$",
    "abs": r"逐元素绝对值 $|A|$",
    "round_matrix": r"四舍五入到 $k$ 位小数",
    # 矩阵乘法
    "matmul": r"矩阵乘法 $A\,B$",
    "kron": r"克罗内克积 $A\otimes B$",
    # 行列式与秩
    "determinant": r"行列式 $\det(A)$",
    "rank": r"秩 $\operatorname{rank}(A)$",
    "rref": r"行简化阶梯形 $\mathrm{RREF}$",
    "charpoly": r"特征多项式 $p(\lambda)$",
    "adjugate": r"伴随矩阵 $\operatorname{adj}(A)$",
    "minor": r"$k$ 阶子式",
    # 逆与广义逆
    "inverse": r"逆矩阵 $A^{-1}$",
    "pinv": r"广义逆 $A^{+}$",
    "inverse_check": r"验证 $A\,A^{-1}$",
    # 特征值
    "eigen": r"特征值与特征向量",
    "eigenvalues": r"特征值 $\lambda_i$",
    "eigenvectors": r"特征向量 $v_i$",
    "spectral_radius": r"谱半径 $\rho(A)$",
    "singular_values": r"奇异值 $\sigma_i$",
    # 矩阵分解
    "lu": r"$LU$ 分解",
    "qr": r"$QR$ 分解",
    "svd": r"奇异值分解 $\mathrm{SVD}$",
    "cholesky": r"Cholesky 分解",
    "eigendecomposition": r"特征分解 $A=PDP^{-1}$",
    # 范数与条件数
    "norms": r"各种范数 $\|A\|$",
    "condition": r"条件数 $\operatorname{cond}(A)$",
    # 线性方程组
    "solve": r"解 $Ax=b$",
    "solve_lstsq": r"最小二乘解 $Ax\approx b$",
    "nullspace": r"零空间 $N(A)$",
    "colspace": r"列空间 $C(A)$",
    "rowspace": r"行空间 $R(A)$",
    # 向量组
    "vector_rank": r"向量组的秩",
    "linear_independence": r"线性相关性判定",
    "gram_schmidt": r"施密特正交化",
    "dot": r"内积 $u\cdot v$",
    "cross": r"叉积 $u\times v$",
    "vector_norm": r"向量范数 $\|v\|$",
    "angle": r"夹角 $\theta$",
    "projection": r"投影 $\operatorname{proj}_u v$",
    "span_contains": r"张成空间判定",
} 


class MatrixTab(QWidget):
    """矩阵 / 向量组计算页。"""

    statusMessage = Signal(str)

    def __init__(self, store: DataSetStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self._palette: dict[str, str] = {}
        self._sheet = ResultSheet(title="矩阵与向量组计算结果")
        self._buttons: list[tuple[Any, QPushButton]] = []
        self._build_ui()
        self.store.subscribe(self._refresh_datasets)

    # ================================================================== #
    # 界面
    # ================================================================== #
    def _build_ui(self) -> None:
        # ---------------- 左：输入 ---------------- #
        # 矩阵是「可增删的一组」，A、B 只是默认的两个；C、D… 由「新增矩阵」添加
        self._editors: dict[str, MatrixEditor] = {}
        self._editor_order: list[str] = []
        self.editor_stack = QStackedWidget()

        self.editor_combo = QComboBox()
        self.editor_combo.setToolTip("选择当前要编辑的矩阵")
        self.editor_combo.currentIndexChanged.connect(self._on_editor_selected)
        add_matrix_btn = QPushButton("＋ 新增矩阵")
        add_matrix_btn.setProperty("accent", "true")
        add_matrix_btn.setToolTip("添加一个新的矩阵（C、D、E…），可以一起参与运算与 manim 动画")
        add_matrix_btn.clicked.connect(lambda: self.add_matrix())
        rename_matrix_btn = QPushButton("重命名")
        rename_matrix_btn.clicked.connect(self.rename_matrix)
        remove_matrix_btn = QPushButton("删除")
        remove_matrix_btn.setToolTip("删除当前矩阵（至少要保留一个）")
        remove_matrix_btn.clicked.connect(self.remove_matrix)

        editor_row = QHBoxLayout()
        editor_row.setSpacing(6)
        editor_row.addWidget(QLabel("编辑"))
        editor_row.addWidget(self.editor_combo, 1)
        editor_row.addWidget(add_matrix_btn)
        editor_row.addWidget(rename_matrix_btn)
        editor_row.addWidget(remove_matrix_btn)

        matrix_buttons_row = QHBoxLayout()
        matrix_buttons_row.setSpacing(6)
        import_btn = QPushButton("从剪贴板导入")
        import_btn.setToolTip("把剪贴板里的二维数据读入当前矩阵")
        import_btn.clicked.connect(lambda: self._import_clipboard(self.current_editor()))
        export_btn = QPushButton("复制")
        export_btn.setToolTip("把当前矩阵复制为制表符分隔的文本")
        export_btn.clicked.connect(lambda: self._export_clipboard(self.current_editor()))
        dataset_btn = QPushButton("数据集导入")
        dataset_btn.setToolTip("把数据集的若干数值列直接变成当前矩阵")
        dataset_btn.clicked.connect(self._import_from_dataset)
        matrix_buttons_row.addWidget(import_btn)
        matrix_buttons_row.addWidget(export_btn)
        matrix_buttons_row.addWidget(dataset_btn)
        matrix_buttons_row.addStretch(1)

        self.matrix_box = QGroupBox("矩阵")
        matrix_layout = QVBoxLayout(self.matrix_box)
        matrix_layout.addLayout(editor_row)
        matrix_layout.addWidget(self.editor_stack, 1)
        matrix_layout.addLayout(matrix_buttons_row)

        self.scalar_spin = QSpinBox()
        self.scalar_spin.setRange(-999, 999)
        self.scalar_spin.setValue(2)
        self.vector_check = QCheckBox("按向量组解释主矩阵的每一行")
        self.vector_check.setChecked(False)
        self.vector_check.setToolTip("勾选后，主矩阵的每一行被视为一个向量，用于向量组的秩、正交化等运算")
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(2, 14)
        self.precision_spin.setValue(6)
        self.precision_spin.setToolTip("结果中的有效数字位数")
        self.precision_spin.valueChanged.connect(self._rerender)
        self.fraction_check = QCheckBox("尽量显示为分数")
        self.fraction_check.setToolTip("把 0.333333 显示为 1/3")
        self.fraction_check.toggled.connect(self._rerender)

        # 运算用哪两个矩阵：主矩阵 = A，副矩阵 = B（二元运算 / 线性方程组右端项）
        self.main_combo = QComboBox()
        self.main_combo.setToolTip("作为运算里的 A")
        self.other_combo = QComboBox()
        self.other_combo.setToolTip("作为运算里的 B（二元运算、Ax=b 的右端项）")
        pick_row = QHBoxLayout()
        pick_row.setSpacing(6)
        pick_row.addWidget(QLabel("主矩阵"))
        pick_row.addWidget(self.main_combo, 1)
        pick_row.addWidget(QLabel("副矩阵"))
        pick_row.addWidget(self.other_combo, 1)

        param_form = QFormLayout()
        param_form.addRow("标量 k / 阶数 n", self.scalar_spin)
        param_form.addRow("小数位数", self.precision_spin)
        # 空白单元格的填充策略 + 自动裁剪有效区域（对 A、B 同时生效）
        self.fill_combo = QComboBox()
        for key, text in FILL_POLICIES.items():
            self.fill_combo.addItem(text, key)
        self.fill_combo.setToolTip(
            "矩阵里没填的格子怎么处理。\n"
            "「按行线性插值」会用同一行两侧已知值插值补全；\n"
            "「不填充」会在有空单元格时报错并指出位置。"
        )
        self.fill_combo.currentIndexChanged.connect(self._on_fill_policy_changed)
        self.trim_check = QCheckBox("自动识别有效区域")
        self.trim_check.setChecked(True)
        self.trim_check.setToolTip(
            "只按实际填写到的最右下单元格确定矩阵大小，\n"
            "右下角的整块空白会被裁掉——填一个 3×4 就得到 3×4 的矩阵。"
        )
        self.trim_check.toggled.connect(self._on_fill_policy_changed)

        param_form.addRow("空白单元格", self.fill_combo)
        param_form.addRow("", self.trim_check)
        param_form.addRow("", self.fraction_check)
        param_form.addRow("", self.vector_check)
        param_box = QGroupBox("运算参数")
        param_layout_inner = QVBoxLayout(param_box)
        param_layout_inner.addLayout(pick_row)
        param_layout_inner.addLayout(param_form)

        expand_btn = QPushButton("放大编辑当前矩阵…")
        expand_btn.setToolTip("在全屏/大窗口里填写矩阵，实时预览识别结果")
        expand_btn.clicked.connect(lambda: self._expand_editor(self.current_editor()))

        self.manim_button = QPushButton("用 manim 渲染计算过程…")
        self.manim_button.setToolTip(
            "把矩阵运算过程导出成 manim 动画：\n"
            "可预览分镜、生成可运行的 manim 源码，装了 manim 还能直接渲染成视频"
        )
        self.manim_button.clicked.connect(self.open_manim_dialog)

        param_layout = QVBoxLayout()
        param_layout.addWidget(param_box)
        param_layout.addWidget(expand_btn)
        param_layout.addWidget(self.manim_button)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(2, 2, 8, 2)
        left_layout.addWidget(self.matrix_box, 3)
        left_layout.addLayout(param_layout)
        left_layout.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_content)
        left_scroll.setMinimumWidth(330)

        # ---------------- 中：运算 ---------------- #
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("搜索运算…")
        self.filter_edit.textChanged.connect(self._apply_filter)

        quick_btn = QPushButton("一键常用量（行列式/秩/迹/逆/特征值）")
        quick_btn.setProperty("accent", "true")
        quick_btn.clicked.connect(self.run_common)
        vector_btn = QPushButton("一键向量组分析")
        vector_btn.clicked.connect(self.run_vector_analysis)
        clear_btn = QPushButton("清空结果")
        clear_btn.clicked.connect(self.clear_results)

        ops_layout = QVBoxLayout()
        ops_layout.setContentsMargins(2, 2, 8, 2)
        ops_layout.addWidget(self.filter_edit)
        ops_layout.addWidget(quick_btn)
        ops_layout.addWidget(vector_btn)

        seen_groups: list[str] = []
        for spec in matrixops.REGISTRY:
            if spec.group not in seen_groups:
                seen_groups.append(spec.group)
        for group in seen_groups:
            items = [s for s in matrixops.REGISTRY if s.group == group]
            box = QGroupBox(group)
            grid = QGridLayout(box)
            grid.setSpacing(4)
            grid.setContentsMargins(8, 6, 8, 8)
            for index, spec in enumerate(items):
                # 标签里的数学符号交给 LaTeX 引擎排版；渲染完成前先用纯文本占位
                button = LatexButton(
                    _LATEX_LABELS.get(spec.key, spec.label),
                    spec.label,
                    fontsize=_BUTTON_FONT_SIZE,
                    color=self._latex_color(),
                    tooltip=f"{spec.description}\n\n{spec.latex_hint}".strip() or spec.label,
                )
                # 单列铺满：按钮宽度一致、图标居中，不会出现右侧被裁掉的情况
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                button.clicked.connect(lambda _=False, key=spec.key: self.run_operation(key))
                grid.addWidget(button, index, 0)
                self._buttons.append((spec, button))
            grid.setColumnStretch(0, 1)
            ops_layout.addWidget(box)
        ops_layout.addWidget(clear_btn)
        ops_layout.addStretch(1)

        ops_content = QWidget()
        ops_content.setLayout(ops_layout)
        ops_scroll = QScrollArea()
        ops_scroll.setWidgetResizable(True)
        ops_scroll.setWidget(ops_content)
        # 关掉横向滚动条：内容宽度会跟随视口，按钮永远是「铺满面板 + 文字居中」，
        # 不会出现右侧被裁掉或需要左右拖的情况。
        ops_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # 单列布局下用「最宽标签的实测宽度」定面板最小宽度，保证每个按钮都完整可见
        ops_scroll.setMinimumWidth(self._ops_panel_width())
        self.ops_scroll = ops_scroll

        # ---------------- 右：结果 ---------------- #
        self.view = LatexView(self)
        self.view.set_message("输入矩阵后，点击左侧运算按钮查看 LaTeX 渲染的计算结果。")

        # 结果工具条用 QToolBar：窗口变窄时 Qt 会自动把放不下的按钮收进「»」溢出菜单，
        # 不会像手写按钮那样被硬生生裁掉。
        self.result_bar = QToolBar()
        self.result_bar.setIconSize(QSize(16, 16))
        self.result_bar.setMovable(False)
        self.result_bar.setStyleSheet(
            "QToolBar { background: transparent; border: none; padding: 0; spacing: 2px; }"
        )
        for text, slot, tip in (
            ("缩小", self.view.zoom_out, "缩小结果视图"),
            ("100%", self.view.reset_zoom, "恢复 100% 缩放"),
            ("放大", self.view.zoom_in, "放大结果视图"),
        ):
            action = QAction(text, self)
            action.setToolTip(tip)
            action.triggered.connect(slot)
            self.result_bar.addAction(action)
        self.result_bar.addSeparator()
        for text, slot, tip in (
            ("复制 LaTeX", self._copy_latex, "复制结果的 LaTeX 源码"),
            ("LaTeX 源码", self._show_latex_source, "查看原始 LaTeX 源码"),
            ("复制图片", self._copy_image, "把结果复制为图片"),
            ("导出 SVG", self._export_svg, "导出为矢量 SVG"),
            ("导出文本", self._export_text, "导出为纯文本结果"),
        ):
            action = QAction(text, self)
            action.setToolTip(tip)
            action.triggered.connect(slot)
            self.result_bar.addAction(action)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.result_bar)
        right_layout.addWidget(self.view, 1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_scroll)
        self.splitter.addWidget(ops_scroll)
        self.splitter.addWidget(right)
        # 禁止把面板拖成 0 宽：否则拖到边界时会「啪」地塌掉再弹回来，看起来像乱跳。
        # setChildrenCollapsible 只影响之后插入的面板，所以还要逐个 setCollapsible。
        self.splitter.setChildrenCollapsible(False)
        for index in range(self.splitter.count()):
            self.splitter.setCollapsible(index, False)
        left_scroll.setMinimumWidth(336)
        right.setMinimumWidth(280)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes([380, self._ops_panel_width() + 24, 760])

        layout = QVBoxLayout(self)
        layout.addWidget(self.splitter, 1)

        # 默认创建 A、B 两个矩阵：B 预填成全 1，方便直接解 Ax=b
        self._create_editor("A", rows=3, cols=3)
        self._create_editor("B", rows=3, cols=1)
        self._editors["B"].set_matrix(np.array([[1.0], [1.0], [1.0]]))
        self.editor_combo.setCurrentIndex(0)
        self._on_fill_policy_changed()

    # ================================================================== #
    # 矩阵集合管理
    # ================================================================== #
    def _create_editor(self, name: str, *, rows: int = 3, cols: int = 3) -> MatrixEditor:
        """创建一个矩阵编辑器并登记到集合里。"""
        editor = MatrixEditor(rows=rows, cols=cols, label=name)
        editor.set_latex_theme(self._latex_color(), self._palette.get("latex_bg", "#ffffff"))
        editor.matrixChanged.connect(self._on_matrix_changed)
        self._editors[name] = editor
        self._editor_order.append(name)
        self.editor_stack.addWidget(editor)
        self._refresh_matrix_combos()
        return editor

    def _next_matrix_name(self) -> str:
        """按 A、B、C… 顺序取一个没用过的名字。"""
        for code in range(ord("A"), ord("Z") + 1):
            name = chr(code)
            if name not in self._editors:
                return name
        index = 1
        while f"M{index}" in self._editors:
            index += 1
        return f"M{index}"

    @property
    def matrix_names(self) -> list[str]:
        """当前所有矩阵名（按创建顺序）。"""
        return list(self._editor_order)

    def current_editor(self) -> MatrixEditor:
        """当前正在编辑的矩阵。"""
        name = self.editor_combo.currentText()
        if name in self._editors:
            return self._editors[name]
        return self._editors[self._editor_order[0]]

    @property
    def editor_a(self) -> MatrixEditor:
        """名为 A 的矩阵（兼容旧接口；A 被删掉时退回第一个）。"""
        return self._editors.get("A") or self._editors[self._editor_order[0]]

    @property
    def editor_b(self) -> MatrixEditor:
        """名为 B 的矩阵（兼容旧接口；B 被删掉时退回主矩阵）。"""
        return self._editors.get("B") or self.editor_a

    def main_editor(self) -> MatrixEditor:
        """运算里当作 A 的矩阵。"""
        name = self.main_combo.currentText()
        return self._editors.get(name) or self._editors[self._editor_order[0]]

    def other_editor(self) -> MatrixEditor:
        """运算里当作 B 的矩阵。"""
        name = self.other_combo.currentText()
        return self._editors.get(name) or self._editors[self._editor_order[0]]

    def add_matrix(self, name: str | None = None, *, rows: int = 3, cols: int = 3) -> MatrixEditor:
        """新增一个矩阵并切换到它。"""
        name = (name or self._next_matrix_name()).strip() or self._next_matrix_name()
        while name in self._editors:
            name = self._next_matrix_name()
        editor = self._create_editor(name, rows=rows, cols=cols)
        self.editor_combo.setCurrentText(name)
        self.statusMessage.emit(f"已新增矩阵 {name}（当前共 {len(self._editors)} 个）")
        return editor

    def remove_matrix(self) -> None:
        """删除当前矩阵（至少保留一个）。"""
        name = self.editor_combo.currentText()
        if name not in self._editors:
            return
        if len(self._editors) <= 1:
            QMessageBox.information(self, "无法删除", "至少要保留一个矩阵。")
            return
        index = self._editor_order.index(name)
        editor = self._editors.pop(name)
        self._editor_order.remove(name)
        self.editor_stack.removeWidget(editor)
        editor.deleteLater()
        self._refresh_matrix_combos()
        self.editor_combo.setCurrentIndex(min(index, len(self._editor_order) - 1))
        self._on_editor_selected()
        self.statusMessage.emit(f"已删除矩阵 {name}")

    def rename_matrix(self) -> None:
        """重命名当前矩阵。"""
        old = self.editor_combo.currentText()
        if old not in self._editors:
            return
        new_name, ok = QInputDialog.getText(self, "重命名矩阵", "新的名称：", text=old)
        new_name = (new_name or "").strip()
        if not ok or not new_name or new_name == old:
            return
        if new_name in self._editors:
            QMessageBox.information(self, "名称重复", f"已经有名为「{new_name}」的矩阵了。")
            return
        editor = self._editors.pop(old)
        self._editors[new_name] = editor
        self._editor_order[self._editor_order.index(old)] = new_name
        editor.set_label(new_name)
        self._refresh_matrix_combos()
        self.editor_combo.setCurrentText(new_name)
        self.statusMessage.emit(f"矩阵 {old} 已重命名为 {new_name}")

    def _refresh_matrix_combos(self) -> None:
        """把矩阵名同步到「编辑 / 主矩阵 / 副矩阵」三个下拉框。"""
        names = self._editor_order
        for combo in (self.editor_combo, self.main_combo, self.other_combo):
            previous = combo.currentText()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(names)
                if previous in names:
                    combo.setCurrentText(previous)
            finally:
                combo.blockSignals(False)
        # 副矩阵尽量与主矩阵不同，二元运算才有意义
        if self.other_combo.currentText() == self.main_combo.currentText() and len(names) > 1:
            for candidate in names:
                if candidate != self.main_combo.currentText():
                    self.other_combo.setCurrentText(candidate)
                    break
        # 默认 A 主、B 副
        if "A" in names:
            self.main_combo.setCurrentText("A")
        if "B" in names and len(names) > 1:
            self.other_combo.setCurrentText("B")

    def _on_editor_selected(self, *_: Any) -> None:
        name = self.editor_combo.currentText()
        editor = self._editors.get(name)
        if editor is None:
            return
        self.editor_stack.setCurrentWidget(editor)
        self.matrix_box.setTitle(f"矩阵 {name}")

    def _on_matrix_changed(self) -> None:
        """矩阵内容变化时刷新状态栏提示。"""
        editor = self.sender()
        label = editor.label() if isinstance(editor, MatrixEditor) else ""
        self.statusMessage.emit(f"矩阵 {label} 已修改")

    # ================================================================== #
    # 主题 / 数据
    # ================================================================== #
    def _latex_color(self) -> str:
        """运算按钮上 LaTeX 文字的颜色（跟随主题）。"""
        return self._palette.get("text") or "#111111"

    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = dict(palette)
        self.view.set_background(palette.get("latex_bg", "#ffffff"))
        self._refresh_latex_buttons()
        self._rerender()

    def _refresh_latex_buttons(self) -> None:
        """主题/字体变化后让所有 LaTeX 按钮按新颜色重新排版。"""
        color = self._latex_color()
        background = self._palette.get("latex_bg", "#ffffff")
        for _spec, button in self._buttons:
            button.set_latex_color(color)
        for editor in self._editors.values():
            editor.set_latex_theme(color, background)

    def _ops_panel_width(self) -> int:
        """按最宽的运算标签算出中间面板需要的宽度（像素）。

        用 LaTeX 引擎实测标签宽度（比渲染成位图便宜得多），再加上按钮内边距、
        分组框边距与滚动条宽度。这样无论标签多长都能完整显示，不会出现
        「右侧按钮被裁掉」。
        """
        widest = 0.0
        for spec in matrixops.REGISTRY:
            text = _LATEX_LABELS.get(spec.key, spec.label)
            try:
                document = LatexDocument(fontsize=_BUTTON_FONT_SIZE, padding=1.5)
                document.add(
                    TextBlock(
                        text=text,
                        fontsize=_BUTTON_FONT_SIZE,
                        space_before=0.0,
                        space_after=0.0,
                        line_spacing=1.1,
                    )
                )
                # 注意要用 content_width（内容实际宽度），不是排版的可用宽度
                widest = max(widest, layout_document(document, width=8000.0).content_width)
            except Exception:
                widest = max(widest, len(spec.label) * _BUTTON_FONT_SIZE)
        # 点 → 像素（96 dpi），再留出按钮内边距 / 分组框边距 / 滚动条
        pixels = widest * 96.0 / 72.0
        return int(min(max(pixels + 96, 220), 460))

    def _refresh_datasets(self) -> None:
        pass  # 数据集通过对话框按需读取

    # ================================================================== #
    # 矩阵读取策略 / 放大编辑
    # ================================================================== #
    def _on_fill_policy_changed(self, *_: Any) -> None:
        policy = self.fill_combo.currentData() or "zero"
        trim = self.trim_check.isChecked()
        for editor in self._editors.values():
            editor.set_fill_policy(policy)
            editor.set_trim(trim)

    def _expand_editor(self, editor: MatrixEditor, name: str = "") -> None:
        editor.set_latex_theme(self._latex_color(), self._palette.get("latex_bg", "#ffffff"))
        label = name or editor.label()
        if editor.open_fullscreen(self):
            self.statusMessage.emit(
                f"已更新矩阵 {label}（{editor.table.rowCount()} 行 × {editor.table.columnCount()} 列）"
            )
            self.fill_combo.setCurrentIndex(max(self.fill_combo.findData(editor.fill_policy), 0))
            self.trim_check.setChecked(editor.trim)

    def open_manim_dialog(self) -> None:
        """打开「用 manim 渲染计算过程」对话框（用主矩阵与副矩阵）。"""
        from .dialogs import ManimDialog

        matrix = self._read_matrix(self.main_editor(), required=True, name=self.main_combo.currentText())
        if matrix is None:
            return
        try:
            other = self.other_editor().matrix()
        except ValueError:
            other = None
        dialog = ManimDialog(
            self,
            matrix=matrix,
            other=other,
            label=self.main_combo.currentText() or "A",
            palette=self._palette,
        )
        dialog.exec()

    # ================================================================== #
    # 输入辅助
    # ================================================================== #
    def _import_clipboard(self, editor: MatrixEditor) -> None:
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            self.statusMessage.emit("剪贴板为空")
            return
        editor.set_text(text)
        self.statusMessage.emit(f"已从剪贴板导入 {editor.label()}")

    def _export_clipboard(self, editor: MatrixEditor) -> None:
        QGuiApplication.clipboard().setText(editor.text())
        self.statusMessage.emit(f"已复制 {editor.label()} 到剪贴板")

    def _import_from_dataset(self) -> None:
        names = self.store.names()
        if not names:
            QMessageBox.information(self, "没有数据集", "请先在「数据」页导入数据。")
            return
        name, ok = QInputDialog.getItem(self, "选择数据集", "数据集：", names, 0, False)
        if not ok or not name:
            return
        dataset = self.store.get(name)
        if dataset is None:
            return
        numeric = numeric_columns(dataset.frame)
        if not numeric:
            QMessageBox.information(self, "没有数值列", "该数据集没有可用的数值列。")
            return
        text, ok = QInputDialog.getText(
            self,
            "选择列",
            f"要组成矩阵的列（逗号分隔）：\n可用列：{', '.join(numeric[:12])}",
            text=",".join(numeric[:2]),
        )
        if not ok or not text.strip():
            return
        columns = [c.strip() for c in text.replace("，", ",").split(",") if c.strip()]
        missing = [c for c in columns if c not in dataset.frame.columns]
        if missing:
            QMessageBox.warning(self, "列不存在", f"找不到这些列：{', '.join(missing)}")
            return
        matrix = dataset.frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        matrix = matrix[~np.isnan(matrix).any(axis=1)]
        if matrix.size == 0:
            QMessageBox.warning(self, "没有有效数据", "选中的列没有完整的数值行。")
            return
        self.current_editor().set_matrix(matrix, precision=self.precision_spin.value())
        self.statusMessage.emit(
            f"已从「{name}」导入 {matrix.shape[0]}×{matrix.shape[1]} 矩阵到 {self.current_editor().label()}"
        )

    # ================================================================== #
    # 运算
    # ================================================================== #
    def _apply_filter(self, text: str) -> None:
        keyword = (text or "").strip().lower()
        for spec, button in self._buttons:
            haystack = f"{spec.label} {spec.key} {spec.group} {spec.description}".lower()
            button.setVisible(not keyword or keyword in haystack)

    def _read_matrix(self, editor: MatrixEditor, *, required: bool, name: str) -> np.ndarray | None:
        try:
            matrix = editor.matrix()
        except ValueError as exc:
            QMessageBox.warning(self, f"矩阵 {name} 有误", str(exc))
            return None
        if matrix is None and required:
            QMessageBox.information(self, f"缺少矩阵 {name}", f"请输入矩阵 {name} 的内容。")
            return None
        return matrix

    def run_operation(self, key: str) -> None:
        spec = next((s for s in matrixops.REGISTRY if s.key == key), None)
        if spec is None:
            return
        main_name = self.main_combo.currentText() or "A"
        other_name = self.other_combo.currentText() or "B"
        matrix_a = self._read_matrix(self.main_editor(), required=spec.arity >= 1, name=main_name)
        if spec.arity >= 1 and matrix_a is None:
            return
        matrix_b = None
        if spec.arity >= 2:
            matrix_b = self._read_matrix(self.other_editor(), required=True, name=other_name)
            if matrix_b is None:
                return
        scalar = self.scalar_spin.value() if spec.scalar else None

        try:
            sheet = matrixops.run(key, matrix_a, matrix_b, scalar)
        except Exception as exc:
            sheet = ResultSheet(
                title="计算失败",
                items=[ResultItem(title="运算出错", note=f"{type(exc).__name__}: {exc}")],
            )
        heading = ResultItem(
            level=1,
            title=("▸ 向量组 · " if spec.vector_input else "▸ ") + spec.label,
        )
        self._sheet.items.append(heading)
        self._sheet.items.extend(sheet.items)
        self._render_sheet()
        self.statusMessage.emit(f"已完成运算：{spec.label}")

    def run_common(self) -> None:
        matrix = self._read_matrix(
            self.main_editor(), required=True, name=self.main_combo.currentText() or "A"
        )
        if matrix is None:
            return
        for key in ("determinant", "rank", "trace", "inverse", "eigen", "norms"):
            spec = next((s for s in matrixops.REGISTRY if s.key == key), None)
            if spec is None:
                continue
            try:
                sheet = matrixops.run(key, matrix, None, None)
            except Exception as exc:
                sheet = ResultSheet(items=[ResultItem(title=spec.label, note=str(exc))])
            self._sheet.items.append(ResultItem(level=1, title=f"▸ {spec.label}"))
            self._sheet.items.extend(sheet.items)
        self._render_sheet()
        self.statusMessage.emit("已计算常用量：行列式、秩、迹、逆、特征值、范数")

    def run_vector_analysis(self) -> None:
        matrix = self._read_matrix(
            self.main_editor(), required=True, name=self.main_combo.currentText() or "A"
        )
        if matrix is None:
            return
        self.vector_check.setChecked(True)
        for key in ("vector_rank", "linear_independence", "gram_schmidt", "vector_norm"):
            spec = next((s for s in matrixops.REGISTRY if s.key == key), None)
            if spec is None:
                continue
            try:
                sheet = matrixops.run(key, matrix, None, None)
            except Exception as exc:
                sheet = ResultSheet(items=[ResultItem(title=spec.label, note=str(exc))])
            self._sheet.items.append(ResultItem(level=1, title=f"▸ {spec.label}"))
            self._sheet.items.extend(sheet.items)
        self._render_sheet()
        self.statusMessage.emit("已完成向量组分析")

    def clear_results(self) -> None:
        self._sheet = ResultSheet(title="矩阵与向量组计算结果")
        self.view.set_message("结果已清空。点击左侧运算按钮重新计算。")

    # ================================================================== #
    # 渲染
    # ================================================================== #
    def _rerender(self, *_: Any) -> None:
        if self._sheet.items:
            self._render_sheet()

    def _render_sheet(self) -> None:
        from core.latex.results import ResultStyle

        palette = self._palette
        style = ResultStyle(
            fontsize=12.5,
            title_color=palette.get("latex_title", "#1d4ed8"),
            text_color=palette.get("latex_text", "#111827"),
            note_color=palette.get("latex_note", "#6b7280"),
            rule_color=palette.get("latex_rule", "#d8dee9"),
            background=palette.get("latex_bg", "#ffffff"),
        )
        self.view.set_sheet(self._sheet, style)

    def _latex_source(self) -> str:
        blocks = []
        for item in self._sheet.items:
            if item.title:
                blocks.append(f"% {item.title}")
            if item.latex:
                blocks.append(f"\\[{item.latex}\\]")
            if item.note:
                blocks.append(f"% {item.note}")
            blocks.append("")
        return "\n".join(blocks).strip()

    def _copy_latex(self) -> None:
        text = self._latex_source()
        if not text:
            self.statusMessage.emit("还没有结果可复制")
            return
        QGuiApplication.clipboard().setText(text)
        self.statusMessage.emit("LaTeX 源码已复制到剪贴板")

    def _copy_image(self) -> None:
        if self.view.copy_image():
            self.statusMessage.emit("结果图片已复制到剪贴板")

    def _show_latex_source(self) -> None:
        TextInfoDialog(self, title="LaTeX 源码", text=self._latex_source() or "（暂无结果）").exec()

    def _export_svg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出结果 SVG", "matrix_result.svg", "SVG 矢量图 (*.svg)")
        if path and self.view.export_svg(path):
            self.statusMessage.emit(f"已导出：{path}")

    def _export_text(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出结果文本", "matrix_result.txt", "文本文件 (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._sheet.plain_text())
                handle.write("\n\n===== LaTeX 源码 =====\n\n")
                handle.write(self._latex_source())
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.statusMessage.emit(f"已导出：{path}")
