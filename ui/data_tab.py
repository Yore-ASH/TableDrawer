"""数据页：查看 / 编辑数据集，导入 CSV，派生列，统计摘要。"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from core.dataset import DataSet, DataSetStore, column_kind, numeric_columns
from core.numberfmt import to_latex
from core.spec import ResultItem, ResultSheet

from .dialogs import CsvImportDialog, ExpressionDialog, ManualDataDialog, ResultViewDialog
from .widgets import DataTableWidget

__all__ = ["DataTab"]


class DataTab(QWidget):
    """数据集管理页。"""

    datasetChanged = Signal(str)
    statusMessage = Signal(str)
    plotRequested = Signal(str)

    def __init__(self, store: DataSetStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self._palette: dict[str, str] = {}
        self._loading = False

        # ---- 顶部工具条 -------------------------------------------------- #
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(200)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_selected)

        import_btn = QPushButton("导入 CSV…")
        import_btn.clicked.connect(self.open_csv_dialog)
        manual_btn = QPushButton("手动添加…")
        manual_btn.clicked.connect(self.open_manual_dialog)
        expression_btn = QPushButton("表达式生成…")
        expression_btn.clicked.connect(self.open_expression_dialog)
        rename_btn = QPushButton("重命名")
        rename_btn.clicked.connect(self.rename_dataset)
        remove_btn = QPushButton("删除数据集")
        remove_btn.setProperty("danger", "true")
        remove_btn.clicked.connect(self.remove_dataset)

        self.info_label = QLabel("还没有数据集，请先导入 CSV 或手动添加。")
        self.info_label.setProperty("muted", "true")
        self.info_label.setWordWrap(True)

        head = QHBoxLayout()
        head.addWidget(QLabel("数据集"))
        head.addWidget(self.dataset_combo)
        head.addWidget(import_btn)
        head.addWidget(manual_btn)
        head.addWidget(expression_btn)
        head.addWidget(rename_btn)
        head.addWidget(remove_btn)
        head.addStretch(1)

        # ---- 表格 -------------------------------------------------------- #
        self.table = DataTableWidget(editable=True)
        self.table.frameEdited.connect(self._on_frame_edited)

        # ---- 列面板 ------------------------------------------------------ #
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.column_list.currentRowChanged.connect(lambda *_: self._update_column_stats())

        add_col = QPushButton("添加列")
        add_col.clicked.connect(self.add_column)
        rename_col = QPushButton("重命名列")
        rename_col.clicked.connect(self.rename_column)
        drop_col = QPushButton("删除列")
        drop_col.clicked.connect(self.drop_column)
        col_buttons = QHBoxLayout()
        col_buttons.addWidget(add_col)
        col_buttons.addWidget(rename_col)
        col_buttons.addWidget(drop_col)

        self.stats_label = QLabel("选择一列查看统计量")
        self.stats_label.setWordWrap(True)
        self.stats_label.setProperty("muted", "true")
        self.stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        summary_btn = QPushButton("统计摘要（LaTeX）")
        summary_btn.clicked.connect(self.show_summary)
        copy_btn = QPushButton("复制表格")
        copy_btn.clicked.connect(self.copy_table)
        export_btn = QPushButton("导出 CSV…")
        export_btn.clicked.connect(self.export_csv)
        plot_btn = QPushButton("用此数据绘图 →")
        plot_btn.setProperty("accent", "true")
        plot_btn.clicked.connect(lambda: self.plotRequested.emit(self.current_name()))

        panel_layout = QVBoxLayout()
        panel_box = QGroupBox("列")
        box_layout = QVBoxLayout(panel_box)
        box_layout.addWidget(self.column_list, 1)
        box_layout.addLayout(col_buttons)
        box_layout.addWidget(QLabel("列统计"))
        box_layout.addWidget(self.stats_label)

        actions = QVBoxLayout()
        actions.addWidget(summary_btn)
        actions.addWidget(copy_btn)
        actions.addWidget(export_btn)
        actions.addWidget(plot_btn)
        actions.addStretch(1)

        panel_layout.addWidget(panel_box, 1)
        panel_layout.addLayout(actions)
        panel = QWidget()
        panel.setLayout(panel_layout)
        # 只给最小宽度，不设上限（硬上限会与分隔条拖动打架）
        panel.setMinimumWidth(250)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.table)
        splitter.addWidget(panel)
        # 禁止把面板拖成 0 宽（塌掉再弹回来会像乱跳）；要逐个设置才生效
        splitter.setChildrenCollapsible(False)
        for index in range(splitter.count()):
            splitter.setCollapsible(index, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([760, 280])

        layout = QVBoxLayout(self)
        layout.addLayout(head)
        layout.addWidget(self.info_label)
        layout.addWidget(splitter, 1)

        self.store.subscribe(self.refresh_datasets)
        self.refresh_datasets()

    # ------------------------------------------------------------------ #
    # 基础
    # ------------------------------------------------------------------ #
    def set_palette(self, palette: dict[str, str]) -> None:
        self._palette = dict(palette)

    def current_name(self) -> str:
        return self.dataset_combo.currentText()

    def current_dataset(self) -> DataSet | None:
        return self.store.get(self.current_name())

    def _on_dataset_selected(self, name: str) -> None:
        if self._loading:
            return
        dataset = self.store.get(name)
        self.table.set_frame(dataset.frame if dataset else None)
        self._refresh_columns()
        if dataset is not None:
            note = self.table.display_note()
            source = {"csv": "CSV 文件", "manual": "手动录入", "expression": "表达式生成", "sample": "内置示例"}.get(
                dataset.source, dataset.source
            )
            extra = f"　{note}" if note else ""
            self.info_label.setText(f"{dataset.shape_text}　来源：{source}　{directory_hint(dataset.path)}{extra}")
        else:
            self.info_label.setText("还没有数据集，请先导入 CSV 或手动添加。")
        self.datasetChanged.emit(name)

    def refresh_datasets(self) -> None:
        """根据仓库重建数据集下拉框。"""
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
        self._on_dataset_selected(self.dataset_combo.currentText())

    def select_dataset(self, name: str) -> None:
        if name in self.store:
            self.dataset_combo.setCurrentText(name)

    def _refresh_columns(self) -> None:
        self._loading = True
        try:
            self.column_list.clear()
            dataset = self.current_dataset()
            if dataset is None:
                return
            for name in dataset.columns:
                kind = column_kind(dataset.frame[name])
                label = {"numeric": "数值", "datetime": "日期", "text": "文本"}.get(kind, kind)
                item = QListWidgetItem(f"{name}　[{label}]")
                item.setData(Qt.UserRole, name)
                self.column_list.addItem(item)
            if self.column_list.count():
                self.column_list.setCurrentRow(0)
        finally:
            self._loading = False
        self._update_column_stats()

    def _on_frame_edited(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        dataset.frame = self.table.frame()
        self._refresh_columns()
        self.info_label.setText(f"{dataset.shape_text}　（已修改，尚未保存到磁盘）")
        self.datasetChanged.emit(dataset.name)

    # ------------------------------------------------------------------ #
    # 导入 / 添加
    # ------------------------------------------------------------------ #
    def open_csv_dialog(self, path: str = "") -> DataSet | None:
        dialog = CsvImportDialog(self, path=path, palette=self._palette)
        if dialog.exec() != CsvImportDialog.Accepted:
            return None
        dataset = dialog.result_dataset()
        if dataset is None:
            return None
        self.store.add(dataset)
        self.select_dataset(dataset.name)
        self.statusMessage.emit(f"已导入数据集「{dataset.name}」（{dataset.shape_text}）")
        return dataset

    def open_manual_dialog(self) -> DataSet | None:
        dialog = ManualDataDialog(self)
        if dialog.exec() != ManualDataDialog.Accepted:
            return None
        dataset = dialog.dataset()
        self.store.add(dataset)
        self.select_dataset(dataset.name)
        self.statusMessage.emit(f"已添加数据集「{dataset.name}」（{dataset.shape_text}）")
        return dataset

    def open_expression_dialog(self) -> None:
        frames = self.store.frames()
        if not frames:
            QMessageBox.information(self, "没有数据集", "请先导入或手动添加一个数据集。")
            return
        dialog = ExpressionDialog(self, frames=frames, initial_dataset=self.current_name())
        if dialog.exec() != ExpressionDialog.Accepted:
            return
        try:
            mode, payload, name = dialog.result()
        except Exception as exc:
            QMessageBox.warning(self, "表达式有误", str(exc))
            return
        if mode == "derive":
            dataset = self.current_dataset()
            if dataset is None:
                return
            frame = dataset.frame.copy()
            frame[name] = np.asarray(payload)
            dataset.frame = frame
            self.table.set_frame(frame)
            self._refresh_columns()
            self.statusMessage.emit(f"已在「{dataset.name}」中新增列「{name}」")
        else:
            dataset = DataSet(name=name, frame=payload, source="expression")
            self.store.add(dataset)
            self.select_dataset(dataset.name)
            self.statusMessage.emit(f"已生成数据集「{dataset.name}」（{dataset.shape_text}）")

    # ------------------------------------------------------------------ #
    # 列操作
    # ------------------------------------------------------------------ #
    def _selected_column(self) -> str:
        item = self.column_list.currentItem()
        return str(item.data(Qt.UserRole)) if item is not None else ""

    def add_column(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        new_name, ok = QInputDialog.getText(self, "添加列", "新列名：", text=f"列{dataset.n_cols + 1}")
        new_name = (new_name or "").strip()
        if not ok or not new_name:
            return
        if new_name in dataset.frame.columns:
            QMessageBox.information(self, "列已存在", f"数据集中已经有名为「{new_name}」的列。")
            return
        frame = dataset.frame.copy()
        frame[new_name] = np.nan
        dataset.frame = frame
        self.table.set_frame(frame)
        self._refresh_columns()
        self.datasetChanged.emit(dataset.name)

    def rename_column(self) -> None:
        dataset = self.current_dataset()
        old = self._selected_column()
        if dataset is None or not old:
            return
        name, ok = QInputDialog.getText(self, "重命名列", "新的列名：", text=old)
        if not ok or not name.strip() or name == old:
            return
        dataset.frame = dataset.frame.rename(columns={old: name.strip()})
        self.table.set_frame(dataset.frame)
        self._refresh_columns()
        self.datasetChanged.emit(dataset.name)

    def drop_column(self) -> None:
        dataset = self.current_dataset()
        old = self._selected_column()
        if dataset is None or not old:
            return
        if dataset.n_cols <= 1:
            QMessageBox.information(self, "无法删除", "数据集至少要保留一列。")
            return
        dataset.frame = dataset.frame.drop(columns=[old])
        self.table.set_frame(dataset.frame)
        self._refresh_columns()
        self.datasetChanged.emit(dataset.name)

    def rename_dataset(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        name, ok = QInputDialog.getText(self, "重命名数据集", "新的名称：", text=dataset.name)
        if not ok or not name.strip() or name == dataset.name:
            return
        self.store.rename(dataset.name, name.strip())
        self.statusMessage.emit(f"数据集已重命名为「{name.strip()}」")

    def remove_dataset(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        answer = QMessageBox.question(
            self, "删除数据集", f"确定要删除数据集「{dataset.name}」吗？", QMessageBox.Yes | QMessageBox.No
        )
        if answer != QMessageBox.Yes:
            return
        self.store.remove(dataset.name)
        self.statusMessage.emit(f"已删除数据集「{dataset.name}」")

    # ------------------------------------------------------------------ #
    # 统计 / 导出
    # ------------------------------------------------------------------ #
    def _update_column_stats(self) -> None:
        if self._loading:
            return
        dataset = self.current_dataset()
        name = self._selected_column()
        # 列表刷新过程中 currentRowChanged 可能带着上一个数据集的列名触发，必须校验
        if dataset is None or not name or name not in dataset.frame.columns:
            self.stats_label.setText("选择一列查看统计量")
            return
        series = dataset.frame[name]
        kind = column_kind(series)
        if kind == "numeric":
            values = pd.to_numeric(series, errors="coerce").dropna()
            if values.empty:
                self.stats_label.setText("该列没有有效数值")
                return
            self.stats_label.setText(
                f"计数：{len(values)}\n"
                f"均值：{values.mean():.6g}\n"
                f"标准差：{values.std():.6g}\n"
                f"最小：{values.min():.6g}\n"
                f"中位：{values.median():.6g}\n"
                f"最大：{values.max():.6g}"
            )
        else:
            self.stats_label.setText(
                f"类型：{kind}\n非空：{series.notna().sum()}\n唯一值：{series.nunique()}\n"
                f"示例：{', '.join(str(v) for v in series.dropna().head(3))}"
            )

    def show_summary(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        sheet = build_summary_sheet(dataset)
        dialog = ResultViewDialog(
            self,
            title=f"统计摘要 · {dataset.name}",
            sheet=sheet,
            background=self._palette.get("latex_bg", "#ffffff"),
        )
        dialog.exec()

    def copy_table(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        QGuiApplication.clipboard().setText(dataset.frame.to_csv(sep="\t", index=False))
        self.statusMessage.emit("已将表格复制到剪贴板（Tab 分隔）")

    def export_csv(self) -> None:
        dataset = self.current_dataset()
        if dataset is None:
            return
        base = "".join(ch for ch in dataset.name if ch.isalnum() or ch in "-_ ") or "data"
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", f"{base}.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            dataset.frame.to_csv(path, index=False, encoding="utf-8-sig")
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.statusMessage.emit(f"已导出到 {path}")


def directory_hint(path: str) -> str:
    """把文件路径缩成适合放在信息栏里的短文本。"""
    if not path:
        return ""
    try:
        relative = os.path.relpath(path, os.getcwd())
    except ValueError:
        relative = path
    text = relative if not relative.startswith("..") else path
    if len(text) > 56:
        text = "…" + text[-55:]
    return f"文件：{text}"


def _tex_text(name: str) -> str:
    """把列名安全地放进 ``\\text{...}``。"""
    cleaned = str(name).replace("{", "(").replace("}", ")")
    return rf"\text{{{cleaned}}}"


#: 统计量 -> describe() 的行标签
_STAT_ROWS: list[tuple[str, str]] = [
    ("count", "计数"),
    ("mean", "均值"),
    ("std", "标准差"),
    ("min", "最小值"),
    ("25%", "下四分位"),
    ("50%", "中位数"),
    ("75%", "上四分位"),
    ("max", "最大值"),
]


def build_summary_sheet(dataset: DataSet, max_columns: int = 8) -> ResultSheet:
    """构造数据集的 LaTeX 统计摘要。"""
    frame = dataset.frame
    sheet = ResultSheet(title=f"数据统计摘要 · {dataset.name}")
    sheet.add(
        ResultItem(
            title="基本信息",
            latex=(
                rf"\text{{行数}} = {len(frame)},\quad \text{{列数}} = {frame.shape[1]},"
                rf"\quad \text{{数值列}} = {len(numeric_columns(frame))}"
            ),
            note=f"来源：{dataset.source}　文件：{dataset.path or '（未保存）'}",
        )
    )

    numeric = numeric_columns(frame)
    if not numeric:
        sheet.add(ResultItem(level=1, title="没有数值列，无法计算统计量"))
        return sheet

    selected = numeric[:max_columns]
    desc = frame[selected].apply(pd.to_numeric, errors="coerce").describe()
    header = " & ".join([r"\text{统计量}"] + [_tex_text(name) for name in selected])
    lines = [header, r"\hline"]
    for key, label in _STAT_ROWS:
        if key not in desc.index:
            continue
        cells = [rf"\text{{{label}}}"]
        for name in selected:
            cells.append(to_latex(desc.loc[key, name], precision=6))
        lines.append(" & ".join(cells))
    latex_table = r"\begin{array}{l|" + "r" * len(selected) + "}" + r" \\ ".join(lines) + r"\end{array}"
    sheet.add(ResultItem(title="描述性统计", latex=latex_table, kind="matrix"))

    if len(numeric) > max_columns:
        sheet.add(
            ResultItem(
                note=f"仅显示前 {max_columns} 个数值列，未显示的列：{', '.join(numeric[max_columns:])}",
            )
        )

    # 相关系数矩阵（>=2 列时）
    if len(selected) >= 2:
        corr = frame[selected].apply(pd.to_numeric, errors="coerce").corr()
        rows = [" & ".join([r"\text{相关系数}"] + [_tex_text(name) for name in selected])]
        for name in selected:
            cells = [_tex_text(name)]
            for other in selected:
                value = corr.loc[name, other]
                cells.append(to_latex(float(value) if pd.notna(value) else float("nan"), precision=4))
            rows.append(" & ".join(cells))
        latex_corr = r"\begin{array}{l|" + "r" * len(selected) + "}" + r" \\ ".join(rows) + r"\end{array}"
        sheet.add(ResultItem(title="相关系数矩阵", latex=latex_corr, kind="matrix"))

    return sheet
