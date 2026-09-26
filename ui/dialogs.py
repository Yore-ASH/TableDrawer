"""各类对话框：CSV 导入、手动录入、表达式生成、源码导出、结果查看。"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import codesync, manimgen
from core.dataset import (
    DELIMITER_CHOICES,
    EXPRESSION_HELP,
    CsvReadOptions,
    DataSet,
    evaluate_expression,
    frame_from_text,
    read_csv_frame,
)
from core.latex import LatexDocument
from core.spec import PlotSpec
from core.latex.results import ResultStyle

from .widgets import DataTableWidget, LatexView, clipboard_to_grid

__all__ = [
    "CsvImportDialog",
    "ManualDataDialog",
    "ExpressionDialog",
    "ExportSourceDialog",
    "ResultViewDialog",
    "TextInfoDialog",
    "ManimDialog",
    "storyboard_document",
]

_MONO = "Consolas, 'Cascadia Mono', 'Courier New', monospace"


def _hline() -> QWidget:
    line = QWidget()
    line.setFixedHeight(1)
    line.setStyleSheet("background: palette(mid);")
    return line


# --------------------------------------------------------------------------- #
# CSV 导入
# --------------------------------------------------------------------------- #
class CsvImportDialog(QDialog):
    """CSV 导入对话框：自动探测 + 手动调整 + 实时预览。"""

    def __init__(self, parent: QWidget | None = None, path: str = "", *, palette: Mapping[str, str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("导入 CSV 数据")
        self.resize(940, 660)
        self._path = path
        self._palette = dict(palette or {})
        self._preview = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(220)
        self._timer.timeout.connect(self._reload)

        # ---- 文件选择 ---------------------------------------------------- #
        self.path_edit = QLineEdit(path)
        self.path_edit.setReadOnly(True)
        browse = QPushButton("浏览…")
        browse.clicked.connect(self._browse)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("文件"))
        file_row.addWidget(self.path_edit, 1)
        file_row.addWidget(browse)

        # ---- 选项 -------------------------------------------------------- #
        self.name_edit = QLineEdit(os.path.splitext(os.path.basename(path))[0] if path else "数据集")
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["自动", "utf-8-sig", "utf-8", "gb18030", "big5", "cp936", "latin-1"])
        self.delimiter_combo = QComboBox()
        for label in DELIMITER_CHOICES:
            self.delimiter_combo.addItem(label, DELIMITER_CHOICES[label])
        self.header_combo = QComboBox()
        self.header_combo.addItem("自动判断", "auto")
        self.header_combo.addItem("有表头", "yes")
        self.header_combo.addItem("无表头", "no")
        self.decimal_combo = QComboBox()
        self.decimal_combo.addItems([".", ","])
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 100000)
        self.skip_spin.setSuffix(" 行")
        self.comment_edit = QLineEdit("#")
        self.comment_edit.setMaxLength(3)
        self.na_edit = QLineEdit()
        self.na_edit.setPlaceholderText("如 -999,NA,缺失")
        self.index_check = QCheckBox("使用第一列作为行索引")

        form = QFormLayout()
        form.addRow("数据集名称", self.name_edit)
        form.addRow("编码", self.encoding_combo)
        form.addRow("分隔符", self.delimiter_combo)
        form.addRow("表头", self.header_combo)
        form.addRow("小数点", self.decimal_combo)
        form.addRow("跳过开头", self.skip_spin)
        form.addRow("注释前缀", self.comment_edit)
        form.addRow("缺失值标记", self.na_edit)
        form.addRow("", self.index_check)

        options_box = QGroupBox("读取选项")
        options_box.setLayout(form)

        self.refresh_button = QPushButton("重新预览")
        self.refresh_button.clicked.connect(self._reload)
        options_layout = QVBoxLayout()
        options_layout.addWidget(options_box)
        options_layout.addWidget(self.refresh_button)
        options_layout.addStretch(1)
        options_panel = QWidget()
        options_panel.setLayout(options_layout)
        options_panel.setMinimumWidth(300)

        # ---- 预览 -------------------------------------------------------- #
        self.info_label = QLabel("尚未读取")
        self.info_label.setProperty("muted", "true")
        self.info_label.setWordWrap(True)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #d29922;")
        self.table = DataTableWidget(editable=False, max_rows=500)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(self.info_label)
        preview_layout.addWidget(self.warn_label)
        preview_layout.addWidget(self.table, 1)
        preview_panel = QWidget()
        preview_panel.setLayout(preview_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(options_panel)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 620])

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("导入")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(splitter, 1)
        layout.addWidget(buttons)

        for widget, signal in (
            (self.encoding_combo, None),
            (self.delimiter_combo, None),
            (self.header_combo, None),
            (self.decimal_combo, None),
        ):
            widget.currentIndexChanged.connect(lambda *_: self._timer.start())
        self.skip_spin.valueChanged.connect(lambda *_: self._timer.start())
        self.comment_edit.textChanged.connect(lambda *_: self._timer.start())
        self.na_edit.textChanged.connect(lambda *_: self._timer.start())
        self.index_check.toggled.connect(lambda *_: self._timer.start())

        if path:
            self._reload()

    # -- 内部 -------------------------------------------------------------- #
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV 文件", "", "表格文件 (*.csv *.tsv *.txt *.dat);;所有文件 (*)"
        )
        if path:
            self._path = path
            self.path_edit.setText(path)
            if not self.name_edit.text().strip() or self.name_edit.text() == "数据集":
                self.name_edit.setText(os.path.splitext(os.path.basename(path))[0])
            self._reload()

    def options(self) -> CsvReadOptions:
        return CsvReadOptions(
            delimiter=self.delimiter_combo.currentData() or "auto",
            encoding="auto" if self.encoding_combo.currentIndex() == 0 else self.encoding_combo.currentText(),
            header=self.header_combo.currentData() or "auto",
            decimal=self.decimal_combo.currentText(),
            skip_rows=self.skip_spin.value(),
            comment=self.comment_edit.text(),
            na_values=self.na_edit.text(),
            use_first_column_as_index=self.index_check.isChecked(),
        )

    def _reload(self) -> None:
        self._timer.stop()
        if not self._path or not os.path.exists(self._path):
            self.info_label.setText("请选择一个 CSV 文件")
            return
        try:
            preview = read_csv_frame(self._path, self.options())
        except Exception as exc:
            self.info_label.setText("读取失败")
            self.warn_label.setText(str(exc))
            self.table.set_frame(None)
            return
        self._preview = preview
        self.table.set_frame(preview.frame)
        self.info_label.setText(
            f"识别结果：编码 {preview.encoding} ／ 分隔符 {_delimiter_label(preview.delimiter)} ／ "
            f"{'有表头' if preview.has_header else '无表头'} ／ {preview.frame.shape[0]} 行 × {preview.frame.shape[1]} 列"
        )
        self.warn_label.setText("；".join(preview.warnings))

    # -- 结果 -------------------------------------------------------------- #
    def dataset_name(self) -> str:
        return self.name_edit.text().strip() or "数据集"

    def result_frame(self) -> pd.DataFrame | None:
        return self._preview.frame if self._preview is not None else None

    def result_dataset(self) -> DataSet | None:
        frame = self.result_frame()
        if frame is None:
            return None
        preview = self._preview
        return DataSet(
            name=self.dataset_name(),
            frame=frame,
            source="csv",
            path=os.path.abspath(self._path) if self._path else "",
            meta={
                "encoding": preview.encoding if preview else "",
                "delimiter": preview.delimiter if preview else "",
                "has_header": preview.has_header if preview else True,
            },
        )

    def accept(self) -> None:  # noqa: D102
        if self.result_frame() is None:
            QMessageBox.warning(self, "无法导入", "还没有成功读取任何数据。")
            return
        if self.result_frame().shape[1] == 0:
            QMessageBox.warning(self, "无法导入", "解析结果中没有可用列。")
            return
        super().accept()


def _delimiter_label(delimiter: str) -> str:
    return {"\t": "Tab", " ": "空格"}.get(delimiter, repr(delimiter).strip("'"))


# --------------------------------------------------------------------------- #
# 手动录入
# --------------------------------------------------------------------------- #
class ManualDataDialog(QDialog):
    """手动添加数据组：表格录入或直接粘贴文本。"""

    def __init__(self, parent: QWidget | None = None, *, name: str = ""):
        super().__init__(parent)
        self.setWindowTitle("手动添加数据组")
        self.resize(880, 620)

        self.name_edit = QLineEdit(name or "手动数据")
        self.name_edit.setPlaceholderText("请输入数据集名称")

        head = QFormLayout()
        head.addRow("数据集名称", self.name_edit)

        # ---- 表格模式 ---------------------------------------------------- #
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 5000)
        self.rows_spin.setValue(8)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 200)
        self.cols_spin.setValue(3)
        self.table = DataTableWidget(editable=True, max_rows=5000)
        self.table.set_frame(pd.DataFrame({f"列{j + 1}": [np.nan] * 8 for j in range(3)}))

        apply_btn = QPushButton("应用尺寸")
        apply_btn.clicked.connect(self._apply_size)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(lambda: self.table.set_frame(pd.DataFrame(
            {f"列{j + 1}": [np.nan] * self.rows_spin.value() for j in range(self.cols_spin.value())}
        )))
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("行"))
        size_row.addWidget(self.rows_spin)
        size_row.addWidget(QLabel("列"))
        size_row.addWidget(self.cols_spin)
        size_row.addWidget(apply_btn)
        size_row.addWidget(clear_btn)
        size_row.addStretch(1)

        hint = QLabel("提示：可以直接从 Excel / 文本编辑器复制后按 Ctrl+V 粘贴；双击单元格即可编辑列名与数值。")
        hint.setWordWrap(True)
        hint.setProperty("muted", "true")

        grid_page = QWidget()
        grid_layout = QVBoxLayout(grid_page)
        grid_layout.addLayout(size_row)
        grid_layout.addWidget(hint)
        grid_layout.addWidget(self.table, 1)

        # ---- 文本模式 ---------------------------------------------------- #
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(
            "在此粘贴数据，例如：\n\ntime,value\n0,1.0\n1,2.5\n2,4.1\n"
        )
        self.text_edit.setFont(QFont(_MONO.split(",")[0], 10))
        self.text_delimiter = QComboBox()
        for label in DELIMITER_CHOICES:
            self.text_delimiter.addItem(label, DELIMITER_CHOICES[label])
        self.text_header = QCheckBox("首行是表头")
        self.text_header.setChecked(True)
        parse_btn = QPushButton("解析到表格")
        parse_btn.clicked.connect(self._parse_text)

        text_opts = QHBoxLayout()
        text_opts.addWidget(QLabel("分隔符"))
        text_opts.addWidget(self.text_delimiter)
        text_opts.addWidget(self.text_header)
        text_opts.addStretch(1)
        text_opts.addWidget(parse_btn)

        text_page = QWidget()
        text_layout = QVBoxLayout(text_page)
        text_layout.addWidget(self.text_edit, 1)
        text_layout.addLayout(text_opts)

        tabs = QTabWidget()
        tabs.addTab(grid_page, "表格录入")
        tabs.addTab(text_page, "文本粘贴")
        self._tabs = tabs

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(head)
        layout.addWidget(tabs, 1)
        layout.addWidget(buttons)

    def _apply_size(self) -> None:
        rows, cols = self.rows_spin.value(), self.cols_spin.value()
        frame = self.table.frame()
        new_columns = list(frame.columns) + [f"列{j + 1}" for j in range(cols)]
        new_columns = new_columns[:cols]
        data = pd.DataFrame(index=range(rows))
        for j, name in enumerate(new_columns):
            column = frame.iloc[:, j] if j < frame.shape[1] else pd.Series([np.nan] * rows)
            values = list(column)[:rows]
            values += [np.nan] * (rows - len(values))
            data[name] = values
        self.table.set_frame(data)

    def _parse_text(self) -> None:
        text = self.text_edit.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "没有内容", "请先粘贴数据。")
            return
        try:
            frame = frame_from_text(
                text,
                delimiter=self.text_delimiter.currentData() or "auto",
                has_header=self.text_header.isChecked(),
            )
        except Exception as exc:
            QMessageBox.warning(self, "解析失败", str(exc))
            return
        if frame.empty:
            QMessageBox.warning(self, "解析失败", "没有解析出任何数据行。")
            return
        self.table.set_frame(frame)
        self.rows_spin.setValue(min(max(frame.shape[0], 1), self.rows_spin.maximum()))
        self.cols_spin.setValue(min(max(frame.shape[1], 1), self.cols_spin.maximum()))
        self._tabs.setCurrentIndex(0)

    def frame(self) -> pd.DataFrame:
        table_frame = self.table.frame()
        if table_frame.shape[1] == 0:
            return frame_from_text(
                self.text_edit.toPlainText(),
                delimiter=self.text_delimiter.currentData() or "auto",
                has_header=self.text_header.isChecked(),
            )
        return table_frame

    def dataset(self) -> DataSet:
        frame = self.frame()
        frame = frame.dropna(axis=0, how="all").reset_index(drop=True)
        return DataSet(name=self.name_edit.text().strip() or "手动数据", frame=frame, source="manual")

    def accept(self) -> None:  # noqa: D102
        try:
            frame = self.dataset().frame
        except Exception as exc:
            QMessageBox.warning(self, "无法保存", str(exc))
            return
        if frame.shape[1] == 0 or frame.shape[0] == 0:
            QMessageBox.warning(self, "无法保存", "请至少录入一行一列数据。")
            return
        super().accept()


# --------------------------------------------------------------------------- #
# 表达式
# --------------------------------------------------------------------------- #
class ExpressionDialog(QDialog):
    """用 NumPy 表达式派生新列，或从零生成一个新数据集。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        frames: Mapping[str, pd.DataFrame] | None = None,
        initial_dataset: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("表达式生成数据")
        self.resize(880, 640)
        self._frames = dict(frames or {})

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("派生新列（在已有数据集上）", "derive")
        self.mode_combo.addItem("新建数据集（从零生成）", "create")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(list(self._frames.keys()))
        if initial_dataset in self._frames:
            self.dataset_combo.setCurrentText(initial_dataset)
        self.column_edit = QLineEdit("新列")
        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 200000)
        self.row_spin.setValue(200)

        derive_form = QFormLayout()
        derive_form.addRow("数据集", self.dataset_combo)
        derive_form.addRow("新列名", self.column_edit)

        self.derive_box = QGroupBox("派生设置")
        self.derive_box.setLayout(derive_form)

        self.create_box = QGroupBox("新建设置")
        row_form = QFormLayout()
        self.new_name_edit = QLineEdit("函数曲线")
        self.row_spin.setSuffix(" 个点")
        row_form.addRow("数据集名称", self.new_name_edit)
        row_form.addRow("数据点数 n", self.row_spin)
        self.create_box.setLayout(row_form)

        # 列定义表（新建模式）
        self.column_table = QTableWidget(3, 2)
        self.column_table.setHorizontalHeaderLabels(["列名", "表达式"])
        self.column_table.horizontalHeader().setStretchLastSection(True)
        self.column_table.setItem(0, 0, QTableWidgetItem("x"))
        self.column_table.setItem(0, 1, QTableWidgetItem("linspace(0, 2*pi, n)"))
        self.column_table.setItem(1, 0, QTableWidgetItem("y"))
        self.column_table.setItem(1, 1, QTableWidgetItem("sin(x)"))
        self.column_table.setItem(2, 0, QTableWidgetItem("z"))
        self.column_table.setItem(2, 1, QTableWidgetItem("cos(x) * exp(-x/5)"))
        add_row = QPushButton("添加列")
        add_row.clicked.connect(lambda: self.column_table.insertRow(self.column_table.rowCount()))
        del_row = QPushButton("删除选中列")
        del_row.clicked.connect(self._remove_selected_columns)
        table_buttons = QHBoxLayout()
        table_buttons.addWidget(add_row)
        table_buttons.addWidget(del_row)
        table_buttons.addStretch(1)

        self.create_page = QWidget()
        create_layout = QVBoxLayout(self.create_page)
        create_layout.setContentsMargins(0, 0, 0, 0)
        create_layout.addWidget(self.create_box)
        create_layout.addWidget(QLabel("列定义（表达式可用 n、t、i、pi、e 以及已定义的列名）"))
        create_layout.addWidget(self.column_table, 1)
        create_layout.addLayout(table_buttons)

        # 表达式输入
        self.expression_edit = QPlainTextEdit("sin(x) * exp(-x/5)")
        self.expression_edit.setFont(QFont(_MONO.split(",")[0], 10))
        self.expression_edit.setMaximumHeight(90)
        self.expression_edit.textChanged.connect(lambda: self._timer_start())

        self.help_label = QLabel(EXPRESSION_HELP)
        self.help_label.setWordWrap(True)
        self.help_label.setProperty("muted", "true")
        self.help_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.preview = DataTableWidget(editable=False, max_rows=30)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(260)
        self._timer.timeout.connect(self._refresh_preview)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons

        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("模式"))
        top.addWidget(self.mode_combo, 1)
        layout.addLayout(top)
        layout.addWidget(self.derive_box)
        layout.addWidget(QLabel("表达式"))
        layout.addWidget(self.expression_edit)
        layout.addWidget(self.create_page)
        layout.addWidget(self.help_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.preview, 1)
        layout.addWidget(buttons)

        for widget in (self.dataset_combo, self.column_edit):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(lambda *_: self._timer_start())
            else:
                widget.textChanged.connect(lambda *_: self._timer_start())

        self._on_mode_changed()

    # -- 内部 -------------------------------------------------------------- #
    def _timer_start(self) -> None:
        self._timer.start()

    def _remove_selected_columns(self) -> None:
        rows = sorted({index.row() for index in self.column_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.column_table.removeRow(row)

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        is_derive = mode == "derive"
        self.derive_box.setVisible(is_derive)
        self.create_page.setVisible(not is_derive)
        self.expression_edit.setVisible(is_derive)
        self.help_label.setVisible(is_derive)
        self._refresh_preview()

    def _column_definitions(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for row in range(self.column_table.rowCount()):
            name_item = self.column_table.item(row, 0)
            expr_item = self.column_table.item(row, 1)
            name = name_item.text().strip() if name_item else ""
            expr = expr_item.text().strip() if expr_item else ""
            if name and expr:
                out.append((name, expr))
        return out

    def _refresh_preview(self) -> None:
        self._timer.stop()
        try:
            if self.mode_combo.currentData() == "derive":
                frame, values, name = self._compute_derived()
                preview = pd.DataFrame({name: values})
                self.status_label.setText(
                    f"共 {len(values)} 个值；前几个：{np.array2string(np.asarray(values[:6], dtype=float), precision=4)}"
                )
                self.status_label.setStyleSheet("")
            else:
                frame = self._compute_created()
                preview = frame
                self.status_label.setText(
                    f"将生成数据集「{self.new_name_edit.text().strip()}」：{frame.shape[0]} 行 × {frame.shape[1]} 列"
                )
                self.status_label.setStyleSheet("")
            self.preview.set_frame(preview.head(30))
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)
        except Exception as exc:
            self.status_label.setText(f"表达式有误：{exc}")
            self.status_label.setStyleSheet("color: #cf222e;")
            self.preview.set_frame(None)
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(False)

    def _compute_derived(self) -> tuple[pd.DataFrame, np.ndarray, str]:
        name = self.dataset_combo.currentText()
        frame = self._frames.get(name)
        if frame is None:
            raise ValueError("请选择一个数据集。")
        expression = self.expression_edit.toPlainText().strip()
        values = evaluate_expression(expression, frame)
        column = self.column_edit.text().strip() or "新列"
        return frame, values, column

    def _compute_created(self) -> pd.DataFrame:
        n = self.row_spin.value()
        namespace: dict[str, Any] = {"n": n, "t": np.arange(n, dtype=float), "i": np.arange(n, dtype=float)}
        definitions = self._column_definitions()
        if not definitions:
            raise ValueError("请至少定义一列（列名 + 表达式）。")
        data: dict[str, np.ndarray] = {}
        for name, expression in definitions:
            values = evaluate_expression(expression, pd.DataFrame({"n": np.arange(n)}), extra=namespace, target_length=n)
            data[name] = values
            namespace[name] = values
        return pd.DataFrame(data)

    # -- 结果 -------------------------------------------------------------- #
    def result(self) -> tuple[str, pd.DataFrame | np.ndarray, str]:
        """返回 ``(模式, 数据, 名称)``。

        * ``("derive", ndarray, 列名)``
        * ``("create", DataFrame, 数据集名)``
        """
        if self.mode_combo.currentData() == "derive":
            _frame, values, name = self._compute_derived()
            return "derive", values, name
        return "create", self._compute_created(), self.new_name_edit.text().strip() or "函数曲线"


# --------------------------------------------------------------------------- #
# 源码导出
# --------------------------------------------------------------------------- #
class ExportSourceDialog(QDialog):
    """把当前绘图配置导出为可独立运行的 matplotlib 源码。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        spec: PlotSpec | None = None,
        datasets: Mapping[str, pd.DataFrame] | None = None,
        csv_paths: Mapping[str, str] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("导出 Python 源码")
        self.resize(940, 700)
        self._spec = spec or PlotSpec()
        self._datasets = dict(datasets or {})
        self._csv_paths = dict(csv_paths or {})

        self.embed_check = QCheckBox("把数据内联进源码（自包含，推荐）")
        self.embed_check.setChecked(True)
        self.format_combo = QComboBox()
        self.format_combo.addItem("紧凑（推荐）", "compact")
        self.format_combo.addItem("每行一个数", "pretty")
        self.comments_check = QCheckBox("写中文分节注释")
        self.comments_check.setChecked(True)
        self.wrapper_check = QCheckBox("包装成 make_figure() 函数")
        self.rows_as_columns = QCheckBox("按列定义数据（列名 -> 数组）")
        self.rows_as_columns.setChecked(True)
        self.save_check = QCheckBox("在脚本末尾写 savefig 示例")
        self.save_check.setChecked(True)
        self.filename_edit = QLineEdit("figure.svg")
        self.precision_spin = QSpinBox()
        self.precision_spin.setRange(3, 15)
        self.precision_spin.setValue(6)

        options_form = QFormLayout()
        options_form.addRow("", self.embed_check)
        options_form.addRow("数组格式", self.format_combo)
        options_form.addRow("有效位数", self.precision_spin)
        options_form.addRow("", self.comments_check)
        options_form.addRow("", self.wrapper_check)
        options_form.addRow("", self.rows_as_columns)
        options_form.addRow("", self.save_check)
        options_form.addRow("示例输出文件", self.filename_edit)

        options_box = QGroupBox("生成选项")
        options_box.setLayout(options_form)

        self.code_edit = QPlainTextEdit()
        self.code_edit.setFont(QFont(_MONO.split(",")[0], 10))
        self.code_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_edit.setReadOnly(True)

        self.info_label = QLabel("")
        self.info_label.setProperty("muted", "true")
        self.info_label.setWordWrap(True)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(options_box)
        left_layout.addWidget(self.info_label)
        left_layout.addStretch(1)
        splitter.addWidget(left)
        splitter.addWidget(self.code_edit)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 560])

        copy_btn = QPushButton("复制源码")
        copy_btn.clicked.connect(self._copy)
        save_btn = QPushButton("保存为 .py")
        save_btn.clicked.connect(self._save)
        refresh_btn = QPushButton("重新生成")
        refresh_btn.clicked.connect(self.regenerate)

        button_row = QHBoxLayout()
        button_row.addWidget(refresh_btn)
        button_row.addStretch(1)
        button_row.addWidget(copy_btn)
        button_row.addWidget(save_btn)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.button(QDialogButtonBox.Close).setText("关闭")
        close_box.rejected.connect(self.reject)
        close_box.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addLayout(button_row)
        layout.addWidget(close_box)

        for widget in (self.embed_check, self.comments_check, self.wrapper_check,
                       self.save_check, self.rows_as_columns):
            widget.toggled.connect(self.regenerate)
        self.format_combo.currentIndexChanged.connect(self.regenerate)
        self.precision_spin.valueChanged.connect(self.regenerate)
        self.filename_edit.textChanged.connect(self.regenerate)

        self.regenerate()

    def options(self) -> "codesync.CodeGenOptions":
        return codesync.CodeGenOptions(
            embed_data=self.embed_check.isChecked(),
            csv_paths=dict(self._csv_paths) if not self.rows_as_columns.isChecked() else {},
            array_format=self.format_combo.currentData() or "compact",
            include_comments=self.comments_check.isChecked(),
            include_sample_save=self.save_check.isChecked(),
            output_filename=self.filename_edit.text().strip() or "figure.svg",
            function_wrapper=self.wrapper_check.isChecked(),
            precision=self.precision_spin.value(),
        )

    def source(self) -> str:
        return self.code_edit.toPlainText()

    def regenerate(self, *_: Any) -> None:
        try:
            code = codesync.generate_script(self._spec, self._datasets, self.options())
        except Exception as exc:
            self.code_edit.setPlainText(f"# 生成失败：{exc}\n")
            self.info_label.setText(f"生成失败：{exc}")
            return
        self.code_edit.setPlainText(code)
        lines = code.count("\n") + 1
        self.info_label.setText(
            f"已生成 {lines} 行源码，涉及数据集：{', '.join(self._spec.used_datasets()) or '（无）'}"
        )

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.source())

    def _save(self) -> None:
        base = self._spec.title.strip() or "plot"
        safe = "".join(ch for ch in base if ch.isalnum() or ch in "-_ ").strip() or "plot"
        path, _ = QFileDialog.getSaveFileName(self, "保存源码", f"{safe}.py", "Python 源码 (*.py)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.source())
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        QMessageBox.information(self, "已保存", f"源码已写入：\n{path}")


# --------------------------------------------------------------------------- #
# 结果查看
# --------------------------------------------------------------------------- #
class ResultViewDialog(QDialog):
    """通用结果查看窗口：用 LaTeX 渲染一个 ResultSheet。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        title: str = "计算结果",
        sheet: Any = None,
        style: ResultStyle | None = None,
        background: str = "#ffffff",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(940, 700)
        self._background = background

        self.view = LatexView(self, background=background)
        self.view.set_sheet(sheet, style)

        zoom_in = QPushButton("放大")
        zoom_out = QPushButton("缩小")
        zoom_reset = QPushButton("100%")
        zoom_in.clicked.connect(self.view.zoom_in)
        zoom_out.clicked.connect(self.view.zoom_out)
        zoom_reset.clicked.connect(self.view.reset_zoom)
        copy_png = QPushButton("复制图片")
        copy_png.clicked.connect(lambda: self.view.copy_image())
        export_svg = QPushButton("导出 SVG")
        export_svg.clicked.connect(self._export_svg)

        bar = QHBoxLayout()
        bar.addWidget(zoom_out)
        bar.addWidget(zoom_reset)
        bar.addWidget(zoom_in)
        bar.addStretch(1)
        bar.addWidget(copy_png)
        bar.addWidget(export_svg)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.button(QDialogButtonBox.Close).setText("关闭")
        close_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)
        layout.addWidget(close_box)

    def _export_svg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出 SVG", "result.svg", "SVG 矢量图 (*.svg)")
        if path:
            self.view.export_svg(path)


class TextInfoDialog(QDialog):
    """纯文本信息窗口（帮助、关于、公式语法说明等）。"""

    def __init__(self, parent: QWidget | None = None, *, title: str, text: str, width: int = 760, height: int = 620):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)
        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        edit.setFont(QFont(_MONO.split(",")[0], 10))
        edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.button(QDialogButtonBox.Close).setText("关闭")
        close_box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(edit, 1)
        layout.addWidget(close_box)


# --------------------------------------------------------------------------- #
# manim 动画
# --------------------------------------------------------------------------- #
class _ManimRenderThread(QThread):
    """在后台跑 manim，避免界面卡死。"""

    finished_signal = Signal(int, str, str, str)

    def __init__(self, script: str, scene: str, quality: str, workdir: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._script = script
        self._scene = scene
        self._quality = quality
        self._workdir = workdir

    def run(self) -> None:  # noqa: D102
        output = ""
        code = -1
        stdout = stderr = ""
        try:
            proc = manimgen.run_manim(
                self._script, self._scene, self._quality, cwd=self._workdir, timeout=1800
            )
            code = proc.returncode
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            output = _find_rendered_video(self._workdir)
        except Exception as exc:
            stderr = f"{type(exc).__name__}: {exc}"
        self.finished_signal.emit(code, stdout, stderr, output)


def _find_rendered_video(workdir: str) -> str:
    """在 manim 的输出目录里找最新渲染出来的 mp4。"""
    newest, newest_time = "", 0.0
    videos = os.path.join(workdir, "media", "videos")
    for root, _dirs, files in os.walk(videos):
        if "partial_movie_files" in root:
            continue
        for name in files:
            if not name.lower().endswith((".mp4", ".webm", ".mov", ".gif")):
                continue
            path = os.path.join(root, name)
            try:
                stamp = os.path.getmtime(path)
            except OSError:
                continue
            if stamp > newest_time:
                newest, newest_time = path, stamp
    return newest


def storyboard_document(plan: "manimgen.ManimScenePlan", style: ResultStyle | None = None) -> LatexDocument:
    """把 manim 分镜渲染成一份可预览的 LaTeX 文档（不需要装 manim）。"""
    from core.latex import LatexDocument, MathBlock, RuleBlock, TextBlock

    style = style or ResultStyle()
    document = LatexDocument(
        fontsize=style.fontsize,
        color=style.text_color,
        math_fontset=style.math_fontset,
        line_spacing=style.line_spacing,
        padding=style.padding,
        background=style.background,
    )
    document.add(
        TextBlock(
            text=f"manim 分镜 · {plan.title}",
            fontsize=style.heading_fontsize,
            bold=True,
            color=style.title_color,
            space_before=0.0,
            space_after=2.0,
        )
    )
    document.add(TextBlock(text=plan.subtitle, fontsize=style.note_fontsize, color=style.note_color,
                           space_before=0.0, space_after=4.0))
    document.add(RuleBlock(thickness=1.2, color=style.rule_color, space_before=0.0, space_after=8.0))

    for index, step in enumerate(plan.steps):
        heading = f"第 {index + 1} / {len(plan.steps)} 步"
        if step.text_note:
            heading += f"　{step.text_note}"
        document.add(
            TextBlock(
                text=heading,
                fontsize=style.title_fontsize,
                bold=True,
                color=style.title_color,
                space_before=0.0 if index == 0 else 10.0,
                space_after=1.0,
            )
        )
        if step.rows:
            document.add(
                MathBlock(
                    text=step.latex(name=plan.matrix_name),
                    fontsize=style.fontsize,
                    color=style.text_color,
                    space_before=1.0,
                    space_after=1.0,
                )
            )
        if step.math_note:
            document.add(
                MathBlock(
                    text=step.math_note,
                    fontsize=style.fontsize,
                    color=style.title_color,
                    space_before=1.0,
                    space_after=1.0,
                )
            )
        for line in step.extra_lines:
            document.add(
                MathBlock(text=line, fontsize=style.fontsize, color=style.text_color,
                          space_before=1.0, space_after=1.0)
            )
        if step.highlight_rows:
            rows = "、".join(str(r + 1) for r in step.highlight_rows)
            document.add(
                TextBlock(
                    text=f"（动画中高亮第 {rows} 行）",
                    fontsize=style.note_fontsize,
                    italic=True,
                    color=style.note_color,
                    indent=10.0,
                    space_before=0.0,
                    space_after=2.0,
                )
            )
    return document


class ManimDialog(QDialog):
    """把矩阵计算过程导出成 manim 动画：分镜预览 / 源码 / 渲染视频。"""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        matrix: Any = None,
        other: Any = None,
        label: str = "A",
        palette: Mapping[str, str] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("用 manim 渲染计算过程")
        self.resize(1180, 800)
        self._matrix = np.atleast_2d(np.asarray(matrix, dtype=float)) if matrix is not None else np.eye(3)
        self._other = np.atleast_2d(np.asarray(other, dtype=float)) if other is not None else None
        self._palette = dict(palette or {})
        self._plan: "manimgen.ManimScenePlan | None" = None
        self._thread: _ManimRenderThread | None = None
        self._started_at = 0.0

        # ---- 顶部选项 ---------------------------------------------------- #
        self.scene_combo = QComboBox()
        for key, name in manimgen.available_scenes().items():
            self.scene_combo.addItem(name, key)
        self.quality_combo = QComboBox()
        for key, (name, resolution, fps, _flag) in manimgen.QUALITIES.items():
            self.quality_combo.addItem(name, key)
        self.quality_combo.setCurrentIndex(1)
        self.background_combo = QComboBox()
        self.background_combo.addItem("深色背景", "dark")
        self.background_combo.addItem("浅色背景", "light")
        self.hold_spin = QDoubleSpinBox()
        self.hold_spin.setRange(0.1, 10.0)
        self.hold_spin.setSingleStep(0.1)
        self.hold_spin.setValue(0.8)
        self.hold_spin.setSuffix(" 秒/步")
        self.index_check = QCheckBox("显示步骤序号")
        self.index_check.setChecked(True)
        self.loop_check = QCheckBox("结尾循环回第一步")

        options = QHBoxLayout()
        options.addWidget(QLabel("动画"))
        options.addWidget(self.scene_combo, 2)
        options.addWidget(QLabel("画质"))
        options.addWidget(self.quality_combo, 2)
        options.addWidget(self.background_combo, 1)
        options.addWidget(self.hold_spin)
        options.addWidget(self.index_check)
        options.addWidget(self.loop_check)
        options.addStretch(1)

        # ---- 状态 -------------------------------------------------------- #
        available, message = manimgen.manim_status()
        self._manim_available = available
        self.status_label = QLabel(message)
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("muted", "true")

        # ---- Tab 1：分镜预览 --------------------------------------------- #
        self.preview = LatexView(self, background=self._palette.get("latex_bg", "#ffffff"))
        preview_bar = QHBoxLayout()
        out_btn = QPushButton("缩小")
        out_btn.clicked.connect(self.preview.zoom_out)
        reset_btn = QPushButton("100%")
        reset_btn.clicked.connect(self.preview.reset_zoom)
        in_btn = QPushButton("放大")
        in_btn.clicked.connect(self.preview.zoom_in)
        preview_bar.addWidget(out_btn)
        preview_bar.addWidget(reset_btn)
        preview_bar.addWidget(in_btn)
        preview_bar.addStretch(1)
        preview_bar.addWidget(QLabel("这是动画会依次展示的每一步（用本程序的 LaTeX 引擎渲染，无需安装 manim）"))
        preview_page = QWidget()
        preview_layout = QVBoxLayout(preview_page)
        preview_layout.setContentsMargins(0, 6, 0, 0)
        preview_layout.addLayout(preview_bar)
        preview_layout.addWidget(self.preview, 1)

        # ---- Tab 2：源码 ------------------------------------------------- #
        self.code_edit = QPlainTextEdit()
        self.code_edit.setFont(QFont(_MONO.split(",")[0], 10))
        self.code_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.code_edit.setReadOnly(True)
        code_bar = QHBoxLayout()
        copy_btn = QPushButton("复制源码")
        copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.code_edit.toPlainText()))
        save_btn = QPushButton("保存为 .py…")
        save_btn.clicked.connect(self._save_script)
        code_bar.addWidget(copy_btn)
        code_bar.addWidget(save_btn)
        code_bar.addStretch(1)
        code_page = QWidget()
        code_layout = QVBoxLayout(code_page)
        code_layout.setContentsMargins(0, 6, 0, 0)
        code_layout.addLayout(code_bar)
        code_layout.addWidget(self.code_edit, 1)

        # ---- Tab 3：渲染 ------------------------------------------------- #
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont(_MONO.split(",")[0], 9))
        self.render_btn = QPushButton("开始渲染视频")
        self.render_btn.setProperty("accent", "true")
        self.render_btn.clicked.connect(self._render)
        self.render_btn.setEnabled(available)
        self.open_dir_btn = QPushButton("打开输出目录")
        self.open_dir_btn.clicked.connect(self._open_output_dir)
        self.open_video_btn = QPushButton("播放已渲染视频")
        self.open_video_btn.clicked.connect(self._open_video)
        self.open_video_btn.setEnabled(False)
        render_bar = QHBoxLayout()
        render_bar.addWidget(self.render_btn)
        render_bar.addWidget(self.open_video_btn)
        render_bar.addWidget(self.open_dir_btn)
        render_bar.addStretch(1)
        render_page = QWidget()
        render_layout = QVBoxLayout(render_page)
        render_layout.setContentsMargins(0, 6, 0, 0)
        render_layout.addLayout(render_bar)
        render_layout.addWidget(self.log_edit, 1)

        self.tabs = QTabWidget()
        self.tabs.addTab(preview_page, "分镜预览")
        self.tabs.addTab(code_page, "manim 源码")
        self.tabs.addTab(render_page, "渲染视频")

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.button(QDialogButtonBox.Close).setText("关闭")
        close_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(options)
        layout.addWidget(self.status_label)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(close_box)

        for widget in (self.scene_combo, self.quality_combo, self.background_combo):
            widget.currentIndexChanged.connect(lambda *_: self.regenerate())
        for widget in (self.index_check, self.loop_check):
            widget.toggled.connect(lambda *_: self.regenerate())
        self.hold_spin.valueChanged.connect(lambda *_: self.regenerate())

        self.regenerate()
        if not available:
            self.tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------ #
    def _options(self) -> "manimgen.ManimOptions":
        return manimgen.ManimOptions(
            quality=self.quality_combo.currentData() or "medium",
            background=self.background_combo.currentData() or "dark",
            hold_duration=float(self.hold_spin.value()),
            show_index=self.index_check.isChecked(),
            loop=self.loop_check.isChecked(),
        )

    def regenerate(self, *_: Any) -> None:
        """按当前选项重新规划分镜并刷新预览与源码。"""
        kind = self.scene_combo.currentData() or "elimination"
        try:
            plan = manimgen.plan_scene(kind, self._matrix, self._other)
            code = manimgen.generate_scene(plan, self._options())
        except Exception as exc:
            self.preview.set_message(f"生成失败：{exc}", muted=False)
            self.code_edit.setPlainText(f"# 生成失败：{exc}")
            return
        self._plan = plan
        self.code_edit.setPlainText(code)
        style = ResultStyle(
            fontsize=12.0,
            title_color=self._palette.get("latex_title", "#1d4ed8"),
            text_color=self._palette.get("latex_text", "#111827"),
            note_color=self._palette.get("latex_note", "#6b7280"),
            rule_color=self._palette.get("latex_rule", "#d8dee9"),
            background=self._palette.get("latex_bg", "#ffffff"),
        )
        self.preview.set_document(storyboard_document(plan, style))
        self.log_edit.setPlainText(
            f"动画类型：{plan.title}\n共 {len(plan.steps)} 步\n"
            f"场景类名：{plan.scene_name}\n\n"
            f"{manimgen.manim_status()[1]}\n"
        )

    def _save_script(self) -> None:
        if self._plan is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 manim 源码", f"{self._plan.scene_name.lower()}.py", "Python 源码 (*.py)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.code_edit.toPlainText())
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        QMessageBox.information(self, "已保存", f"manim 源码已写入：\n{path}")

    # ------------------------------------------------------------------ #
    def _workdir(self) -> str:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
        stamp = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(base, "TableDrawer", "manim", stamp)
        os.makedirs(path, exist_ok=True)
        return path

    def _render(self) -> None:
        if self._plan is None or self._thread is not None:
            return
        workdir = self._workdir()
        script = os.path.join(workdir, f"{self._plan.scene_name.lower()}.py")
        try:
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(self.code_edit.toPlainText())
        except OSError as exc:
            QMessageBox.warning(self, "无法写入脚本", str(exc))
            return

        self._output_dir = workdir
        self.render_btn.setEnabled(False)
        self.render_btn.setText("渲染中…")
        self.log_edit.appendPlainText(f"工作目录：{workdir}")
        self.log_edit.appendPlainText("命令：" + " ".join(manimgen.render_command(script, self._plan.scene_name, self._options().quality)))
        self.log_edit.appendPlainText("manim 正在渲染，请稍候（首次运行可能需要几分钟）…\n")

        self._thread = _ManimRenderThread(script, self._plan.scene_name, self._options().quality, workdir, self)
        self._thread.finished_signal.connect(self._on_rendered)
        self._thread.start()

    def _on_rendered(self, code: int, stdout: str, stderr: str, output: str) -> None:
        self._thread = None
        self.render_btn.setEnabled(True)
        self.render_btn.setText("开始渲染视频")
        if stdout:
            self.log_edit.appendPlainText(stdout[-4000:])
        if code == 0 and output:
            self._video_path = output
            self.open_video_btn.setEnabled(True)
            self.log_edit.appendPlainText(f"\n✅ 渲染完成：{output}")
        else:
            self.log_edit.appendPlainText(f"\n❌ 渲染失败（退出码 {code}）")
            if stderr:
                self.log_edit.appendPlainText(stderr[-4000:])
            if not self._manim_available:
                self.log_edit.appendPlainText("\n" + manimgen.INSTALL_HINT)
        self.tabs.setCurrentIndex(2)

    def _open_output_dir(self) -> None:
        path = getattr(self, "_output_dir", "") or os.path.expanduser("~")
        _open_in_explorer(path)

    def _open_video(self) -> None:
        path = getattr(self, "_video_path", "")
        if path:
            _open_in_explorer(path)

    def closeEvent(self, event):  # noqa: N802
        if self._thread is not None and self._thread.isRunning():
            answer = QMessageBox.question(
                self,
                "正在渲染",
                "manim 还在渲染，现在关闭会在后台继续跑完。确定关闭窗口吗？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self._thread.wait(3000)
        super().closeEvent(event)


def _open_in_explorer(path: str) -> None:
    """在系统文件管理器里定位到该路径。"""
    try:
        if not os.path.exists(path):
            path = os.path.dirname(path)
        if sys.platform.startswith("win"):
            if os.path.isdir(path):
                os.startfile(path)  # noqa: S606
            else:
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path if os.path.isdir(path) else os.path.dirname(path)])
        else:
            subprocess.Popen(["xdg-open", path if os.path.isdir(path) else os.path.dirname(path)])
    except Exception:
        pass
