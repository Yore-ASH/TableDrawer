"""PySide6 界面层。

子模块：

* :mod:`ui.theme`    —— 配色板与 QSS 样式表
* :mod:`ui.widgets`  —— matplotlib 画布、LaTeX 结果视图、数据表、矩阵编辑器
* :mod:`ui.dialogs`  —— CSV 导入、手动录入、表达式、源码导出等对话框
* :mod:`ui.data_tab` / :mod:`ui.plot_tab` / :mod:`ui.matrix_tab`
* :mod:`ui.main_window` —— 主窗口

所有子模块都惰性导入，因此 ``import ui`` 不会强制加载 Qt 控件。
"""

from __future__ import annotations

import importlib
import os

os.environ.setdefault("QT_API", "pyside6")

__all__ = [
    "MainWindow",
    "theme",
    "widgets",
    "dialogs",
    "data_tab",
    "plot_tab",
    "matrix_tab",
    "main_window",
]

_SUBMODULES = {"theme", "widgets", "dialogs", "data_tab", "plot_tab", "matrix_tab", "main_window"}


def __getattr__(name: str):
    """按需导入主窗口或子模块（PEP 562）。"""
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name == "MainWindow":
        from .main_window import MainWindow

        globals()["MainWindow"] = MainWindow
        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
