"""TableDrawer 核心层：数据、绘图、矩阵运算与 LaTeX 渲染引擎。

该包不依赖 PySide6，可单独用于脚本 / 测试::

    from core.dataset import load_csv
    from core.plotting import build_figure
    from core.matrixops import run

子模块采用**惰性导入**：``from core import plotting`` 可以正常工作，
但 ``import core`` 本身不会一次性把 numpy / pandas / matplotlib 全部加载进来。
"""

from __future__ import annotations

import importlib

__all__ = [
    "numberfmt",
    "spec",
    "mplsetup",
    "dataset",
    "plotting",
    "matrixops",
    "codesync",
    "latex",
]

__version__ = "1.0.0"


def __getattr__(name: str):
    """按需导入子模块（PEP 562）。"""
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
