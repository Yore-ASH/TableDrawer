"""matplotlib 绘图引擎：把 :class:`core.spec.PlotSpec` 渲染为 Figure。

设计约定
--------
* 本模块**不**调用 :func:`matplotlib.use`，也**不**导入 ``matplotlib.pyplot``：
  图对象用 ``Figure(...)`` + ``fig.add_subplot(...)`` 手工创建，
  因此界面层可以把返回的 Figure 直接塞进 Qt 画布（FigureCanvasQTAgg）。
* 主题通过 ``matplotlib.rc_context`` 局部生效，绘图结束后全局 rcParams 自动还原。
* 所有可恢复的问题都写入返回的 ``warnings``（简体中文），**从不抛异常**。

主要入口::

    fig, notes = build_figure(spec, datasets)
    svg_text = figure_to_svg(fig)
    save_figure(fig, "out.png")
"""

from __future__ import annotations

import io
import math
import os
import re
import warnings as _warnings
from contextlib import contextmanager, suppress
from typing import Any, Iterator, Mapping

import matplotlib
import matplotlib.style as mstyle
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

from core.mplsetup import cjk_font_family, setup_matplotlib
from core.numberfmt import format_number
from core.spec import KINDS_2D, KINDS_3D, PlotSpec, SeriesSpec

__all__ = [
    "KIND_HELP",
    "KIND_PORTRAIT",
    "TEXTLIKE_KINDS",
    "GRID_KINDS",
    "COLORBAR_KINDS",
    "VECTOR_KINDS",
    "DEFAULT_THEMES",
    "THEME_ALIASES",
    "EmptySeries",
    "to_grid",
    "summary_text",
    "theme_rcparams",
    "build_figure",
    "resolve_frame",
    "available_columns",
    "figure_to_svg",
    "save_figure",
    "default_series",
    "default_spec",
    "has_cjk",
    "split_cjk_math",
]

#: 导入时完成一次全局初始化（无中文字体环境下也不能让整个模块导入失败）。
with suppress(Exception):  # pragma: no cover - 取决于运行环境
    setup_matplotlib()

#: 没有 ``matplotlib.style.library`` 条目、直接使用默认 rcParams 的主题名
DEFAULT_THEMES: frozenset[str] = frozenset({"light", ""})

#: 界面主题名 -> matplotlib 样式名（``spec.py`` 的 ``THEMES`` 里有些键不是样式名）
THEME_ALIASES: dict[str, str] = {"dark": "dark_background"}

# --------------------------------------------------------------------------- #
# 常量表
# --------------------------------------------------------------------------- #
#: 每个图表类型的中文一句话说明 + 需要的列
KIND_HELP: dict[str, str] = {
    "line": "折线图：x 为横坐标（留空用行号），y 可填多列，每个 y 列画一条线",
    "scatter": "散点图：需要 x、y 列（留空用行号），y 可填多列",
    "bar": "柱形图：需要 x、y 列，y 可填多列；多条 y 会自动并排分组",
    "barh": "条形图：需要 x、y 列，沿水平方向并排绘制；多条 y 会自动并排分组",
    "stem": "火柴杆图：需要 x、y 列，适合离散序列",
    "step": "阶梯图：需要 x、y 列，相邻点之间以阶梯连接",
    "fill": "面积填充图：需要 x、y 列，在曲线与基线之间填充；多条 y 会依次叠加",
    "errorbar": "误差棒图：需要 x、y 列，并在「误差列」中指定误差列（留空自动找同名的 _err 列）",
    "hist": "直方图：只需 y 列（可多列叠加），bins 控制分组数",
    "box": "箱线图：只需 y 列，每个 y 列一个箱体",
    "pie": "饼图：用第一个 y 列作为扇形数值，x 列作为扇形标签",
    "quiver": "向量图：需要 x、y 为起点坐标，u、v 为分量",
    "contour": "等高线图：需要 x、y、z 三列，散点会自动插值成网格",
    "heatmap": "热力图：需要 x、y、z 三列，散点会自动插值成网格",
    "line3d": "三维折线图：需要 x、y、z 三列",
    "scatter3d": "三维散点图：需要 x、y、z 三列",
    "surface": "三维曲面图：需要 x、y、z 三列，z 按 x/y 的网格取值（支持长表）",
    "wireframe": "三维线框图：需要 x、y、z 三列，z 按 x/y 的网格取值（支持长表）",
    "bar3d": "三维柱形图：需要 x、y、z 三列",
    "quiver3d": "三维向量场：需要 x、y、z 为起点，u、v、w 为分量",
    "contour3d": "三维等高线图：需要 x、y、z 三列，z 按 x/y 的网格取值（支持长表）",
    "trisurf": "三角剖分曲面：需要 x、y、z 三列散点，自动三角剖分",
}

#: 图表类型 -> 需要在坐标系里填写的列用途（用于校验与警告文案）
KIND_PORTRAIT: dict[str, tuple[str, ...]] = {
    "line": ("x", "y"),
    "scatter": ("x", "y"),
    "bar": ("x", "y"),
    "barh": ("x", "y"),
    "stem": ("x", "y"),
    "step": ("x", "y"),
    "fill": ("x", "y"),
    "errorbar": ("x", "y"),
    "hist": ("y",),
    "box": ("y",),
    "pie": ("x", "y"),
    "quiver": ("x", "y", "u", "v"),
    "contour": ("x", "y", "z"),
    "heatmap": ("x", "y", "z"),
    "line3d": ("x", "y", "z"),
    "scatter3d": ("x", "y", "z"),
    "surface": ("x", "y", "z"),
    "wireframe": ("x", "y", "z"),
    "bar3d": ("x", "y", "z"),
    "quiver3d": ("x", "y", "z", "u", "v", "w"),
    "contour3d": ("x", "y", "z"),
    "trisurf": ("x", "y", "z"),
}

#: 需要把 (x, y, z) 三元组转成网格的图表类型
GRID_KINDS: frozenset[str] = frozenset({"surface", "wireframe", "contour3d", "contour", "heatmap"})

#: ``spec.colorbar`` 真正生效的图表类型
COLORBAR_KINDS: frozenset[str] = frozenset({"surface", "wireframe", "contour", "heatmap", "contour3d"})

#: 向量图（需要 u/v/w 分量）
VECTOR_KINDS: frozenset[str] = frozenset({"quiver", "quiver3d"})

#: 不使用 x 轴的图表类型
NO_X_KINDS: frozenset[str] = frozenset({"hist", "box", "pie"})

#: 绘制完成后坐标轴为「文本/分类」性质、不应再做 xlim/yscale 的图表类型
TEXTLIKE_KINDS: frozenset[str] = frozenset({"pie", "bar", "barh"})

#: 散点类图最多绘制的点数（超出后等间隔抽样，避免界面卡死）
_MAX_SCATTER_POINTS = 200_000

#: 三维曲面/线框的网格上限（超出后等间隔抽样）
_MAX_GRID_CELLS = 400_000

#: ``fill`` 自动分配颜色时填充色循环
_FILL_CYCLE: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
)


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class EmptySeries(Exception):
    """内部信号：某条序列没有可绘制的数据（由 ``_draw_series`` 捕获并转为警告）。"""


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def _finite(values: Any) -> np.ndarray:
    """转成浮点数组并保留有穷元素，同时返回有效掩码。"""
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    return arr, mask


def _numeric(series: pd.Series) -> np.ndarray:
    """把一列转成浮点数组（文本列按出现顺序编码为整数）。"""
    values = series.to_numpy()
    if values.dtype.kind in "biufc":
        return values.astype(float)
    if values.dtype.kind in "mM":  # datetime64 / timedelta64 -> 数值时间轴
        return values.astype("datetime64[ns]").astype("int64").astype(float)
    codes, _ = pd.factorize(series, sort=False)
    return codes.astype(float)


def _column_values(frame: pd.DataFrame, name: str) -> np.ndarray:
    """取出列并转成浮点数组；列不存在或全为空时抛 :class:`EmptySeries`。"""
    if name not in frame.columns:
        raise EmptySeries(f"数据集中找不到列「{name}」")
    raw = frame[name]
    if raw.isna().all():
        raise EmptySeries(f"列「{name}」全为空值")
    values = _numeric(raw)
    if values.size == 0:
        raise EmptySeries(f"列「{name}」没有任何数据")
    return values


def _resolve_x(frame: pd.DataFrame, name: str | None, length: int) -> np.ndarray:
    """x 轴：列名为空 / 列不存在时回退为行号。"""
    if name and name in frame.columns:
        raw = frame[name]
        if not raw.isna().all():
            values = _numeric(raw)
            if values.size == length:
                return values
            if values.size < length:
                padded = np.full(length, np.nan)
                padded[: values.size] = values
                return padded
            return values[:length]
    return np.arange(length, dtype=float)


def _column_label(name: str) -> str:
    """把列名包装成图例安全的标签（逗号会与多列分隔符冲突）。"""
    return str(name).replace(",", "\\,")


def _series_labels(series: SeriesSpec, columns: list[str]) -> list[str]:
    """为每条 y 列生成图例标签；``series.label`` 非空时优先。"""
    dataset = str(series.dataset or "")
    base = str(series.label).strip()
    if base:
        dataset = base
    if dataset:
        return [f"{dataset}.{_column_label(c)}" for c in columns]
    return [_column_label(c) for c in columns]


def _finite_pairs(*arrays: np.ndarray) -> np.ndarray:
    """返回所有数组都取到有穷值的公共掩码。"""
    mask: np.ndarray | None = None
    for array in arrays:
        current = np.isfinite(np.asarray(array, dtype=float))
        mask = current if mask is None else (mask & current)
    if mask is None:
        return np.zeros(0, dtype=bool)
    return mask


def _has_finite(values: Any) -> bool:
    """数组里是否至少有一个有穷数值。"""
    try:
        return bool(np.any(np.isfinite(np.asarray(values, dtype=float))))
    except (TypeError, ValueError):
        return False


def _color_or_none(series: SeriesSpec, index: int = 0) -> str | None:
    """非空颜色才返回，多列时后续列由 matplotlib 自动配色。"""
    color = str(series.color or "").strip()
    if not color:
        return None
    if index == 0:
        return color
    return color if mcolors.is_color_like(color) else None


def _points_cap(x: np.ndarray, y: np.ndarray, notes: list[str], where: str) -> tuple[np.ndarray, np.ndarray]:
    """点数过多时等间隔抽样，并给出中文提示。"""
    if x.size <= _MAX_SCATTER_POINTS:
        return x, y
    step = int(math.ceil(x.size / _MAX_SCATTER_POINTS))
    notes.append(f"{where}：数据点过多（{x.size} 个），已每隔 {step} 个点抽样显示。")
    return x[::step], y[::step]


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """DataFrame 中可当作数值绘制的列（数值列、时间列、以及能转成数值的其它列）。"""
    out: list[str] = []
    for name in frame.columns:
        dtype = frame[name].dtype
        if dtype.kind in "biufcMm":
            out.append(str(name))
            continue
        try:
            pd.to_numeric(frame[name], errors="raise")
        except (TypeError, ValueError, OverflowError):
            continue
        out.append(str(name))
    return out


# --------------------------------------------------------------------------- #
# 网格化
# --------------------------------------------------------------------------- #
def to_grid(x: Any, y: Any, z: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 (x, y, z) 三元组整理成规则网格，返回 ``(X, Y, Z)``（行优先，y 为行）。

    三种情形：

    * **规则长表**：x、y 恰好是唯一值的笛卡尔积（``len(x) == len(ux) * len(uy)``）时，
      按 ``np.unique`` 排序后重排成网格，结果精确无误差；
    * **近网格散点**（每行仍有多个 x）：先对每一行做一维 ``np.interp`` 得到近似网格；
    * **完全散乱的数据**：改用 matplotlib 自带的 ``LinearTriInterpolator``
      （底层是纯 NumPy）做 Delaunay 三角剖分线性插值，凸包之外的格子用最近邻传播补齐。

    不依赖 scipy，不修改入参。数据太少或退化时返回含 ``np.nan`` 的网格，
    由调用方决定提示文案（本模块会在 ``warnings`` 里说明已降级）。
    """
    xa = np.asarray(x, dtype=float).reshape(-1)
    ya = np.asarray(y, dtype=float).reshape(-1)
    za = np.asarray(z, dtype=float).reshape(-1)
    size = min(xa.size, ya.size, za.size)
    if size == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    xa, ya, za = xa[:size], ya[:size], za[:size]

    ux = np.unique(xa[np.isfinite(xa)])
    uy = np.unique(ya[np.isfinite(ya)])
    if ux.size * uy.size != xa.size or ux.size < 2 or uy.size < 2:
        return _grid_scattered(xa, ya, za)

    ix = np.searchsorted(ux, xa)
    iy = np.searchsorted(uy, ya)
    # 行优先：flat = iy * len(ux) + ix
    flat = iy.astype(np.intp) * ux.size + ix.astype(np.intp)
    if np.unique(flat).size != flat.size:  # 有重复点，无法一一对应
        return _grid_scattered(xa, ya, za)
    gz = np.full(uy.size * ux.size, np.nan, dtype=float)
    gz[flat] = za
    gx, gy = np.meshgrid(ux, uy)
    return gx, gy, gz.reshape(uy.size, ux.size)


def _grid_scattered(xa: np.ndarray, ya: np.ndarray, za: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """非规则长表的网格化：能按行插值就按行插值，否则三角剖分插值。"""
    ux = np.unique(xa)
    uy = np.unique(ya)
    gx, gy = np.meshgrid(ux, uy)
    if ux.size < 2 or uy.size < 2:
        return gx, gy, np.full((uy.size, ux.size), np.nan)

    good = np.isfinite(za)
    px, py, pz = xa[good], ya[good], za[good]
    if px.size == 0:
        return gx, gy, np.full(gx.shape, np.nan)

    gz = np.full(gx.shape, np.nan, dtype=float)
    row_counts: list[int] = []
    for row, yv in enumerate(uy):
        row_mask = py == yv
        if not row_mask.any():
            row_counts.append(0)
            continue
        xs = px[row_mask]
        zs = pz[row_mask]
        order = np.argsort(xs)
        xs, zs = xs[order], zs[order]
        row_counts.append(int(np.unique(xs).size))
        gz[row, :] = zs[0] if xs.size == 1 else np.interp(ux, xs, zs)

    # 每行平均不足 2 个不同 x 时，说明数据不是按行排列的散点，
    # 按行插值只是粗略近似，改走三角剖分线性插值。
    if float(np.median(row_counts)) < 2.0:
        return gx, gy, _interp_triangulated(gx, gy, px, py, pz)
    if np.isfinite(gz).any() and not np.isfinite(gz).all():
        gz = _nearest_fill(gz)
    return gx, gy, gz


def _interp_triangulated(
    gx: np.ndarray,
    gy: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
) -> np.ndarray:
    """用 Delaunay 三角剖分 + 线性插值填满整个网格（纯 NumPy，不依赖 scipy）。

    直接复用 matplotlib 自带的 :class:`~matplotlib.tri.LinearTriInterpolator`
    （它的底层就是 NumPy），凸包之外的格子交由最近邻传播补齐。
    """
    from matplotlib import tri as mtri

    try:
        triangulation = mtri.Triangulation(px, py)
        interpolator = mtri.LinearTriInterpolator(triangulation, pz)
        values = interpolator(gx, gy)
    except (ValueError, RuntimeError, IndexError):
        return np.full(gx.shape, np.nan)

    gz = np.ma.filled(np.ma.masked_invalid(np.asarray(values, dtype=float)), np.nan)
    if np.isfinite(gz).any() and not np.isfinite(gz).all():
        gz = _nearest_fill(gz)
    return gz


def _nearest_fill(grid: np.ndarray) -> np.ndarray:
    """迭代传播最近的有效值，填满网格中的 NaN。"""
    filled = grid.copy()
    for _ in range(int(max(grid.shape)) + 1):
        holes = ~np.isfinite(filled)
        if not holes.any():
            break
        padded = np.pad(filled, 1, mode="constant", constant_values=np.nan)
        stack = np.stack([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)  # 邻居全为空时 nanmean 会告警
            neighbour = np.nanmean(stack, axis=0)
        filled = np.where(holes & np.isfinite(neighbour), neighbour, filled)
    return filled


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
def resolve_frame(series: SeriesSpec, df: pd.DataFrame) -> pd.DataFrame:
    """在 ``df`` 上应用 ``series.expr``（若有）派生额外列后返回新 DataFrame。

    表达式可用的变量：``x``（x 列或行号）、``y``（第一个 y 列或 0）、``i``（行号），
    以及 ``np`` 与数据集的全部列名。派生结果写入 ``series.y`` 的第一列名下
    （该列为空时写入 ``__expr__``）。入参 ``df`` 不会被修改。
    """
    frame = df.copy()
    expr = str(getattr(series, "expr", "") or "").strip()
    if not expr:
        return frame

    columns = series.y_columns()
    first = columns[0] if columns else ""
    if series.x and series.x in frame.columns:
        x_values = _numeric(frame[series.x])
    else:
        x_values = np.arange(len(frame), dtype=float)
    if first and first in frame.columns:
        y_values = _numeric(frame[first])
    else:
        y_values = np.zeros(len(frame), dtype=float)

    namespace: dict[str, Any] = {"np": np, "i": np.arange(len(frame), dtype=float), "x": x_values, "y": y_values}
    for name in frame.columns:
        if name != first:
            namespace[str(name)] = frame[name].to_numpy()
    namespace[first or "y"] = y_values

    try:
        result = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307 - 用户显式输入的数学表达式
    except Exception:
        return frame

    values = np.asarray(result, dtype=object)
    if values.ndim == 0:
        values = np.full(len(frame), values.item(), dtype=object)
    values = values.reshape(-1)
    if values.size != len(frame):
        return frame
    target = first or "__expr__"
    frame[target] = values
    return frame


def _prepare_frames(
    spec: PlotSpec,
    datasets: Mapping[str, pd.DataFrame],
    notes: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[int, str]]:
    """解析每条序列引用的数据集，套用表达式，返回 ``(数据集缓存, 序列序号->键)``。"""
    cache: dict[str, pd.DataFrame] = {}
    keys: dict[int, str] = {}
    name_of: dict[str, pd.DataFrame] = {str(k): v for k, v in datasets.items()}
    for index, series in enumerate(spec.series):
        key = str(series.dataset or "")
        if not key:
            if name_of:
                key = next(iter(name_of))
                series.dataset = key
            else:
                notes.append(f"第 {index + 1} 条序列没有指定数据集，已跳过。")
                continue
        if key not in name_of:
            notes.append(f"第 {index + 1} 条序列引用的数据集「{key}」不存在，已跳过。")
            continue
        if key in cache:
            keys[index] = key
            continue
        frame = name_of[key]
        if frame is None or not isinstance(frame, pd.DataFrame):
            notes.append(f"数据集「{key}」不是有效的数据表，已跳过。")
            continue
        keys[index] = key
        if str(series.expr or "").strip():
            resolved = resolve_frame(series, frame)
            if resolved is frame:
                notes.append(f"数据集「{key}」的表达式「{series.expr}」无法计算，已按原始列绘制。")
                resolved = frame.copy()
            cache[key] = resolved
        else:
            cache[key] = frame
    return cache, keys


def _warn_missing_columns(frame: pd.DataFrame, series: SeriesSpec, notes: list[str], where: str) -> None:
    """把引用了但数据集中不存在的列记入 warnings。"""
    for column in series.used_columns():
        if column not in frame.columns:
            notes.append(f"{where}：找不到列「{column}」")


# --------------------------------------------------------------------------- #
# 主题
# --------------------------------------------------------------------------- #
def theme_rcparams(theme: str) -> dict[str, Any]:
    """取主题对应的 rcParams 字典；``"light"`` / 未知主题返回空字典（用默认样式）。

    ``THEMES`` 里的 ``"dark"`` 在 ``matplotlib.style.library`` 中并不存在，
    实际对应的是 ``"dark_background"``，这里做一次别名映射。
    """
    name = str(theme or "").strip()
    if name in DEFAULT_THEMES:
        return {}
    name = THEME_ALIASES.get(name, name)
    try:
        return dict(mstyle.library[name])
    except KeyError:
        return {}


@contextmanager
def _theme_context(spec: PlotSpec) -> Iterator[None]:
    """用 ``matplotlib.rc_context`` 包住整个绘图过程（结束自动还原）。"""
    rc = theme_rcparams(spec.theme)
    family = str(spec.font_family or "").strip()
    if family:
        # 只在本序列显式指定字体时才覆盖；否则沿用 core.mplsetup 配置好的
        # 全局回退链（西文 → 中文 → 兜底），这样「Times New Roman + 宋体」
        # 这类设置才能同时作用于英文与中文。
        rc = dict(rc)
        rc["font.family"] = ["sans-serif"]
        rc["font.sans-serif"] = [family, "DejaVu Sans"]
    with matplotlib.rc_context(rc):
        yield


# --------------------------------------------------------------------------- #
# 内部绘图
# --------------------------------------------------------------------------- #
def _error_bars(frame: pd.DataFrame, series: SeriesSpec, y_columns: list[str], column: str) -> tuple[Any, bool]:
    """解析误差列：``a,b`` 一一对应 y 列；单列则复用到所有 y 列。"""
    if not column:
        return None, True
    names = [c.strip() for c in str(column).split(",") if c.strip()]
    missing = [n for n in names if n not in frame.columns]
    if missing:
        return None, False
    arrays = [_numeric(frame[n]) for n in names]
    if len(arrays) == 1:
        return arrays[0], True
    if len(arrays) == len(y_columns):
        return arrays, True
    return arrays[0], True


def _draw_2d(ax: Any, spec: PlotSpec, series: SeriesSpec, frame: pd.DataFrame, notes: list[str], where: str) -> None:
    """二维图表的分派绘制。"""
    kind = str(series.kind or "line").lower()
    if kind not in KINDS_2D:
        notes.append(f"{where}：二维投影不支持图表类型「{kind}」，已按折线图绘制。")
        kind = "line"

    y_columns = series.y_columns()
    marker = str(series.marker or "")
    linestyle = str(series.linestyle if series.linestyle is not None else "-")
    alpha = float(np.clip(float(series.alpha or 1.0), 0.0, 1.0))
    lw = _coerce_positive(series.linewidth, 2.0)
    ms = _coerce_positive(series.markersize, 6.0)
    zorder = int(series.zorder)
    labels = _series_labels(series, y_columns)

    if kind in NO_X_KINDS:
        x_values = np.array([])
    else:
        x_values = _resolve_x(frame, series.x, len(frame))

    if kind == "hist":
        if not y_columns:
            raise EmptySeries("直方图需要至少一个 y 列")
        for index, column in enumerate(y_columns):
            data = _column_values(frame, column)
            data = data[np.isfinite(data)]
            if data.size == 0:
                notes.append(f"{where}：列「{column}」没有有效数值，已跳过。")
                continue
            with suppress(Exception):
                ax.hist(
                    data,
                    bins=max(1, int(_coerce_positive(series.bins, 20.0))),
                    density=bool(series.density),
                    cumulative=bool(series.cumulative),
                    orientation="horizontal" if series.horizontal else "vertical",
                    histtype="stepfilled" if series.filled else "step",
                    alpha=alpha,
                    label=labels[index] if index < len(labels) else column,
                    color=_color_or_none(series, index),
                    zorder=zorder,
                )
        ax.set_ylabel("频数" if not series.density else "概率密度")
        return

    if kind == "box":
        if not y_columns:
            raise EmptySeries("箱线图需要至少一个 y 列")
        datasets: list[np.ndarray] = []
        names: list[str] = []
        for index, column in enumerate(y_columns):
            data = _column_values(frame, column)
            data = data[np.isfinite(data)]
            if data.size == 0:
                notes.append(f"{where}：列「{column}」没有有效数值，已跳过。")
                continue
            datasets.append(data)
            names.append(labels[index] if index < len(labels) else column)
        if not datasets:
            raise EmptySeries("箱线图没有可用数据")
        with suppress(Exception):
            ax.boxplot(datasets, tick_labels=names, showmeans=False, patch_artist=True)

        return

    if kind == "pie":
        if not y_columns:
            raise EmptySeries("饼图需要至少一个 y 列")
        values = _column_values(frame, y_columns[0])
        mask = np.isfinite(values)
        values, row_index = values[mask], np.flatnonzero(mask)
        if values.size == 0:
            raise EmptySeries("饼图没有有效数值")
        if np.any(values < 0):
            notes.append(f"{where}：饼图数据包含负值，已取绝对值绘制。")
            values = np.abs(values)
        if float(values.sum()) <= 0:
            raise EmptySeries("饼图数据之和为 0，无法绘制")
        wedge_labels: list[str] | None = None
        if series.x and series.x in frame.columns:
            raw = frame[series.x].to_numpy()[row_index]
            wedge_labels = [format_number(v) if isinstance(v, (int, float, np.number)) else str(v) for v in raw]
        with suppress(Exception):
            ax.pie(
                values,
                labels=wedge_labels,
                autopct=(series.options or {}).get("autopct"),
                startangle=float((series.options or {}).get("startangle", 90.0)),
            )
            ax.set_aspect("equal")
        return

    if kind == "quiver":
        if not (series.u and series.v):
            raise EmptySeries("向量图需要指定 u、v 分量列")
        u = _column_values(frame, series.u)
        v = _column_values(frame, series.v)
        length = min(u.size, v.size, len(frame))
        xs = _resolve_x(frame, series.x, len(frame))[:length]
        ys = _numeric(frame[series.y]) if series.y and series.y in frame.columns else np.zeros(length)
        if ys.size < length:
            ys = np.resize(ys, length)
        ys = ys[:length]
        u, v = u[:length], v[:length]
        if series.normalize:
            magnitude = np.hypot(u, v)
            magnitude[magnitude == 0] = 1.0
            u, v = u / magnitude, v / magnitude
        scale = float(series.scale) if series.scale else 1.0
        with suppress(Exception):
            ax.quiver(
                xs, ys, u, v,
                angles="xy",
                scale_units="xy",
                scale=1.0 / scale if scale else 1.0,
                color=_color_or_none(series),
                alpha=alpha,
                label=series.label or None,
                zorder=zorder,
            )
        return

    if kind == "contour":
        _draw_contour(ax, spec, series, frame, notes, where)
        return

    if kind == "heatmap":
        _draw_heatmap(ax, spec, series, frame, notes, where)
        return

    # ---- 需要 y 列的常规二维图 -------------------------------------------- #
    if not y_columns:
        raise EmptySeries(f"图表类型「{kind}」需要至少一个 y 列")
    if kind == "barh" or (kind == "bar" and series.horizontal):
        kind = "barh"

    width = _coerce_positive(series.bar_width, 0.8)
    base = x_values if x_values.size else np.arange(len(frame), dtype=float)

    for index, column in enumerate(y_columns):
        values = _column_values(frame, column)
        length = min(values.size, base.size, len(frame))
        xs, ys = base[:length], values[:length]
        label = labels[index] if index < len(labels) else column
        color = _color_or_none(series, index)
        if kind in {"line", "step", "fill"}:
            # 折线类允许 NaN 作为断线，只剔除 x、y 同时缺失的点
            keep = ~(np.isnan(xs) & np.isnan(ys))
            xs, ys = xs[keep], ys[keep]
        elif kind in {"bar", "barh", "scatter", "stem"}:
            # 这些图元不允许 NaN/Inf，按需剔除
            keep = _finite_pairs(xs, ys)
            xs, ys = xs[keep], ys[keep]

        if kind == "line":
            ax.plot(
                xs, ys,
                color=color, linestyle=linestyle or "None", linewidth=lw,
                marker=marker or None, markersize=ms, alpha=alpha, label=label, zorder=zorder,
            )
        elif kind == "scatter":
            sx, sy = _points_cap(xs, ys, notes, where)
            ax.scatter(
                sx, sy,
                s=max(1.0, ms * ms), color=color, marker=marker or "o",
                alpha=alpha, label=label, zorder=zorder,
            )
        elif kind == "bar":
            offset = (index - (len(y_columns) - 1) / 2.0) * width / max(1, len(y_columns))
            ax.bar(
                xs + offset, ys, width=width / max(1, len(y_columns)), color=color,
                alpha=alpha, label=label, zorder=zorder,
            )
        elif kind == "barh":
            offset = (index - (len(y_columns) - 1) / 2.0) * width / max(1, len(y_columns))
            ax.barh(
                xs + offset, ys, height=width / max(1, len(y_columns)), color=color,
                alpha=alpha, label=label, zorder=zorder,
            )
        elif kind == "stem":
            stem_color = color or "C0"
            with suppress(Exception):
                ax.vlines(xs, 0.0, ys, color=stem_color, linewidth=max(0.5, lw * 0.6), alpha=alpha, zorder=zorder)
                ax.plot(
                    xs, ys, color=stem_color, linestyle="None", marker=marker or "o",
                    markersize=ms, alpha=alpha, label=label, zorder=zorder + 1,
                )
        elif kind == "step":
            ax.step(
                xs, ys, where="pre", color=color, linestyle=linestyle or "-", linewidth=lw,
                marker=marker or None, markersize=ms, alpha=alpha, label=label, zorder=zorder,
            )
        elif kind == "fill":
            fill_color = color or _FILL_CYCLE[index % len(_FILL_CYCLE)]
            fill_alpha = alpha if alpha < 1.0 else 0.35
            with suppress(Exception):
                ax.fill_between(xs, ys, 0.0, color=fill_color, alpha=fill_alpha, label=label, zorder=zorder)
            ax.plot(
                xs, ys, color=fill_color, linestyle=linestyle or "-", linewidth=lw,
                marker=marker or None, markersize=ms, alpha=alpha, zorder=zorder + 1,
            )
        elif kind == "errorbar":
            yerr, ok = _error_bars(frame, series, y_columns, series.yerr)
            if series.yerr and not ok:
                notes.append(f"{where}：找不到误差列「{series.yerr}」，已按普通折线绘制。")
                yerr = None
            error: np.ndarray | None = None
            if yerr is not None:
                error = np.asarray(yerr, dtype=object)
                error = error[index] if error.ndim > 1 and error.shape[0] > 1 else yerr
                error = np.asarray(error, dtype=float)
                if error.size != ys.size:
                    error = error[: ys.size] if error.size > ys.size else np.resize(error, ys.size)
            if error is None:
                keep = _finite_pairs(xs, ys)
            else:
                keep = _finite_pairs(xs, ys, error)
                error = error[keep]
            exs, eys = xs[keep], ys[keep]
            kwargs: dict[str, Any] = {
                "color": color,
                "linestyle": linestyle or "None",
                "linewidth": lw,
                "marker": marker or "o",
                "markersize": ms,
                "alpha": alpha,
                "label": label,
                "zorder": zorder,
                "capsize": float((series.options or {}).get("capsize", 3.0)),
            }
            if error is not None:
                kwargs["yerr"] = error
            with suppress(Exception):
                ax.errorbar(exs, eys, **kwargs)


def _draw_contour(ax: Any, spec: PlotSpec, series: SeriesSpec, frame: pd.DataFrame, notes: list[str], where: str) -> None:
    """等高线图：三元组先网格化，再画 ``contour`` / ``contourf``。"""
    if not series.z:
        raise EmptySeries("等高线图需要指定 z 列")
    xs = _resolve_x(frame, series.x, len(frame))
    ys = _column_values(frame, series.y) if series.y else np.arange(len(frame), dtype=float)
    zs = _column_values(frame, series.z)
    length = min(xs.size, ys.size, zs.size)
    if length < 3:
        raise EmptySeries("等高线图至少需要 3 个数据点")
    gx, gy, gz = to_grid(xs[:length], ys[:length], zs[:length])
    if not _has_finite(gz):
        raise EmptySeries("等高线图的数据无法插值成网格")
    levels = max(2, int(_coerce_positive(series.levels, 12.0)))
    cmap = series.cmap or "viridis"
    contour_set = None
    if series.filled:
        with suppress(Exception):
            contour_set = ax.contourf(gx, gy, gz, levels=levels, cmap=cmap, alpha=float(series.alpha or 1.0))
    if contour_set is None:
        with suppress(Exception):
            contour_set = ax.contour(gx, gy, gz, levels=levels, cmap=cmap, alpha=float(series.alpha or 1.0))
    if contour_set is None:
        notes.append(f"{where}：等高线绘制失败，已跳过该序列。")
        return
    if spec.colorbar:
        with suppress(Exception):
            ax.figure.colorbar(contour_set, ax=ax)
    if (series.options or {}).get("label_levels"):
        with suppress(Exception):
            lines = ax.contour(gx, gy, gz, levels=levels, colors="k", linewidths=0.6)
            ax.clabel(lines, inline=True, fontsize=8)


def _draw_heatmap(ax: Any, spec: PlotSpec, series: SeriesSpec, frame: pd.DataFrame, notes: list[str], where: str) -> None:
    """热力图：三元组网格化后用 ``pcolormesh`` 绘制。"""
    if not series.z:
        raise EmptySeries("热力图需要指定 z 列")
    xs = _resolve_x(frame, series.x, len(frame))
    ys = _column_values(frame, series.y) if series.y else np.arange(len(frame), dtype=float)
    zs = _column_values(frame, series.z)
    length = min(xs.size, ys.size, zs.size)
    if length < 2:
        raise EmptySeries("热力图至少需要 2 个数据点")
    gx, gy, gz = to_grid(xs[:length], ys[:length], zs[:length])
    if not _has_finite(gz):
        raise EmptySeries("热力图的数据无法插值成网格")
    with suppress(Exception):
        mesh = ax.pcolormesh(
            gx, gy, gz, cmap=series.cmap or "viridis", shading="auto",
            alpha=float(series.alpha or 1.0),
        )
        if spec.colorbar:
            ax.figure.colorbar(mesh, ax=ax)


def _draw_3d(ax: Any, spec: PlotSpec, series: SeriesSpec, frame: pd.DataFrame, notes: list[str], where: str) -> None:
    """三维图表的分派绘制。"""
    kind = str(series.kind or "line3d").lower()
    if kind not in KINDS_3D:
        notes.append(f"{where}：三维投影不支持图表类型「{kind}」，已按三维折线图绘制。")
        kind = "line3d"

    y_columns = series.y_columns()
    marker = str(series.marker or "")
    alpha = float(np.clip(float(series.alpha or 1.0), 0.0, 1.0))
    lw = _coerce_positive(series.linewidth, 2.0)
    ms = _coerce_positive(series.markersize, 6.0)

    # 网格类图表：z 是网格列名，y 只取第一列
    if kind in {"surface", "wireframe", "contour3d"}:
        if not series.z:
            raise EmptySeries(f"{KINDS_3D[kind]}需要指定 z 列")
        xs = _resolve_x(frame, series.x, len(frame))
        ys = _column_values(frame, series.y) if series.y else np.arange(len(frame), dtype=float)
        zs = _column_values(frame, series.z)
        length = min(xs.size, ys.size, zs.size)
        if length < 4:
            raise EmptySeries(f"{KINDS_3D[kind]}至少需要 4 个数据点")
        gx, gy, gz = to_grid(xs[:length], ys[:length], zs[:length])
        if not _has_finite(gz):
            raise EmptySeries(f"{KINDS_3D[kind]}的数据无法插值成网格")
        if not np.isfinite(gz).all():
            notes.append(f"{where}：数据点不构成规则网格，已降级为插值网格并留空部分区域。")
        if gx.size > _MAX_GRID_CELLS:
            gx, gy, gz = _thin_grid(gx, gy, gz)
        cmap = series.cmap or "viridis"
        surface = None
        if kind == "surface":
            with suppress(Exception):
                surface = ax.plot_surface(
                    gx, gy, gz, cmap=cmap, alpha=alpha,
                    linewidth=0 if series.filled else max(0.0, lw / 4.0),
                    edgecolor="none" if series.filled else None,
                    antialiased=True,
                )
        elif kind == "wireframe":
            with suppress(Exception):
                surface = ax.plot_wireframe(
                    gx, gy, gz, color=_color_or_none(series), linewidth=max(0.2, lw / 2.0), alpha=alpha,
                )
        else:
            with suppress(Exception):
                surface = ax.contour3D(gx, gy, gz, levels=max(2, int(_coerce_positive(series.levels, 12.0))), cmap=cmap, alpha=alpha)
        if surface is not None and spec.colorbar and kind in {"surface", "wireframe"}:
            with suppress(Exception):
                ax.figure.colorbar(surface, ax=ax, shrink=0.6)
        return

    if kind == "quiver3d":
        if not (series.u and series.v and series.w):
            raise EmptySeries("三维向量场需要指定 u、v、w 分量列")
        xs = _resolve_x(frame, series.x, len(frame))
        ys = _column_values(frame, series.y) if series.y else np.zeros(len(frame))
        zs = _column_values(frame, series.z) if series.z else np.zeros(len(frame))
        us = _column_values(frame, series.u)
        vs = _column_values(frame, series.v)
        ws = _column_values(frame, series.w)
        length = min(xs.size, ys.size, zs.size, us.size, vs.size, ws.size)
        if length == 0:
            raise EmptySeries("三维向量场没有可用数据")
        xs, ys, zs = xs[:length], ys[:length], zs[:length]
        us, vs, ws = us[:length], vs[:length], ws[:length]
        if series.normalize:
            magnitude = np.sqrt(us ** 2 + vs ** 2 + ws ** 2)
            magnitude[magnitude == 0] = 1.0
            us, vs, ws = us / magnitude, vs / magnitude, ws / magnitude
        length_scale = 1.0 / float(series.scale) if series.scale else 1.0
        with suppress(Exception):
            ax.quiver(xs, ys, zs, us, vs, ws, length=length_scale, normalize=False, color=_color_or_none(series), alpha=alpha)
        return

    if kind == "bar3d":
        if not y_columns:
            raise EmptySeries("三维柱形图需要至少一个 y 列")
        xs = _resolve_x(frame, series.x, len(frame))
        zs = _column_values(frame, series.z) if series.z else np.zeros(len(frame))
        dy = _coerce_positive(series.bar_width, 0.8)
        for index, column in enumerate(y_columns):
            values = _column_values(frame, column)
            length = min(values.size, xs.size, zs.size, len(frame))
            if length == 0:
                continue
            x0 = xs[:length] - dy / 2.0
            y0 = np.zeros(length)
            z0 = zs[:length]
            dz = values[:length]
            color = _color_or_none(series, index) or _FILL_CYCLE[index % len(_FILL_CYCLE)]
            with suppress(Exception):
                ax.bar3d(x0, y0, z0, dy, dy, dz, color=color, alpha=alpha, shade=True)
        return

    if kind == "trisurf":
        xs = _resolve_x(frame, series.x, len(frame))
        ys = _column_values(frame, series.y) if series.y else np.arange(len(frame), dtype=float)
        zs = _column_values(frame, series.z) if series.z else np.zeros(len(frame))
        length = min(xs.size, ys.size, zs.size)
        if length < 3:
            raise EmptySeries("三角剖分曲面至少需要 3 个数据点")
        with suppress(Exception):
            surface = ax.plot_trisurf(xs[:length], ys[:length], zs[:length], cmap=series.cmap or "viridis", alpha=alpha)
            if spec.colorbar:
                ax.figure.colorbar(surface, ax=ax, shrink=0.6)
        return

    # ---- 三维点/线 -------------------------------------------------------- #
    if not y_columns:
        raise EmptySeries(f"{KINDS_3D[kind]}需要至少一个 y 列")
    xs = _resolve_x(frame, series.x, len(frame))
    zs = _column_values(frame, series.z) if series.z and series.z in frame.columns else np.zeros(len(frame))
    labels = _series_labels(series, y_columns)
    for index, column in enumerate(y_columns):
        values = _column_values(frame, column)
        length = min(values.size, xs.size, zs.size, len(frame))
        if length == 0:
            notes.append(f"{where}：列「{column}」没有有效数值，已跳过。")
            continue
        label = labels[index] if index < len(labels) else column
        color = str(series.color or "").strip() or None
        if kind == "line3d":
            ax.plot(
                xs[:length], values[:length], zs[:length],
                color=color, linestyle=str(series.linestyle or "-"), linewidth=lw,
                marker=marker or None, markersize=ms, alpha=alpha, label=label,
            )
        else:
            ax.scatter(
                xs[:length], values[:length], zs[:length],
                color=color, marker=marker or "o", s=max(1.0, ms * ms), alpha=alpha, label=label,
            )


def _thin_grid(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """网格过大时等间隔抽稀。"""
    step = int(math.ceil(math.sqrt(gx.size / _MAX_GRID_CELLS)))
    return gx[::step, ::step], gy[::step, ::step], gz[::step, ::step]


# --------------------------------------------------------------------------- #
# 坐标轴设置
# --------------------------------------------------------------------------- #
def _apply_scales(ax: Any, spec: PlotSpec, notes: list[str]) -> None:
    """应用 x/y/z 缩放方式。"""
    pairs = [(("x", spec.xscale), ax.set_xscale), (("y", spec.yscale), ax.set_yscale)]
    if spec.is_3d():
        pairs.append((("z", spec.zscale), getattr(ax, "set_zscale", None)))
    for (name, scale), setter in pairs:
        if setter is None:
            continue
        value = str(scale or "linear").strip() or "linear"
        if value == "linear":
            continue
        with suppress(Exception):
            setter(value)
            continue
        notes.append(f"{name} 轴不支持缩放方式「{value}」，已使用线性坐标。")


def _apply_limits(ax: Any, spec: PlotSpec) -> None:
    """应用 xlim/ylim/zlim（3D 用 set_box_aspect 之外的原始限值）。"""
    for values, setter in (
        (spec.xlim, ax.set_xlim),
        (spec.ylim, ax.set_ylim),
        (spec.zlim, getattr(ax, "set_zlim", None)),
    ):
        if not values or setter is None:
            continue
        pair = tuple(values)
        if len(pair) != 2:
            continue
        with suppress(Exception):
            setter(pair)


# --------------------------------------------------------------------------- #
# 中英文与公式混排的标签
# --------------------------------------------------------------------------- #
_CJK_RE = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef]")


def has_cjk(text: str) -> bool:
    """文本里是否含有中日韩字符。"""
    return bool(_CJK_RE.search(str(text or "")))


def split_cjk_math(text: str) -> list[tuple[str, str]] | None:
    """把 ``"中文 $x^2$ 文本"`` 拆成 ``[("text", "中文 "), ("math", "x^2"), ...]``。

    matplotlib 有一个**硬限制**：只要字符串里出现了成对的 ``$``，整行（包括
    ``$`` 外面的中文）都会被交给 mathtext 解析，而 mathtext 的数学字体没有
    汉字字形，于是中文会被替换成替代符号（豆腐块）——即使用
    ``mathtext.fontset="custom"`` 把 ``mathtext.rm`` 指到宋体也一样。

    因此当标题/轴标签**同时**含中文与公式时，必须拆成多个 ``Text`` 图元分别渲染。
    返回值 ``None`` 表示不需要特殊处理（可以交给普通 Text）。
    """
    text = str(text or "")
    if not text:
        return None
    math_markers = text.count("$") - text.count(r"\$")
    if math_markers < 2 or not has_cjk(text):
        return None
    parts: list[tuple[str, str]] = []
    for index, chunk in enumerate(re.split(r"(?<!\\)\$", text)):
        if not chunk:
            continue
        parts.append(("math" if index % 2 else "text", chunk))
    return parts or None


def _rich_label_box(text: str, *, fontsize: float, color: str, math_fontfamily: str):
    """把混排标签做成 ``HPacker``（每个片段一个 Text，各自用正确的字体）。"""
    from matplotlib.offsetbox import HPacker, TextArea

    parts = split_cjk_math(text)
    if parts is None:
        return None
    children = []
    for kind, chunk in parts:
        if kind == "math":
            children.append(
                TextArea(
                    f"${chunk}$",
                    textprops={
                        "color": color,
                        "fontsize": fontsize,
                        "math_fontfamily": math_fontfamily,
                    },
                )
            )
        else:
            # 正文片段交给字体回退链处理（英文 Times、中文宋体）
            children.append(TextArea(chunk, textprops={"color": color, "fontsize": fontsize}))
    return HPacker(children=children, align="baseline", pad=0, sep=0)


def _measure_chunk(kind: str, chunk: str, fontsize: float) -> float:
    """测量一个片段的宽度（点），与 matplotlib 使用同一套度量。"""
    from .latex.measure import MathParseError, measure_math, measure_text

    if kind == "math":
        try:
            return float(measure_math(chunk, fontsize)[0])
        except MathParseError:
            pass
    family = matplotlib.rcParams.get("font.family", ["sans-serif"])
    try:
        return float(measure_text(chunk, fontsize, family)[0])
    except Exception:
        return len(chunk) * fontsize * 0.6


class _RichLabel:
    """把「中文 + 公式」混排的标签拆成多个 ``Text`` 并跟随 matplotlib 的标签位置。

    matplotlib 把标签画在哪儿（左侧要给刻度标签留多少空、标题要抬多高、Y 轴要
    旋转 90°）它自己算得最准，所以这里保留一个 **alpha=0 的原始标签**作为
    「位置参照」：它照常参与 ``tight_layout`` 的边距计算，而真正显示的是我们
    拆出来的若干片段。每次绘制时读一次参照标签的包围盒，把片段居中铺在上面。
    """

    def __init__(
        self,
        ax: Any,
        parts: list[tuple[str, str]],
        widths: list[float],
        reference: Any,
        kind: str,
        fontsize: float,
        math_fontfamily: str,
    ) -> None:
        import matplotlib.artist as martist
        from matplotlib.transforms import IdentityTransform

        self._ax = ax
        self._parts = parts
        self._widths = widths
        self._reference = reference
        self._vertical = kind == "ylabel"
        self._texts: list[Any] = []
        for part_kind, chunk in parts:
            if not chunk.strip():
                self._texts.append(None)
                continue
            text = ax.text(
                0.0,
                0.0,
                f"${chunk}$" if part_kind == "math" else chunk,
                fontsize=fontsize,
                ha="center",
                va="center",
                rotation=90.0 if self._vertical else 0.0,
                rotation_mode="anchor",
                # 注意：transform=None 会退回 transData（把像素当成数据坐标），
                # 必须显式用恒等变换把坐标解释为显示坐标。
                transform=IdentityTransform(),
                clip_on=False,
                zorder=2.6,
            )
            if part_kind == "math":
                with suppress(Exception):
                    text.set_math_fontfamily(math_fontfamily)
            self._texts.append(text)

        controller = martist.Artist()
        controller.set_figure(ax.get_figure())
        controller.set_zorder(2.5)       # 先于片段绘制，好让位置就绪
        controller.draw = self._reposition      # type: ignore[method-assign]
        ax.add_artist(controller)
        self._controller = controller

    def _reposition(self, renderer: Any) -> None:
        try:
            bbox = self._reference.get_window_extent(renderer)
        except Exception:
            return
        # 度量结果是「点」，而显示坐标是「像素」，必须按当前 dpi 换算
        try:
            scale = float(self._ax.get_figure().dpi) / 72.0
        except Exception:
            scale = 1.0
        widths = [width * scale for width in self._widths]
        center_x = (bbox.x0 + bbox.x1) / 2.0
        center_y = (bbox.y0 + bbox.y1) / 2.0
        cursor = -sum(widths) / 2.0
        for (_part_kind, _chunk), width, text in zip(self._parts, widths, self._texts):
            if text is not None:
                if self._vertical:
                    text.set_position((center_x, center_y + cursor + width / 2.0))
                else:
                    text.set_position((center_x + cursor + width / 2.0, center_y))
            cursor += width


def _set_label_with_math(ax: Any, text: str, kind: str, *, fontsize: float, pad: float) -> None:
    """设置标题 / 轴标签。

    当文本**同时**含中文与 ``$公式$`` 时，matplotlib 会把整行交给 mathtext，
    而 mathtext 的数学字体没有汉字字形，中文会变成替代符号（豆腐块）——即使把
    ``mathtext.rm`` 指到宋体也一样（实测 matplotlib 3.11）。因此这里改为拆成
    若干个 ``Text``：中文片段走字体回退链，公式片段走 mathtext，并按**点**精确
    对齐，全部保持矢量、也照常参与布局。
    """
    parts = split_cjk_math(text)
    if parts is None:
        with suppress(Exception):
            if kind == "title":
                ax.set_title(text)
            elif kind == "xlabel":
                ax.set_xlabel(text)
            elif kind == "ylabel":
                ax.set_ylabel(text)
        return

    try:
        plain = re.sub(r"\$([^$]*)\$", r"\1", text)
        if kind == "title":
            ax.set_title(plain, alpha=0.0)
            reference = ax.title
        elif kind == "xlabel":
            ax.set_xlabel(plain, alpha=0.0, labelpad=pad)
            reference = ax.xaxis.label
        else:
            ax.set_ylabel(plain, alpha=0.0, labelpad=pad)
            reference = ax.yaxis.label

        widths = [_measure_chunk(part_kind, chunk, fontsize) for part_kind, chunk in parts]
        _RichLabel(
            ax,
            parts,
            widths,
            reference,
            kind,
            fontsize,
            str(matplotlib.rcParams.get("mathtext.fontset", "stix")),
        )
        return
    except Exception:
        pass

    # 极端情况下的兜底：整串按纯文本画出来（至少不是豆腐块）
    with suppress(Exception):
        plain = re.sub(r"\$([^$]*)\$", r"\1", text)
        if kind == "title":
            ax.set_title(plain)
        elif kind == "xlabel":
            ax.set_xlabel(plain)
        elif kind == "ylabel":
            ax.set_ylabel(plain)


def _apply_axis_text(ax: Any, spec: PlotSpec) -> None:
    """标题与轴标签。

    ``$...$`` 公式、中文都可以直接写；两者**同时**出现时会切换到多图元渲染，
    避免 matplotlib 把中文交给 mathtext 而显示成替代符号。
    """
    fontsize = float(matplotlib.rcParams.get("axes.labelsize", 11.0))
    title_size = float(matplotlib.rcParams.get("axes.titlesize", 12.0))
    title_pad = float(matplotlib.rcParams.get("axes.titlepad", 6.0))
    label_pad = float(matplotlib.rcParams.get("axes.labelpad", 4.0))

    if spec.title:
        _set_label_with_math(ax, spec.title, "title", fontsize=title_size, pad=title_pad)
    if spec.is_3d():
        # 三维轴标签由 mpl_toolkits 自行排布（会随视角旋转），无法插入自定义
        # 多图元文本；混排时退化为纯文本，避免出现替代符号（豆腐块）。
        for value, setter in (
            (spec.xlabel, ax.set_xlabel),
            (spec.ylabel, ax.set_ylabel),
            (spec.zlabel, ax.set_zlabel),
        ):
            if not value:
                continue
            text = value
            if split_cjk_math(text) is not None:
                text = re.sub(r"\$([^$]*)\$", r"\1", text)
            with suppress(Exception):
                setter(text)
        return
    if spec.xlabel:
        _set_label_with_math(ax, spec.xlabel, "xlabel", fontsize=fontsize, pad=label_pad)
    if spec.ylabel:
        _set_label_with_math(ax, spec.ylabel, "ylabel", fontsize=fontsize, pad=label_pad)


def _apply_legend(ax: Any, spec: PlotSpec) -> None:
    """开启图例（没有任何带标签的图元时跳过，避免 matplotlib 警告）。"""
    if not spec.legend:
        return
    handles = []
    with suppress(Exception):
        handles, _labels = ax.get_legend_handles_labels()
    if not handles:
        return
    kwargs: dict[str, Any] = {"loc": spec.legend_loc or "best", "frameon": bool(spec.legend_frame)}
    with suppress(Exception):
        ax.legend(**kwargs)
        return
    with suppress(Exception):
        ax.legend(loc="best", frameon=bool(spec.legend_frame))


def _apply_grid(ax: Any, spec: PlotSpec) -> None:
    """网格开关（3D 轴没有 grid()，忽略即可）。"""
    if not spec.grid:
        return
    with suppress(Exception):
        ax.grid(True, alpha=0.35)


def _apply_aspect(ax: Any, spec: PlotSpec) -> None:
    """等比例坐标轴。"""
    if not spec.equal_aspect:
        return
    with suppress(Exception):
        ax.set_aspect("equal", adjustable="datalim")
        return
    with suppress(Exception):
        ax.set_aspect("equal")


def _apply_colorcycle(ax: Any, spec: PlotSpec, notes: list[str]) -> None:
    """自定义颜色循环。"""
    colors = [str(c) for c in (spec.colorcycle or []) if str(c).strip()]
    if not colors:
        return
    invalid = [c for c in colors if not mcolors.is_color_like(c)]
    if invalid:
        notes.append("颜色循环中有无法识别的颜色：" + "、".join(invalid))
        colors = [c for c in colors if mcolors.is_color_like(c)]
    if not colors:
        return
    with suppress(Exception):
        ax.set_prop_cycle(color=colors)


def _apply_3d_view(ax: Any, spec: PlotSpec) -> None:
    """三维视角。"""
    if not spec.is_3d():
        return
    with suppress(Exception):
        ax.view_init(elev=float(spec.view_elev), azim=float(spec.view_azim))


def _apply_tight_layout(fig: Figure, spec: PlotSpec) -> None:
    """紧凑布局（3D 轴可能不支持，失败就忽略）。"""
    if not spec.tight_layout:
        return
    with suppress(Exception), _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        fig.tight_layout()


def _coerce_figsize(value: Any, fallback: tuple[float, float] = (8.0, 5.0)) -> tuple[float, float]:
    """把 ``spec.figsize`` 规整为两个正的浮点数，坏值一律回退默认尺寸。"""
    try:
        width, height = (float(v) for v in tuple(value))
    except (TypeError, ValueError):
        return fallback
    if not (math.isfinite(width) and math.isfinite(height)) or width <= 0 or height <= 0:
        return fallback
    return width, height


def _coerce_positive(value: Any, fallback: float, *, low: float = 0.0) -> float:
    """把数值字段规整为不小于 ``low`` 的有限浮点数，坏值回退 ``fallback``。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number) or number <= low:
        return fallback
    return number


def _empty_message(ax: Any, text: str = "没有可绘制的数据") -> None:
    """在空坐标系中央写一句中文提示。"""
    with suppress(Exception):
        ax.text(
            0.5, 0.5, text,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=13.0, color="#888888", wrap=True,
        )


# --------------------------------------------------------------------------- #
# 公开 API
# --------------------------------------------------------------------------- #
def build_figure(
    spec: PlotSpec,
    datasets: Mapping[str, pd.DataFrame],
    *,
    figure: Figure | None = None,
) -> tuple[Figure, list[str]]:
    """按 ``spec`` 渲染图片，返回 ``(figure, warnings)``。

    ``warnings`` 是简体中文提示列表（缺列、数据为空、需要降级等）；
    任何可恢复的问题都只记录警告并继续绘制其它序列，绝不抛异常。

    传入 ``figure`` 时会**复用**该 Figure（先 ``clear()`` 再重建）。嵌入 Qt 画布时
    应始终传入手上的 Figure：这样不需要重建 ``FigureCanvasQTAgg``，
    重绘时界面不会抖动。
    """
    notes: list[str] = []
    figsize = _coerce_figsize(spec.figsize)
    dpi = int(_coerce_positive(spec.dpi, 110.0))
    font_size = _coerce_positive(spec.font_size, 11.0)

    with _theme_context(spec):
        matplotlib.rcParams.update(
            {
                "font.size": font_size,
                "axes.titlesize": font_size + 1.5,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size - 1.5,
                "ytick.labelsize": font_size - 1.5,
                "legend.fontsize": font_size - 1.5,
            }
        )

        if figure is None:
            fig = Figure(figsize=figsize, dpi=dpi)
        else:
            fig = figure
            fig.clear()
            fig.set_size_inches(figsize, forward=True)
            fig.set_dpi(dpi)
            fig.patch.set_visible(True)
        if spec.is_3d():
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
        try:
            ax.margins(x=0.02)
        except Exception:
            pass

        available = {str(k): v for k, v in (datasets or {}).items() if isinstance(v, pd.DataFrame)}
        if not available:
            notes.append("没有可用的数据集，已绘制空白图。")
        if not spec.series:
            notes.append("图规格中没有任何序列，已绘制空白图。")

        frames, _keys = _prepare_frames(spec, available, notes)
        drawn = 0
        for index, series in enumerate(spec.series):
            if series.dataset and series.dataset not in frames:
                continue
            frame = frames.get(series.dataset)
            if frame is None:
                continue
            where = f"序列 {index + 1}（{series.dataset}）"
            _warn_missing_columns(frame, series, notes, where)
            try:
                if spec.is_3d():
                    _draw_3d(ax, spec, series, frame, notes, where)
                else:
                    _draw_2d(ax, spec, series, frame, notes, where)
                drawn += 1
            except EmptySeries as exc:
                notes.append(f"{where}：{exc}")
            except Exception as exc:  # 单条序列失败不影响整张图
                notes.append(f"{where}：绘制失败（{type(exc).__name__}: {exc}），已跳过。")

        if drawn == 0:
            _empty_message(ax)

        _apply_axis_text(ax, spec)
        _apply_scales(ax, spec, notes)
        _apply_limits(ax, spec)
        _apply_colorcycle(ax, spec, notes)
        _apply_3d_view(ax, spec)
        _apply_grid(ax, spec)
        _apply_aspect(ax, spec)
        _apply_legend(ax, spec)
        _apply_tight_layout(fig, spec)

    return fig, notes


def available_columns(spec: PlotSpec, datasets: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    """返回 ``{数据集名: [列名...]}``，界面用它填充列选择下拉框。

    优先只列出图中引用到的数据集；若 ``spec`` 没有引用任何数据集，
    则列出全部已加载的数据集。列名统一转为字符串。
    """
    table = {str(k): v for k, v in (datasets or {}).items() if isinstance(v, pd.DataFrame)}
    used = [name for name in spec.used_datasets() if name in table]
    targets = used or list(table)
    return {name: [str(c) for c in table[name].columns] for name in targets}


def figure_to_svg(fig: Figure, **savefig_kwargs: Any) -> str:
    """把图渲染为 SVG 文本（纯内存，不写磁盘）。

    返回值从 ``<svg`` 开始：matplotlib 生成的 XML 声明与 DOCTYPE 会被去掉，
    便于界面层直接塞进 ``QTextBrowser`` / ``QSvgWidget``。
    """
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {"format": "svg", "metadata": {"Date": None}}
    kwargs.update(savefig_kwargs)
    try:
        fig.savefig(buffer, **kwargs)
    except Exception:
        buffer = io.BytesIO()
        kwargs.pop("metadata", None)
        with suppress(Exception):
            fig.savefig(buffer, format="svg")
    text = buffer.getvalue().decode("utf-8", errors="replace")
    start = text.find("<svg")
    return text[start:] if start > 0 else text


def save_figure(fig: Figure, path: str, **savefig_kwargs: Any) -> str:
    """按扩展名保存图片，返回实际写入的路径。

    支持 ``.svg`` / ``.png`` / ``.pdf`` / ``.pgf`` / ``.eps`` / ``.jpg``；
    其它扩展名回退为 PNG（仅替换扩展名，保留原目录）。
    """
    target = str(path)
    root, ext = os.path.splitext(target)
    fmt = ext.lstrip(".").lower()
    if fmt not in {"svg", "png", "pdf", "pgf", "eps", "jpg"}:
        target = f"{root or target}.png"
        fmt = "png"
    if fmt == "jpg":
        fmt = "jpeg"
    dirname = os.path.dirname(os.path.abspath(target))
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if fmt == "svg":
        # SVG 走文本路径，保证落盘内容与 figure_to_svg 一致（无 XML 声明）
        options = {k: v for k, v in savefig_kwargs.items() if k not in {"format", "metadata"}}
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(figure_to_svg(fig, **options))
        return target

    kwargs: dict[str, Any] = {"format": fmt}
    kwargs.update(savefig_kwargs)
    kwargs.setdefault("transparent", False)
    fig.savefig(target, **kwargs)
    return target


def default_series(dataset: str, df: pd.DataFrame, kind: str = "line") -> SeriesSpec:
    """根据 DataFrame 的列自动猜一条合理的初始 :class:`SeriesSpec`。

    第一个数值列作 x，其余数值列作 y / z；``kind`` 为 ``surface``、
    ``wireframe``、``contour3d``、``contour``、``heatmap`` 时把第二个数值列
    作为 y、第三个作为 z（网格类图表需要）。
    """
    columns = _numeric_columns(df) if isinstance(df, pd.DataFrame) else []
    series = SeriesSpec(dataset=str(dataset), kind=str(kind or "line"))
    if not columns:
        return series

    series.x = columns[0]
    if len(columns) >= 3:
        series.y = columns[1]
        series.z = columns[2]
    elif len(columns) == 2:
        series.y = columns[1]
    else:
        series.y = columns[0]
        series.x = ""
    return series


def default_spec(dataset: str, df: pd.DataFrame, projection: str = "2d") -> PlotSpec:
    """生成一张开箱即用的图规格（一条序列 + 自动标题与轴标签）。"""
    is_3d = str(projection).lower() == "3d"
    columns = _numeric_columns(df) if isinstance(df, pd.DataFrame) else []
    rows = int(len(df)) if isinstance(df, pd.DataFrame) else 0
    name = str(dataset)

    if is_3d:
        kind = "surface" if len(columns) >= 3 else "line3d"
    elif len(columns) >= 3 and _looks_like_grid(df, columns[0]):
        kind = "heatmap"
    elif len(columns) == 1 and rows > 12:
        kind = "hist"
    else:
        kind = "line"

    series = default_series(name, df, kind)
    if kind == "hist":
        series.x = ""
    if kind in {"surface", "wireframe", "contour3d", "contour", "heatmap"} and len(columns) >= 3:
        series.x, series.y, series.z = columns[0], columns[1], columns[2]

    spec = PlotSpec(projection="3d" if is_3d else "2d", series=[series])
    kind_name = (KINDS_3D if is_3d else KINDS_2D).get(kind, kind)
    spec.title = f"{name} · {kind_name}" if name else kind_name
    if series.x:
        spec.xlabel = series.x
    if series.y:
        spec.ylabel = series.y
    if is_3d and series.z:
        spec.zlabel = series.z
    return spec


def _looks_like_grid(df: pd.DataFrame, x_column: str, min_repeat: int = 4) -> bool:
    """粗略判断 x 列是否「重复出现同一批取值」（说明是网格长表）。"""
    if not isinstance(df, pd.DataFrame) or x_column not in df.columns or len(df) < min_repeat * 2:
        return False
    values = df[x_column].to_numpy()
    unique = np.unique(values)
    if unique.size < 2 or unique.size > len(df) // 2:
        return False
    counts = pd.Series(values).value_counts()
    return bool(counts.min() >= 2)


def summary_text(fig: Figure) -> str:
    """把图对象的基本情况汇总成一句中文说明（用于日志/状态栏）。"""
    try:
        axes = fig.get_axes()
    except Exception:
        axes = []
    kinds = sorted({type(ax).__name__ for ax in axes})
    size = fig.get_size_inches()
    return (
        f"{len(axes)} 个坐标轴（{'、'.join(kinds) or '无'}），"
        f"画布 {format_number(float(size[0]))}×{format_number(float(size[1]))} 英寸，"
        f"dpi {format_number(float(fig.dpi))}"
    )
