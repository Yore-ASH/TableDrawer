"""代码同步：把 :class:`core.spec.PlotSpec` 反向导出成**可独立运行**的 matplotlib 源码。

设计目标
--------
1. **自包含**：生成的脚本不导入本项目任何模块，只用 ``numpy`` / ``pandas`` /
   ``matplotlib``，把 ``.venv\\Scripts\\python.exe`` 指向它就能直接画图。
2. **保真**：序列类型映射、轴设置、图例/色条/主题的语义与 :mod:`core.plotting`
   保持一致（同一个 ``PlotSpec`` 在界面里和导出的脚本里应该长得一样）。
3. **可读**：中文分节注释、内联数据带缩进换行、NaN/Inf 写成 ``np.nan``/``np.inf``。

典型用法::

    from core.codesync import CodeGenOptions, generate_script

    source = generate_script(spec, datasets, CodeGenOptions(embed_data=True))
    pathlib.Path("figure.py").write_text(source, encoding="utf-8")
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import keyword
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.spec import KINDS_2D, KINDS_3D, PlotSpec, SeriesSpec, kind_label

__all__ = [
    "CodeGenOptions",
    "dataset_var_name",
    "generate_series_code",
    "generate_script",
    "reset_var_registry",
]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
#: 内联数组单行最大宽度
_MAX_LINE = 100

#: 内联数组超过这个元素个数时改用 ``.reshape(...)`` 紧凑写法
_COMPACT_MIN = 24

#: 生成脚本里使用的中文字体候选（与 :data:`core.mplsetup.CJK_CANDIDATES` 同源）
CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Hiragino Sans GB",
    "DengXian",
    "SimSun",
    "Arial Unicode MS",
)

#: 主题名：这些等同于 matplotlib 默认样式，不写 ``plt.style.use(...)``
_DEFAULT_THEMES: frozenset[str] = frozenset({"light", ""})

#: 需要把 (x, y, z) 三元组整理成规则网格的 3D 类型
_GRID_3D: frozenset[str] = frozenset({"surface", "wireframe", "contour3d"})

#: ``spec.colorbar`` 真正生效的序列类型
_COLORBAR_KINDS: frozenset[str] = frozenset(
    {"surface", "wireframe", "contour", "contour3d", "heatmap", "trisurf"}
)

#: 不使用 x 轴的 2D 类型（与 :mod:`core.plotting` 的 ``NO_X_KINDS`` 一致）
_NO_X_KINDS: frozenset[str] = frozenset({"hist", "box", "pie"})

#: 坐标轴为分类/文本性质，不再应用 xlim / yscale 的 2D 类型
_TEXTLIKE_KINDS: frozenset[str] = frozenset({"pie", "bar", "barh"})

#: 饼图默认起始角（与 :mod:`core.plotting` 一致）
_PIE_STARTANGLE = 90.0

#: 误差棒默认帽宽
_DEFAULT_CAPSIZE = 3.0

#: ``fill`` 与 ``bar3d`` 的兜底配色循环（与 :mod:`core.plotting` 的 ``_FILL_CYCLE`` 一致）
_FILL_CYCLE: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
)

_PY_STRING_SAFE = re.compile(r"[A-Za-z0-9_]")

# --------------------------------------------------------------------------- #
# 公开选项
# --------------------------------------------------------------------------- #
@dataclass
class CodeGenOptions:
    """导出源码时的可调选项。"""

    embed_data: bool = True                         # True: 数据内联为 np.array；False: 从 CSV 读取
    csv_paths: dict[str, str] = field(default_factory=dict)  # 数据集名 -> CSV 路径
    array_format: str = "compact"                   # "compact"（reshape+整行）或 "pretty"（每行一个数）
    include_comments: bool = True                   # 是否写中文分节注释
    include_sample_save: bool = True                # 是否在末尾写 fig.savefig(...) 示例
    output_filename: str = "figure.svg"             # 示例保存文件名
    function_wrapper: bool = False                  # True: 生成 def make_figure() -> Figure 形式
    precision: int = 6                              # 内联数值的位数上限（同时决定数组折行宽度）


# --------------------------------------------------------------------------- #
# 变量名注册表
# --------------------------------------------------------------------------- #
_VAR_LOCK = threading.RLock()
_USED_VARS: set[str] = set()


def reset_var_registry() -> None:
    """清空 :func:`dataset_var_name` 的去重表（每次生成源码前调用）。"""
    with _VAR_LOCK:
        _USED_VARS.clear()


def _ascii_name(name: str) -> str:
    """把任意名字压成 ascii 安全的标识符片段（非 ascii 字符用短哈希替代）。"""
    text = str(name).strip()
    if not text:
        return ""
    parts: list[str] = []
    for char in text:
        if char.isascii() and (char.isalnum() or char == "_"):
            parts.append(char)
        elif char.isspace() or char in ".-+*/\\|,;:!?()[]{}<>#'\"@#$%^&=":
            parts.append("_")
        else:  # 中文等非 ascii 字符：用码位保证同名稳定、异名不同
            parts.append(f"_{ord(char):x}_")
    raw = "".join(parts)
    raw = re.sub(r"_{2,}", "_", raw).strip("_")
    if not raw:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        raw = f"data_{digest}"
    if raw[0].isdigit():
        raw = f"ds_{raw}"
    return raw


def dataset_var_name(name: str) -> str:
    """把数据集名转成合法的 Python 变量名（中文/空格/重名都能处理）。

    * 非 ascii 字符（中文等）按其码位编码成 ascii 片段，同名结果稳定；
    * 与已使用的名字冲突时自动追加 ``_2``、``_3`` …；
    * 同时避开 Python 关键字与内置名（``import``、``list`` …）。
    """
    base = _ascii_name(name) or "data"
    if keyword.iskeyword(base) or base in {"np", "pd", "plt", "matplotlib", "fig", "ax"}:
        base = f"{base}_data"
    with _VAR_LOCK:
        candidate = base
        suffix = 2
        while candidate in _USED_VARS:
            candidate = f"{base}_{suffix}"
            suffix += 1
        _USED_VARS.add(candidate)
    return candidate


def _var_for(name: str, registry: dict[str, str]) -> str:
    """在 ``registry`` 中登记并返回数据集 ``name`` 对应的变量名。"""
    key = str(name or "")
    existing = registry.get(key)
    if existing is not None:
        return existing
    resolved = dataset_var_name(key)
    registry[key] = resolved
    return resolved


# --------------------------------------------------------------------------- #
# 源码文本小工具
# --------------------------------------------------------------------------- #
def _py_str(value: Any) -> str:
    """把任意值转成 Python 字符串字面量（中文与 ``$...$`` 公式安全）。"""
    return repr("" if value is None else str(value))


def _py_literal(value: Any) -> str:
    """把标量转成 Python 字面量（nan/inf 用 ``np.nan`` / ``np.inf``）。"""
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if math.isnan(number):
        return "np.nan"
    if math.isinf(number):
        return "np.inf" if number > 0 else "-np.inf"
    if number == int(number) and abs(number) < 1e15:
        return f"{number:.1f}"
    return repr(number)


def _num_token(value: Any, precision: int) -> str:
    """把数值转成源码里的 token；整数值保持整数形式，特殊值用 ``np.nan`` / ``np.inf``。"""
    if value is None:
        return "np.nan"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "np.nan"
        if math.isinf(number):
            return "np.inf" if number > 0 else "-np.inf"
        if number == int(number) and abs(number) < 1e15:
            return f"{int(number)}.0"
        text = repr(number)
        if precision > 0 and len(text) > precision + 8:
            shortened = f"{number:.{precision}g}"
            if "." in shortened or "e" in shortened or "E" in shortened:
                try:
                    if float(shortened) == number:
                        text = shortened
                except ValueError:  # pragma: no cover - 理论上不可达
                    pass
        return text
    return "np.nan"


def _element_tokens(values: np.ndarray, precision: int) -> list[str]:
    """把一维数组转换成源码 token 列表（字符串/布尔/数值分别处理）。"""
    array = np.asarray(values)
    if array.dtype.kind == "b":
        return ["True" if bool(v) else "False" for v in array]
    if array.dtype.kind in "iu":
        return [str(int(v)) for v in array]
    if array.dtype.kind == "f":
        return [_num_token(v, precision) for v in array]
    if array.dtype.kind == "c":
        return [f"complex({_num_token(v.real, precision)}, {_num_token(v.imag, precision)})" for v in array]
    if array.dtype.kind in "mM":
        return [_py_str(np.datetime_as_string(v, timezone="naive")) for v in array]
    tokens: list[str] = []
    for value in array.tolist():
        if value is None:
            tokens.append("None")
        elif isinstance(value, float) and math.isnan(value):
            tokens.append("np.nan")
        elif isinstance(value, (bool, np.bool_)):
            tokens.append("True" if bool(value) else "False")
        else:
            tokens.append(_py_str(value))
    return tokens


def _inline(tokens: Sequence[str]) -> str:
    """一行内拼接 token（保证字符串里有逗号也不会被拆坏）。"""
    return "[" + ", ".join(tokens) + "]"


def _wrap_array(
    tokens: Sequence[str],
    prefix: str,
    indent: str,
    suffix: str,
    *,
    pad: str = "",
) -> list[str]:
    """把 token 列表折行包进 ``[...]``，返回带缩进的多行。

    ``prefix`` 是第一行的前缀（例如 ``np.array(`` 或 ``    '列名': np.array(``），
    ``indent`` 是 ``[`` 所在的列（续行与 ``]`` 都对齐到这里），``pad`` 是
    在 ``[`` 与内容之间插入的缩进（把内容推到下一列）。
    """
    lines: list[str] = []
    current = f"{indent}{pad}["
    for index, token in enumerate(tokens):
        piece = token if index == 0 else f", {token}"
        if index == 0 and len(current) + len(piece) > _MAX_LINE - 1:
            # 前缀过长（例如容器键名 + pd.to_datetime(np.array()）：数据另起一行
            lines.append(current)
            current = f"{indent}{pad} {token}"
            continue
        if index > 0 and len(current) + len(piece) > _MAX_LINE:
            lines.append(current + ",")
            current = f"{indent}{pad} {token}"
            continue
        current += piece
    lines.append(current + "]")
    lines[-1] = lines[-1] + suffix
    return [prefix + lines[0]] + lines[1:]


def _array_literal(
    values: np.ndarray,
    options: CodeGenOptions,
    indent: str = "",
    prefix: str = "np.array(",
) -> list[str]:
    """生成 ``np.array([...])`` 字面量（可带 ``.reshape(...)`` / ``dtype=object``）。

    返回**多行**列表：第一行自带 ``prefix``，续行缩进到 ``indent``。
    ``array_format="compact"`` 时每行尽量放满（超过 :data:`_COMPACT_MIN`
    个元素时用 ``.reshape`` 还原形状）；``"pretty"`` 时每行一个数。
    """
    array = np.asarray(values)
    precision = max(1, int(options.precision or 6))
    flat = array.reshape(-1)
    dtype = "object" if array.ndim <= 1 and array.dtype.kind == "O" else None
    if dtype is None and array.dtype.kind == "U" and flat.size and any(v is None for v in flat.tolist()):
        dtype = "object"
    suffix = ""
    if array.ndim > 1:
        suffix = ".reshape(" + ", ".join(str(int(d)) for d in array.shape) + ")"
    suffix += ", dtype=object)" if dtype else ")"

    tokens = _element_tokens(flat, precision)
    if not tokens:
        return [f"{prefix}[]{suffix[:-1]}"]

    compact = str(options.array_format).lower() != "pretty"
    if not compact and len(tokens) > 1:  # pretty：每行一个数
        inner = ",\n".join(f"{indent}    {token}" for token in tokens)
        return [f"{prefix}[", inner + f",]{suffix}"]

    if len(tokens) < _COMPACT_MIN and len(prefix) + len(_inline(tokens)) <= _MAX_LINE:
        return [f"{prefix}{_inline(tokens)}{suffix}"]
    return _wrap_array(tokens, prefix, indent, suffix)


def _column_lines(
    column: str,
    series: pd.Series,
    options: CodeGenOptions,
    indent: str,
    prefix: str,
) -> list[str]:
    """把一列格式化成源码的 ``key: 表达式,`` 多行块。"""
    values = np.asarray(series.to_numpy())
    kind = getattr(series.dtype, "kind", "")
    if kind == "M" or pd.api.types.is_datetime64_any_dtype(series.dtype):
        texts = [None if pd.isna(v) else pd.Timestamp(v).isoformat() for v in series]
        tokens = ["None" if v is None else _py_str(v) for v in texts]
        if len(tokens) >= _COMPACT_MIN:
            return _wrap_array(tokens, f"{prefix}pd.to_datetime(np.array(", indent, ")),")
        return [f"{prefix}pd.to_datetime(np.array({_inline(tokens)})),"]
    if kind == "m" or pd.api.types.is_timedelta64_dtype(series.dtype):
        tokens = [_py_str(str(v)) for v in values]
        return [f"{prefix}pd.to_timedelta(np.array({_inline(tokens)})),"]
    if kind == "b":
        body = _array_literal(values, options, indent, f"{prefix}np.array(")
        body[-1] = body[-1] + ".astype(bool),"
        return body
    body = _array_literal(values, options, indent, f"{prefix}np.array(")
    body[-1] = body[-1] + ","
    return body


# --------------------------------------------------------------------------- #
# 数据集代码块
# --------------------------------------------------------------------------- #
def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """数据集中可当数值绘制的列（数值列、时间列、可转数值的列）。"""
    out: list[str] = []
    for column in frame.columns:
        dtype = frame[column].dtype
        if getattr(dtype, "kind", "") in "biufcMm":
            out.append(str(column))
            continue
        try:
            pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError, OverflowError):
            continue
        out.append(str(column))
    return out


def _dataset_block(
    name: str,
    variable: str,
    frame: pd.DataFrame | None,
    options: CodeGenOptions,
    comments: bool,
    indent: str = "",
) -> list[str]:
    """生成一个数据集的源码块（字典形式，列名做键）。"""
    lines: list[str] = []
    if comments:
        lines.append(f"{indent}# ---- 数据集：{name or '（未命名）'} ----")
    if not len(frame.columns):
        lines.append(f"{indent}{variable} = {{}}")
        lines.append("")
        return lines
    lines.append(f"{indent}{variable} = {{")
    for column in frame.columns:
        prefix = f"{indent}    {_py_str(str(column))}: "
        lines += _column_lines(str(column), frame[column], options, f"{indent}    ", prefix)
    lines.append(f"{indent}}}")
    numeric = set(_numeric_columns(frame))
    text = [str(c) for c in frame.columns if str(c) not in numeric]
    if text and comments:
        lines.append(f"{indent}# 说明：文本列按原样内联（'甲'、'乙'…），"
                     f"matplotlib 会自动当作类别标签使用。")
    lines.append("")
    return lines


def _csv_block(
    name: str,
    variable: str,
    frame: pd.DataFrame | None,
    options: CodeGenOptions,
    comments: bool,
    indent: str = "",
) -> list[str]:
    """生成从 CSV 读取数据集的源码块。"""
    path = str((options.csv_paths or {}).get(name, "") or "")
    if not path:
        safe = _ascii_name(name) or "data"
        path = f"{safe}.csv"
    lines: list[str] = []
    if comments:
        lines.append(f"{indent}# ---- 数据集：{name or '（未命名）'}（从 CSV 读取） ----")
    lines.append(f"{indent}{variable} = pd.read_csv({_py_str(path)}, encoding=\"utf-8\")")
    if comments:
        wanted = [str(c) for c in frame.columns]
        if wanted:
            lines.append(f"{indent}# 需要的列：{'、'.join(wanted)}")
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# 网格化辅助（生成「运行时」代码而不是计算结果）
# --------------------------------------------------------------------------- #
def _is_regular_grid(x: Any, y: Any) -> bool:
    """判断 (x, y) 是否构成严格笛卡尔积网格（可无损 reshape）。"""
    xa = _numeric_array(x)
    ya = _numeric_array(y)
    if xa.size != ya.size or xa.size < 4:
        return False
    ux = np.unique(xa[np.isfinite(xa)])
    uy = np.unique(ya[np.isfinite(ya)])
    if ux.size < 2 or uy.size < 2 or ux.size * uy.size != xa.size:
        return False
    return np.unique(np.stack([xa, ya]), axis=1).shape[1] == xa.size


def _numeric_array(values: Any) -> np.ndarray:
    """把任意一列转成浮点数组（文本列按出现顺序编码为整数，与绘图引擎一致）。"""
    array = np.asarray(values)
    if array.dtype.kind in "biufc":
        return array.astype(float)
    if array.dtype.kind in "mM":
        return array.astype("datetime64[ns]").astype("int64").astype(float)
    codes, _ = pd.factorize(pd.Series(array.reshape(-1)), sort=False)
    return np.asarray(codes, dtype=float)


def _grid_lines(x_expr: str, y_expr: str, z_expr: str, indent: str, *, var: str = "_G") -> list[str]:
    """生成「笛卡尔积 -> 网格」的运行时代码（行优先，y 为行）。"""
    return [
        f"{indent}_ux = np.unique({x_expr})",
        f"{indent}_uy = np.unique({y_expr})",
        f"{indent}{var} = np.asarray({z_expr}).reshape(_uy.size, _ux.size)",
        f"{indent}_gx, _gy = np.meshgrid(_ux, _uy)",
    ]


# --------------------------------------------------------------------------- #
# 序列绘制代码
# --------------------------------------------------------------------------- #
def _axis_var(index: int) -> str:
    return "ax" if index == 0 else f"ax{index + 1}"


def _label_list(series: SeriesSpec, columns: Sequence[str], variable: str) -> list[str]:
    """复现 :func:`core.plotting._series_labels` 的命名规则。"""
    base = str(series.label or "").strip() or str(series.dataset or "").strip()
    out: list[str] = []
    for column in columns:
        column = str(column).replace(",", r"\,")
        out.append(f"{base}.{column}" if base else column)
    return out


def _error_expr(series: SeriesSpec, frame: pd.DataFrame | None, ds: str) -> tuple[str | None, str | None]:
    """解析误差列，返回 ``(误差表达式, 提示)``。"""
    raw = str(series.yerr or "").strip()
    if raw:
        names = [c.strip() for c in raw.split(",") if c.strip()]
        missing = [n for n in names if n not in frame.columns]
        if missing:
            return None, f"找不到误差列「{'、'.join(missing)}」，已按普通折线绘制。"
        if len(names) == 1:
            return f"{ds}[{_py_str(names[0])}]", None
        items = ", ".join(f"{ds}[{_py_str(n)}]" for n in names)
        return f"np.column_stack([{items}])", None
    y_columns = series.y_columns()
    for column in y_columns:
        candidates = [f"{column}_err", f"{column}_error", f"{column}_std", f"{column}_sd"]
        for candidate in candidates:
            if candidate in frame.columns:
                return f"{ds}[{_py_str(candidate)}]", None
    return None, None


def _series_comment(index: int, series: SeriesSpec, frame: pd.DataFrame | None) -> str:
    """``# 序列 1：折线图（数据集 sensor，x=time，y=value）``"""
    parts: list[str] = []
    if series.dataset:
        parts.append(f"数据集 {series.dataset}")
    for key, label in (("x", "x"), ("y", "y"), ("z", "z"), ("u", "u"), ("v", "v"), ("w", "w"), ("yerr", "误差")):
        value = str(getattr(series, key, "") or "").strip()
        if value:
            parts.append(f"{label}={value}")
    if series.expr:
        parts.append(f"expr={series.expr}")
    detail = "，".join(parts)
    return f"# 序列 {index}：{kind_label(series.kind)}（{detail}）" if detail else f"# 序列 {index}：{kind_label(series.kind)}"


def _normalize_lines(series: SeriesSpec, vars_: Mapping[str, str], indent: str) -> list[str]:
    """向量归一化代码。"""
    if not series.normalize:
        return []
    kind = str(series.kind or "").lower()
    u, v, w = vars_.get("u", ""), vars_.get("v", ""), vars_.get("w", "")
    if kind == "quiver":
        return [
            f"{indent}_mag = np.hypot({u}, {v})",
            f"{indent}_mag[_mag == 0] = 1.0",
            f"{indent}{u}, {v} = {u} / _mag, {v} / _mag",
        ]
    return [
        f"{indent}_mag = np.sqrt({u} ** 2 + {v} ** 2 + {w} ** 2)",
        f"{indent}_mag[_mag == 0] = 1.0",
        f"{indent}{u}, {v}, {w} = {u} / _mag, {v} / _mag, {w} / _mag",
    ]


def _color_expr(series: SeriesSpec, index: int) -> str | None:
    """非空颜色才返回；多列时后续列由 matplotlib 自动配色。"""
    color = str(series.color or "").strip()
    if not color:
        return None
    return color


def _row_reference(series: SeriesSpec, frame: pd.DataFrame | None, ds: str) -> str:
    """取数据集里任意一列，用来得到行数（x 列缺失时回退行号）。"""
    if frame is not None and len(frame.columns):
        return f"{ds}[{_py_str(str(frame.columns[0]))}]"
    return f"next(iter({ds}.values()))"


def _x_expression(series: SeriesSpec, frame: pd.DataFrame | None, ds: str) -> str:
    """x 轴表达式：有 x 列就用它，否则回退为行号（与绘图引擎一致）。"""
    column = str(series.x or "").strip()
    if column and (frame is None or column in frame.columns):
        return f"_codes({ds}[{_py_str(column)}])"
    return f"np.arange(len({_row_reference(series, frame, ds)}))"


def _draw_lines(
    series: SeriesSpec,
    ax: str,
    ds: str,
    frame: pd.DataFrame | None,
    options: CodeGenOptions,
    comments: bool,
    indent: str = "",
) -> list[str]:
    """常规 2D 图表（line/scatter/bar/barh/stem/step/fill/errorbar）。"""
    kind = str(series.kind or "line").lower()
    y_columns = series.y_columns()
    labels = _label_list(series, y_columns, ds)
    zorder = int(series.zorder)
    alpha = float(min(max(float(series.alpha or 1.0), 0.0), 1.0))
    linewidth = float(series.linewidth)
    markersize = float(series.markersize)
    marker = str(series.marker or "")
    linestyle = str(series.linestyle if series.linestyle is not None else "-")
    width = float(series.bar_width) if series.bar_width else 0.8
    horizontal = kind == "barh" or (kind == "bar" and bool(series.horizontal))
    kind = "barh" if horizontal else kind

    lines: list[str] = []
    lines.append(f"{indent}x = {_x_expression(series, frame, ds)}")
    if comments:
        lines.append(f"{indent}# {kind_label(kind)}")

    multi = len(y_columns) > 1
    for index, column in enumerate(y_columns):
        if multi:
            lines.append(f"{indent}# 第 {index + 1} 个 y 列：{column}")
        lines.append(f"{indent}y = _codes({ds}[{_py_str(column)}])")
        if str(series.expr or "").strip():
            expression = str(series.expr).strip()
            lines.append(f"{indent}try:")
            lines.append(
                f"{indent}    y = _codes(eval({_py_str(expression)}, "
                f"{{'__builtins__': {{}}, 'np': np, 'x': x, 'y': y, 'i': np.arange(len(y))}}))"
            )
            lines.append(f"{indent}except Exception:")
            lines.append(f"{indent}    pass  # 表达式无法求值时按原始列绘制")
        if float(series.scale or 1.0) != 1.0:
            lines.append(f"{indent}y = y * {_py_literal(float(series.scale))}")
        color = _color_expr(series, index)
        color_arg = f", color={_py_str(color)}" if color else ""
        label = labels[index] if index < len(labels) else column
        label_arg = f", label={_py_str(label)}"
        common = (
            f"{color_arg}, alpha={_py_literal(alpha)}, zorder={zorder}{label_arg}"
        )
        if kind in {"bar", "barh"}:
            # 多列时并排分组；单列时 _offset 为 0，仍然写出来避免沿用上一条序列的旧值
            bar_width = width / max(1, len(y_columns))
            lines.append(
                f"{indent}_offset = {_py_literal((index - (len(y_columns) - 1) / 2.0) * bar_width)}"
            )
        if kind == "line":
            args = f"x, y{color_arg}, linestyle={_py_str(linestyle or 'None')}, linewidth={_py_literal(linewidth)}, marker={_py_str(marker) or 'None'}, markersize={_py_literal(markersize)}, alpha={_py_literal(alpha)}, zorder={zorder}{label_arg}"
            lines.append(f"{indent}{ax}.plot({args})")
        elif kind == "scatter":
            args = f"x, y{color_arg}, marker={_py_str(marker or 'o')}, s={_py_literal(max(1.0, markersize ** 2))}, alpha={_py_literal(alpha)}, zorder={zorder}{label_arg}"
            lines.append(f"{indent}{ax}.scatter({args})")
        elif kind == "bar":
            lines.append(
                f"{indent}{ax}.bar(x + _offset, y, width={_py_literal(bar_width)}{common})"
            )
        elif kind == "barh":
            lines.append(
                f"{indent}{ax}.barh(x + _offset, y, height={_py_literal(bar_width)}{common})"
            )
        elif kind == "step":
            args = f"x, y, where=\"pre\"{color_arg}, linestyle={_py_str(linestyle or '-')}, linewidth={_py_literal(linewidth)}, marker={_py_str(marker) or 'None'}, markersize={_py_literal(markersize)}, alpha={_py_literal(alpha)}, zorder={zorder}{label_arg}"
            lines.append(f"{indent}{ax}.step({args})")
        elif kind == "stem":
            stem_color = color or _FILL_CYCLE[index % len(_FILL_CYCLE)]
            lines.append(
                f"{indent}{ax}.vlines(x, 0.0, y, color={_py_str(stem_color)}, "
                f"linewidth={_py_literal(max(0.5, linewidth * 0.6))}, alpha={_py_literal(alpha)}, zorder={zorder})"
            )
            lines.append(
                f"{indent}{ax}.plot(x, y, color={_py_str(stem_color)}, linestyle=\"None\", "
                f"marker={_py_str(marker or 'o')}, markersize={_py_literal(markersize)}, "
                f"alpha={_py_literal(alpha)}, zorder={zorder + 1}{label_arg})"
            )
        elif kind == "fill":
            fill_color = color or _FILL_CYCLE[index % len(_FILL_CYCLE)]
            fill_alpha = alpha if alpha < 1.0 else 0.35
            lines.append(
                f"{indent}{ax}.fill_between(x, y, 0.0, color={_py_str(fill_color)}, "
                f"alpha={_py_literal(fill_alpha)}, zorder={zorder}{label_arg})"
            )
            lines.append(
                f"{indent}{ax}.plot(x, y, color={_py_str(fill_color)}, linestyle={_py_str(linestyle or '-')}, "
                f"linewidth={_py_literal(linewidth)}, marker={_py_str(marker) or 'None'}, "
                f"markersize={_py_literal(markersize)}, alpha={_py_literal(alpha)}, zorder={zorder + 1})"
            )
        elif kind == "errorbar":
            err_expr, note = _error_expr(series, frame, ds)
            if note and comments:
                lines.append(f"{indent}# 提示：{note}")
            caps = float((series.options or {}).get("capsize", _DEFAULT_CAPSIZE))
            if err_expr:
                lines.append(f"{indent}_err = _codes({err_expr})")
                err_arg = ", yerr=_err"
            else:
                lines.append(f"{indent}_err = None")
                err_arg = ""
            lines.append(
                f"{indent}{ax}.errorbar(x, y{err_arg}{color_arg}, linestyle={_py_str(linestyle or 'None')}, "
                f"linewidth={_py_literal(linewidth)}, marker={_py_str(marker or 'o')}, "
                f"markersize={_py_literal(markersize)}, capsize={_py_literal(caps)}, "
                f"alpha={_py_literal(alpha)}, zorder={zorder}{label_arg})"
            )
        else:  # 兜底：按折线绘制（与 core.plotting 的降级策略一致）
            lines.append(f"{indent}# 未知类型「{kind}」，已按折线图绘制")
            lines.append(
                f"{indent}{ax}.plot(x, y{color_arg}, linewidth={_py_literal(linewidth)}, "
                f"alpha={_py_literal(alpha)}, zorder={zorder}{label_arg})"
            )
    return lines


def _draw_hist(
    series: SeriesSpec, ax: str, ds: str, frame: pd.DataFrame | None, comments: bool, indent: str = ""
) -> list[str]:
    """直方图。"""
    y_columns = series.y_columns()
    labels = _label_list(series, y_columns, ds)
    alpha = float(min(max(float(series.alpha or 1.0), 0.0), 1.0))
    lines: list[str] = [f"{indent}# {kind_label('hist')}"]
    for index, column in enumerate(y_columns):
        color = _color_expr(series, index)
        color_arg = f", color={_py_str(color)}" if color else ""
        lines.append(f"{indent}_data = _codes({ds}[{_py_str(column)}])")
        lines.append(f"{indent}_data = _data[np.isfinite(_data)]")
        lines.append(
            f"{indent}{ax}.hist(_data, bins={max(1, int(series.bins))}, "
            f"density={bool(series.density)}, cumulative={bool(series.cumulative)}, "
            f"orientation={_py_str('horizontal' if series.horizontal else 'vertical')}, "
            f"histtype={_py_str('stepfilled' if series.filled else 'step')}, "
            f"alpha={_py_literal(alpha)}, label={_py_str(labels[index])}{color_arg})"
        )
    lines.append(f"{indent}{ax}.set_ylabel({_py_str('概率密度' if series.density else '频数')})")
    return lines


def _draw_box(
    series: SeriesSpec, ax: str, ds: str, frame: pd.DataFrame | None, indent: str = ""
) -> list[str]:
    """箱线图（每个 y 列一个箱体）。"""
    y_columns = series.y_columns()
    labels = _label_list(series, y_columns, ds)
    lines: list[str] = [f"{indent}# {kind_label('box')}"]
    arrays: list[str] = []
    for index, column in enumerate(y_columns):
        name = f"_box{index + 1}"
        arrays.append(name)
        lines.append(f"{indent}{name} = _codes({ds}[{_py_str(column)}])")
        lines.append(f"{indent}{name} = {name}[np.isfinite({name})]")
    names = ", ".join(_py_str(label) for label in labels)
    lines.append(
        f"{indent}{ax}.boxplot([{', '.join(arrays)}], tick_labels=[{names}], patch_artist=True)"
    )
    if series.horizontal:
        lines.append(f"{indent}{ax}.set_ylabel({_py_str('数值')})")
    return lines


def _draw_pie(
    series: SeriesSpec, ax: str, ds: str, frame: pd.DataFrame | None, indent: str = ""
) -> list[str]:
    """饼图（x 列作为扇形标签）。"""
    y_columns = series.y_columns()
    lines: list[str] = [f"{indent}# {kind_label('pie')}"]
    lines.append(f"{indent}_values = _codes({ds}[{_py_str(y_columns[0])}])")
    lines.append(f"{indent}_mask = np.isfinite(_values)")
    lines.append(f"{indent}_values = np.abs(_values[_mask])")
    if series.x and frame is not None and series.x in frame.columns:
        lines.append(f"{indent}_wedge_labels = [str(v) for v in np.asarray({ds}[{_py_str(series.x)}])[_mask]]")
        labels_arg = ", labels=_wedge_labels"
    else:
        labels_arg = ""
    autopct = (series.options or {}).get("autopct")
    autopct_arg = f", autopct={_py_str(autopct)}" if autopct else ""
    startangle = float((series.options or {}).get("startangle", _PIE_STARTANGLE))
    lines.append(
        f"{indent}{ax}.pie(_values{labels_arg}{autopct_arg}, startangle={_py_literal(startangle)})"
    )
    lines.append(f"{indent}{ax}.set_aspect(\"equal\")")
    return lines


def _draw_quiver(
    series: SeriesSpec, ax: str, ds: str, frame: pd.DataFrame | None, comments: bool, indent: str = ""
) -> list[str]:
    """二维向量图。"""
    lines: list[str] = [f"{indent}# {kind_label('quiver')}"]
    lines.append(f"{indent}x = {_x_expression(series, frame, ds)}")
    if series.y and frame is not None and series.y in frame.columns:
        lines.append(f"{indent}y = _codes({ds}[{_py_str(series.y)}])")
    else:
        lines.append(f"{indent}y = np.zeros_like(x, dtype=float)")
    lines.append(f"{indent}u = _codes({ds}[{_py_str(series.u)}])")
    lines.append(f"{indent}v = _codes({ds}[{_py_str(series.v)}])")
    if series.normalize:
        lines.append(f"{indent}_mag = np.hypot(u, v)")
        lines.append(f"{indent}_mag[_mag == 0] = 1.0")
        lines.append(f"{indent}u, v = u / _mag, v / _mag")
    scale = float(series.scale) if series.scale else 1.0
    color = _color_expr(series, 0)
    color_arg = f", color={_py_str(color)}" if color else ""
    label = str(series.label or "").strip()
    label_arg = f", label={_py_str(label)}" if label else ""
    lines.append(
        f"{indent}{ax}.quiver(x, y, u, v, angles=\"xy\", scale_units=\"xy\", "
        f"scale={_py_literal(1.0 / scale) if scale else '1.0'}, "
        f"alpha={_py_literal(float(series.alpha or 1.0))}{color_arg}{label_arg})"
    )
    return lines


def _draw_contour(
    series: SeriesSpec, ax: str, ds: str, frame: pd.DataFrame | None, spec: PlotSpec, indent: str = ""
) -> list[str]:
    """等高线图（规则网格用 contourf，散点用 tricontourf）。"""
    y_columns = series.y_columns()
    y_name = y_columns[0] if y_columns else ""
    x_name = series.x if series.x and frame is not None and series.x in frame.columns else ""
    lines: list[str] = [f"{indent}# {kind_label('contour')}"]
    x_expr = _x_expression(series, frame, ds)
    y_expr = f"_codes({ds}[{_py_str(y_name)}])" if y_name else x_expr
    z_expr = f"_codes({ds}[{_py_str(series.z)}])"
    lines.append(f"{indent}_x, _y, _z = {x_expr}, {y_expr}, {z_expr}")
    level = max(2, int(series.levels))
    cmap = series.cmap or "viridis"
    alpha = float(series.alpha or 1.0)
    if x_name and y_name and _is_regular_grid(frame[x_name].to_numpy(), frame[y_name].to_numpy()):
        lines.append(f"{indent}_ux, _uy = np.unique(_x), np.unique(_y)")
        lines.append(f"{indent}_Z = _z.reshape(_uy.size, _ux.size)")
        lines.append(f"{indent}_gx, _gy = np.meshgrid(_ux, _uy)")
        call = "contourf" if series.filled else "contour"
        lines.append(
            f"{indent}_last = {ax}.{call}(_gx, _gy, _Z, levels={level}, cmap={_py_str(cmap)}, alpha={_py_literal(alpha)})"
        )
    else:
        lines.append(f"{indent}_tri = matplotlib.tri.Triangulation(_x, _y)")
        call = "tricontourf" if series.filled else "tricontour"
        lines.append(
            f"{indent}_last = {ax}.{call}(_tri, _z, levels={level}, cmap={_py_str(cmap)}, alpha={_py_literal(alpha)})"
        )
    if spec.colorbar:
        lines.append(f"{indent}_mappables.append(_last)")
    return lines


def _draw_heatmap(
    series: SeriesSpec, ax: str, ds: str, frame: pd.DataFrame | None, spec: PlotSpec, indent: str = ""
) -> list[str]:
    """热力图（规则网格用 pcolormesh，散点退化为彩色散点）。"""
    y_columns = series.y_columns()
    y_name = y_columns[0] if y_columns else ""
    x_name = series.x if series.x and frame is not None and series.x in frame.columns else ""
    lines: list[str] = [f"{indent}# {kind_label('heatmap')}"]
    x_expr = _x_expression(series, frame, ds)
    y_expr = f"_codes({ds}[{_py_str(y_name)}])" if y_name else x_expr
    lines.append(f"{indent}_x, _y = {x_expr}, {y_expr}")
    lines.append(f"{indent}_z = _codes({ds}[{_py_str(series.z)}])")
    cmap = series.cmap or "viridis"
    alpha = float(series.alpha or 1.0)
    if x_name and y_name and _is_regular_grid(frame[x_name].to_numpy(), frame[y_name].to_numpy()):
        lines.append(f"{indent}_ux, _uy = np.unique(_x), np.unique(_y)")
        lines.append(f"{indent}_Z = _z.reshape(_uy.size, _ux.size)")
        lines.append(
            f"{indent}_last = {ax}.pcolormesh(_ux, _uy, _Z, cmap={_py_str(cmap)}, "
            f"shading=\"auto\", alpha={_py_literal(alpha)})"
        )
    else:
        lines.append(
            f"{indent}_last = {ax}.scatter(_x, _y, c=_z, cmap={_py_str(cmap)}, "
            f"alpha={_py_literal(alpha)})"
        )
    if spec.colorbar:
        lines.append(f"{indent}_mappables.append(_last)")
    return lines


def _draw_3d(
    series: SeriesSpec,
    ax: str,
    ds: str,
    frame: pd.DataFrame | None,
    spec: PlotSpec,
    comments: bool,
    indent: str = "",
) -> list[str]:
    """三维序列绘制。"""
    kind = str(series.kind or "line3d").lower()
    y_columns = series.y_columns()
    alpha = float(min(max(float(series.alpha or 1.0), 0.0), 1.0))
    linewidth = float(series.linewidth)
    markersize = float(series.markersize)
    marker = str(series.marker or "")
    labels = _label_list(series, y_columns, ds)
    lines: list[str] = []

    x_expr = _x_expression(series, frame, ds)
    z_expr = (
        f"_codes({ds}[{_py_str(series.z)}])"
        if series.z and frame is not None and series.z in frame.columns
        else "np.zeros_like(_x, dtype=float)"
    )

    if kind in _GRID_3D:
        y_name = y_columns[0] if y_columns else ""
        y_expr = (
            f"_codes({ds}[{_py_str(y_name)}])"
            if y_name
            else _x_expression(dataclasses.replace(series, x=""), frame, ds)
        )
        lines.append(f"{indent}_x, _y = {x_expr}, {y_expr}")
        lines.append(f"{indent}_z = _codes({ds}[{_py_str(series.z)}])")
        cmap = series.cmap or "viridis"
        color = _color_expr(series, 0)
        regular = bool(
            series.x
            and y_name
            and frame is not None and series.x in frame.columns
            and _is_regular_grid(frame[series.x].to_numpy(), frame[y_name].to_numpy())
        )
        if regular:
            lines.append(f"{indent}_ux, _uy = np.unique(_x), np.unique(_y)")
            lines.append(f"{indent}_gx, _gy = np.meshgrid(_ux, _uy)")
            lines.append(f"{indent}_Z = _z.reshape(_uy.size, _ux.size)")
            if kind == "surface":
                edge = "\"none\"" if series.filled else "None"
                lines.append(
                    f"{indent}_last = {ax}.plot_surface(_gx, _gy, _Z, cmap={_py_str(cmap)}, "
                    f"alpha={_py_literal(alpha)}, linewidth={_py_literal(0.0 if series.filled else max(0.0, linewidth / 4.0))}, "
                    f"edgecolor={edge}, antialiased=True)"
                )
            elif kind == "wireframe":
                color_arg = f"color={_py_str(color)}, " if color else ""
                lines.append(
                    f"{indent}_last = {ax}.plot_wireframe(_gx, _gy, _Z, {color_arg}"
                    f"linewidth={_py_literal(max(0.2, linewidth / 2.0))}, alpha={_py_literal(alpha)})"
                )
            else:
                lines.append(
                    f"{indent}_last = {ax}.contour3D(_gx, _gy, _Z, levels={max(2, int(series.levels))}, "
                    f"cmap={_py_str(cmap)}, alpha={_py_literal(alpha)})"
                )
        else:
            if comments:
                lines.append(f"{indent}# 数据点不构成规则网格，已降级为三角剖分绘制")
            if kind == "surface":
                lines.append(
                    f"{indent}_last = {ax}.plot_trisurf(_x, _y, _z, cmap={_py_str(cmap)}, "
                    f"alpha={_py_literal(alpha)}, linewidth={_py_literal(max(0.2, linewidth / 4.0))}, antialiased=True)"
                )
            elif kind == "wireframe":
                lines.append(
                    f"{indent}_last = {ax}.plot_trisurf(_x, _y, _z, cmap={_py_str(cmap)}, "
                    f"alpha={_py_literal(alpha)}, linewidth={_py_literal(max(0.2, linewidth / 2.0))})"
                )
            else:
                lines.append(f"{indent}_tri = matplotlib.tri.Triangulation(_x, _y)")
                lines.append(
                    f"{indent}_last = {ax}.tricontour(_tri, _z, levels={max(2, int(series.levels))}, "
                    f"cmap={_py_str(cmap)}, alpha={_py_literal(alpha)})"
                )
        if spec.colorbar and kind in {"surface", "wireframe", "contour3d"}:
            lines.append(f"{indent}_mappables.append(_last)")
        return lines

    if kind == "quiver3d":
        y_name = y_columns[0] if y_columns else ""
        lines.append(f"{indent}_x = {x_expr}")
        lines.append(f"{indent}_y = _codes({ds}[{_py_str(y_name)}])" if y_name
                     else f"{indent}_y = np.zeros_like(_x, dtype=float)")
        lines.append(f"{indent}_z = {z_expr}")
        for name, key in (("_u", "u"), ("_v", "v"), ("_w", "w")):
            lines.append(f"{indent}{name} = _codes({ds}[{_py_str(getattr(series, key))}])")
        if series.normalize:
            lines.append(f"{indent}_mag = np.sqrt(_u ** 2 + _v ** 2 + _w ** 2)")
            lines.append(f"{indent}_mag[_mag == 0] = 1.0")
            lines.append(f"{indent}_u, _v, _w = _u / _mag, _v / _mag, _w / _mag")
        scale = float(series.scale) if series.scale else 1.0
        color = _color_expr(series, 0)
        color_arg = f", color={_py_str(color)}" if color else ""
        lines.append(
            f"{indent}{ax}.quiver(_x, _y, _z, _u, _v, _w, length={_py_literal(1.0 / scale)}, "
            f"normalize=False, alpha={_py_literal(alpha)}{color_arg})"
        )
        return lines

    if kind == "bar3d":
        lines.append(f"{indent}_x = {x_expr}")
        lines.append(f"{indent}_z0 = {z_expr}")
        depth = float(series.bar_width) if series.bar_width else 0.8
        for index, column in enumerate(y_columns):
            color = _color_expr(series, index) or _FILL_CYCLE[index % len(_FILL_CYCLE)]
            lines.append(f"{indent}_height = _codes({ds}[{_py_str(column)}])")
            lines.append(
                f"{indent}{ax}.bar3d(_x - {_py_literal(depth / 2.0)}, np.zeros_like(_x), _z0, "
                f"{_py_literal(depth)}, {_py_literal(depth)}, _height, "
                f"color={_py_str(color)}, alpha={_py_literal(alpha)}, shade=True)"
            )
        return lines

    if kind == "trisurf":
        lines.append(f"{indent}_x = {x_expr}")
        y_name = y_columns[0] if y_columns else ""
        if y_name:
            lines.append(f"{indent}_y = _codes({ds}[{_py_str(y_name)}])")
        else:
            lines.append(f"{indent}_y = np.arange(len(_x), dtype=float)")
        lines.append(f"{indent}_z = {z_expr}")
        lines.append(
            f"{indent}_last = {ax}.plot_trisurf(_x, _y, _z, cmap={_py_str(series.cmap or 'viridis')}, "
            f"alpha={_py_literal(alpha)})"
        )
        if spec.colorbar:
            lines.append(f"{indent}_mappables.append(_last)")
        return lines

    # ---- 三维点 / 线 ------------------------------------------------------ #
    lines.append(f"{indent}_x = {x_expr}")
    lines.append(f"{indent}_z = {z_expr}")
    for index, column in enumerate(y_columns):
        color = _color_expr(series, index)
        color_arg = f", color={_py_str(color)}" if color else ""
        label = labels[index] if index < len(labels) else column
        lines.append(f"{indent}_y = _codes({ds}[{_py_str(column)}])")
        if kind == "scatter3d":
            lines.append(
                f"{indent}{ax}.scatter(_x, _y, _z{color_arg}, marker={_py_str(marker or 'o')}, "
                f"s={_py_literal(max(1.0, markersize ** 2))}, alpha={_py_literal(alpha)}, label={_py_str(label)})"
            )
        else:
            lines.append(
                f"{indent}{ax}.plot(_x, _y, _z{color_arg}, linestyle={_py_str(str(series.linestyle or '-'))}, "
                f"linewidth={_py_literal(linewidth)}, marker={_py_str(marker) or 'None'}, "
                f"markersize={_py_literal(markersize)}, alpha={_py_literal(alpha)}, label={_py_str(label)})"
            )
    return lines


def _series_lines(
    series: SeriesSpec,
    ds_var: str,
    frame: pd.DataFrame | None,
    options: CodeGenOptions,
    spec: PlotSpec | None,
    index: int,
    comments: bool,
    axis: str,
    indent: str = "",
) -> list[str]:
    """把一条序列转成绘图代码行（不含前置注释/空行）。

    ``frame`` 为 ``None`` 时按「数据集列名未知」处理（界面查看源码时用），
    此时 x 回退为行号表达式。
    """
    data = frame if isinstance(frame, pd.DataFrame) else None
    projection = "3d" if (spec is not None and spec.is_3d()) else "2d"
    kind = str(series.kind or "").lower()
    lines: list[str] = []

    if projection == "3d" or kind in KINDS_3D:
        available = KINDS_3D
    else:
        available = KINDS_2D
    if kind not in available:
        fallback = "line3d" if projection == "3d" else "line"
        if comments:
            lines.append(f"{indent}# 不支持的图表类型「{kind}」，已按{kind_label(fallback)}绘制")
        series = dataclasses.replace(series, kind=fallback)
        kind = fallback

    if projection == "3d":
        return lines + _draw_3d(series, axis, ds_var, data, spec or PlotSpec(), comments, indent)

    if kind == "hist":
        return lines + _draw_hist(series, axis, ds_var, data, comments, indent)
    if kind == "box":
        return lines + _draw_box(series, axis, ds_var, data, indent)
    if kind == "pie":
        return lines + _draw_pie(series, axis, ds_var, data, indent)
    if kind == "quiver":
        return lines + _draw_quiver(series, axis, ds_var, data, comments, indent)
    if kind == "contour":
        return lines + _draw_contour(series, axis, ds_var, data, spec or PlotSpec(), indent)
    if kind == "heatmap":
        return lines + _draw_heatmap(series, axis, ds_var, data, spec or PlotSpec(), indent)
    return lines + _draw_lines(series, axis, ds_var, data, options, comments, indent)


def generate_series_code(series: SeriesSpec, ds_var: str) -> list[str]:
    """生成单条序列的绘图代码行（供界面「查看当前序列源码」用）。

    ``ds_var`` 是该数据集在生成源码里的变量名（见 :func:`dataset_var_name`）；
    返回的行已经带 4 空格缩进，可以直接贴进脚本的绘图区。
    """
    if not isinstance(series, SeriesSpec):
        raise TypeError("series 必须是 core.spec.SeriesSpec")
    variable = str(ds_var or "").strip() or dataset_var_name(series.dataset or "data")
    spec = PlotSpec(projection="3d" if str(series.kind).lower() in KINDS_3D else "2d")
    spec.series = [series]
    with _VAR_LOCK:
        snapshot = set(_USED_VARS)
        _USED_VARS.add(variable)
    try:
        return _series_lines(series, variable, None, CodeGenOptions(), spec, 0, True, "ax", "    ")
    finally:
        with _VAR_LOCK:
            _USED_VARS.clear()
            _USED_VARS.update(snapshot)


# --------------------------------------------------------------------------- #
# 头部 / 尾部 / 坐标轴
# --------------------------------------------------------------------------- #
def _header(spec: PlotSpec, options: CodeGenOptions, datasets: Mapping[str, pd.DataFrame]) -> list[str]:
    """生成中文头部注释。"""
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    projection = "三维（3d）" if spec.is_3d() else "二维（2d）"
    kinds = "、".join(
        f"{kind_label(s.kind)}" for s in spec.series
    ) or "（无序列）"
    names = "、".join(str(n) for n in datasets.keys()) or "（无）"
    return [
        "# -*- coding: utf-8 -*-",
        '"""由 TableDrawer 导出的 matplotlib 源码。',
        "",
        f"生成时间：{now}",
        f"投影方式：{projection}；图表类型：{kinds}",
        f"主题：{spec.theme or 'light'}；画布：{spec.figsize[0]} x {spec.figsize[1]} 英寸 @ {spec.dpi} dpi",
        f"数据集：{names}",
        "",
        "本文件是自包含的：只依赖 numpy / pandas / matplotlib，不依赖 TableDrawer 项目本身，",
        "可以直接用 python 运行，也可以复制进其它工程继续修改。",
        '"""',
        "",
    ]


def _imports(spec: PlotSpec, options: CodeGenOptions) -> list[str]:
    """生成 import 区与 ``_codes`` 运行时小工具。"""
    lines = ["import numpy as np", "import pandas as pd"]
    if spec.is_3d():
        lines.append("import matplotlib")
        lines.append("import matplotlib.pyplot as plt")
        lines.append("from mpl_toolkits.mplot3d import Axes3D  # noqa: F401")
    else:
        lines.append("import matplotlib")
        lines.append("import matplotlib.pyplot as plt")
    lines += [
        "",
        "",
        "def _codes(values):",
        '    """任意一列 -> 浮点数组：数值列原样，文本列按出现顺序编码为整数。"""',
        "    array = np.asarray(values)",
        "    if array.dtype.kind in \"biufc\":",
        "        return array.astype(float)",
        "    if array.dtype.kind in \"mM\":  # 时间列 -> 数值时间轴",
        "        return array.astype(\"datetime64[ns]\").astype(\"int64\").astype(float)",
        "    codes, _ = pd.factorize(pd.Series(array.reshape(-1)), sort=False)",
        "    return np.asarray(codes, dtype=float)",
        "",
    ]
    return lines


def _font_block(spec: PlotSpec, indent: str = "") -> list[str]:
    """生成中文字体设置块（内联，不依赖 core.mplsetup）。"""
    family = str(spec.font_family or "").strip()
    preferred = [family] if family else []
    candidates = preferred + [f for f in CJK_FONT_CANDIDATES if f != family]
    joined = ", ".join(_py_str(f) for f in candidates)
    i = indent
    lines = [
        f"{i}# ---- 中文字体与数学字体（保证标题/标签里的中文与 $...$ 不乱码） ----",
        f"{i}plt.rcParams[\"font.family\"] = \"sans-serif\"",
        f"{i}plt.rcParams[\"font.sans-serif\"] = [{joined}, \"DejaVu Sans\"]",
        f"{i}plt.rcParams[\"axes.unicode_minus\"] = False  # 负号用 ASCII 连字符显示",
        f"{i}plt.rcParams[\"mathtext.fontset\"] = \"cm\"     # $...$ 公式使用 LaTeX 风格字体",
        f"{i}plt.rcParams[\"figure.dpi\"] = {int(spec.dpi)}",
        f"{i}plt.rcParams[\"savefig.bbox\"] = \"standard\"",
        f"{i}plt.rcParams[\"svg.fonttype\"] = \"path\"        # SVG 以路径嵌入文字，避免缺字体",
        f"{i}plt.rcParams[\"font.size\"] = {_py_literal(float(spec.font_size or 11.0))}",
        f"{i}plt.rcParams[\"axes.titlesize\"] = {_py_literal(float(spec.font_size or 11.0) + 1.5)}",
        f"{i}plt.rcParams[\"axes.labelsize\"] = {_py_literal(float(spec.font_size or 11.0))}",
        f"{i}plt.rcParams[\"xtick.labelsize\"] = {_py_literal(float(spec.font_size or 11.0) - 1.5)}",
        f"{i}plt.rcParams[\"ytick.labelsize\"] = {_py_literal(float(spec.font_size or 11.0) - 1.5)}",
        f"{i}plt.rcParams[\"legend.fontsize\"] = {_py_literal(float(spec.font_size or 11.0) - 1.5)}",
        "",
    ]
    return lines


def _theme_block(spec: PlotSpec, indent: str = "") -> list[str]:
    """生成主题应用代码。"""
    theme = str(spec.theme or "").strip()
    if theme in _DEFAULT_THEMES:
        return [f"{indent}# 主题：light（使用 matplotlib 默认样式，不额外加载样式表）", ""]
    return [
        f"{indent}# 主题：{theme}",
        f"{indent}plt.style.use({_py_str(theme)})",
        "",
    ]


def _figure_block(spec: PlotSpec, indent: str = "") -> list[str]:
    """生成画布与坐标轴创建代码。"""
    figsize = tuple(spec.figsize) if spec.figsize and len(tuple(spec.figsize)) == 2 else (8.0, 5.0)
    dpi = int(spec.dpi) if spec.dpi and spec.dpi > 0 else 110
    subplot_kw = ", subplot_kw={\"projection\": \"3d\"}" if spec.is_3d() else ""
    lines = [
        f"{indent}# ---- 画布与坐标轴 ----",
        f"{indent}fig, ax = plt.subplots(figsize=({float(figsize[0])}, {float(figsize[1])}), "
        f"dpi={dpi}{subplot_kw})",
        f"{indent}# 留白与 TableDrawer 内部渲染保持一致",
        f"{indent}try:",
        f"{indent}    ax.margins(x=0.02)",
        f"{indent}except Exception:",
        f"{indent}    pass",
        f"{indent}_mappables = []  # 需要加色条的图元",
    ]
    colors = [str(c) for c in (spec.colorcycle or []) if str(c).strip()]
    if colors:
        joined = ", ".join(_py_str(c) for c in colors)
        lines.append(f"{indent}ax.set_prop_cycle(color=[{joined}])")
    lines.append("")
    return lines


def _axis_lines(spec: PlotSpec, options: CodeGenOptions, indent: str = "") -> list[str]:
    """生成坐标轴设置区（标题、轴标签、刻度、图例、色条、布局）。"""
    comments = bool(options.include_comments)
    lines: list[str] = []

    def add(text: str, extra: str = "") -> None:
        lines.append(f"{indent}{extra}{text}")

    if comments:
        add("# ---- 3D 坐标轴设置 ----" if spec.is_3d() else "# ---- 2D 坐标轴设置 ----")
    if spec.title:
        add(f"ax.set_title({_py_str(spec.title)})")
    if spec.xlabel:
        add(f"ax.set_xlabel({_py_str(spec.xlabel)})")
    if spec.ylabel:
        add(f"ax.set_ylabel({_py_str(spec.ylabel)})")
    if spec.is_3d() and spec.zlabel:
        add(f"ax.set_zlabel({_py_str(spec.zlabel)})")

    for axis, scale in (("x", spec.xscale), ("y", spec.yscale), ("z", spec.zscale)):
        value = str(scale or "linear").strip() or "linear"
        if value == "linear" or (axis == "z" and not spec.is_3d()):
            continue
        add(f"ax.set_{axis}scale({_py_str(value)})")

    textlike = any(str(s.kind).lower() in _TEXTLIKE_KINDS for s in spec.series)
    if spec.xlim and len(tuple(spec.xlim)) == 2:
        add(f"ax.set_xlim({_py_literal(tuple(spec.xlim)[0])}, {_py_literal(tuple(spec.xlim)[1])})")
    if spec.ylim and len(tuple(spec.ylim)) == 2:
        if not (textlike and not spec.is_3d()):
            add(f"ax.set_ylim({_py_literal(tuple(spec.ylim)[0])}, {_py_literal(tuple(spec.ylim)[1])})")
    if spec.is_3d() and spec.zlim and len(tuple(spec.zlim)) == 2:
        add(f"ax.set_zlim({_py_literal(tuple(spec.zlim)[0])}, {_py_literal(tuple(spec.zlim)[1])})")

    if spec.grid:
        add("ax.grid(True, alpha=0.35)")
    if spec.equal_aspect:
        add("ax.set_aspect(\"equal\", adjustable=\"datalim\")")
    if spec.is_3d():
        add(
            f"ax.view_init(elev={_py_literal(float(spec.view_elev))}, "
            f"azim={_py_literal(float(spec.view_azim))})"
        )
    if spec.legend:
        add("_handles, _labels = ax.get_legend_handles_labels()")
        add("if any(not str(_l).startswith(\"_\") for _l in _labels):")
        add(
            f"ax.legend(loc={_py_str(spec.legend_loc or 'best')}, "
            f"frameon={bool(spec.legend_frame)})",
            extra="    ",
        )
    elif comments:
        add("# 图例已关闭（spec.legend = False）")
    if spec.colorbar and any(str(s.kind).lower() in _COLORBAR_KINDS for s in spec.series):
        add("for _mappable in _mappables:")
        add("fig.colorbar(_mappable, ax=ax, shrink=0.6)", extra="    ")
    if spec.tight_layout:
        add("fig.tight_layout()")
    lines.append("")
    return lines


def _function_wrapper_block() -> list[str]:
    """生成 ``def make_figure() -> Figure`` 形式的脚本主体。"""
    return [
        "def make_figure() -> \"matplotlib.figure.Figure\":",
        '    """按导出的配置创建并返回 matplotlib Figure（不保存、不显示）。"""',
        "",
    ]


def _main_block(spec: PlotSpec, options: CodeGenOptions, indent: str = "    ") -> list[str]:
    """生成 ``__main__`` 区（保存示例 + 显示）。"""
    lines: list[str] = []
    if options.include_sample_save:
        filename = str(options.output_filename or "figure.svg")
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "svg"
        lines += [
            f"{indent}# 保存到当前目录（如需其它格式，改扩展名即可：png / pdf / svg）",
            f"{indent}fig.savefig({_py_str(filename)}, format={_py_str(extension)}, "
            f"dpi=fig.dpi, bbox_inches=\"tight\", transparent={bool(spec.transparent)})",
        ]
    lines.append(f"{indent}if matplotlib.get_backend().lower() != \"agg\":")
    lines.append(f"{indent}    plt.show()  # Agg 后端下没有窗口，跳过显示")
    return lines


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def generate_script(
    spec: PlotSpec,
    datasets: Mapping[str, pd.DataFrame],
    options: CodeGenOptions | None = None,
) -> str:
    """生成完整的、可直接 ``python xxx.py`` 运行的 matplotlib 脚本源码。

    参数
    ----
    spec
        绘图规格（:class:`core.spec.PlotSpec`）。
    datasets
        ``{数据集名: DataFrame}``；只有被 ``spec`` 引用到的数据集会被写进源码。
    options
        :class:`CodeGenOptions`，``None`` 时使用默认值。

    返回
    ----
    str
        UTF-8 源码文本（末尾带换行）。
    """
    if not isinstance(spec, PlotSpec):
        raise TypeError("spec 必须是 core.spec.PlotSpec")
    opts = options or CodeGenOptions()
    if not isinstance(opts, CodeGenOptions):
        raise TypeError("options 必须是 CodeGenOptions")

    reset_var_registry()
    table: dict[str, pd.DataFrame] = {
        str(key): value for key, value in (datasets or {}).items() if isinstance(value, pd.DataFrame)
    }

    # 1) 规划数据集变量名（按序列出现顺序，保证可复现）
    vars_map: dict[str, str] = {}
    ordered: list[tuple[str, pd.DataFrame]] = []
    for series in spec.series:
        key = str(series.dataset or "")
        if not key:
            continue
        if key in table and key not in vars_map:
            vars_map[key] = _var_for(key, vars_map)
            ordered.append((key, table[key]))

    lines: list[str] = []

    # 2) 头部 + import + 主题 + 字体
    header = _header(spec, opts, dict(ordered)) if opts.include_comments else [
        "# -*- coding: utf-8 -*-",
        '"""由 TableDrawer 导出的 matplotlib 源码。"""',
        "",
    ]
    lines += header
    lines += _imports(spec, opts)
    lines += _theme_block(spec)
    lines += _font_block(spec)

    wrapper = bool(opts.function_wrapper)
    body_indent = "    " if wrapper else ""

    # 3) 数据区
    data_lines: list[str] = _figure_block(spec, body_indent)
    if ordered:
        if opts.include_comments:
            data_lines.append(f"{body_indent}# " + "=" * 68)
            data_lines.append(
                f"{body_indent}# 数据区"
                + ("（已内联，无需外部文件）" if opts.embed_data else "（从 CSV 读取）")
            )
            data_lines.append(f"{body_indent}# " + "=" * 68)
        for name, frame in ordered:
            variable = vars_map[name]
            if opts.embed_data:
                data_lines += _dataset_block(
                    name, variable, frame, opts, opts.include_comments, body_indent
                )
            else:
                data_lines += _csv_block(
                    name, variable, frame, opts, opts.include_comments, body_indent
                )
    elif opts.include_comments:
        data_lines.append(f"{body_indent}# 数据区：本图没有引用任何数据集")

    # 4) 绘图区
    draw_lines: list[str] = []
    if opts.include_comments:
        draw_lines.append(f"{body_indent}# " + "=" * 68)
        draw_lines.append(f"{body_indent}# 绘图区")
        draw_lines.append(f"{body_indent}# " + "=" * 68)
    if not spec.series and opts.include_comments:
        draw_lines.append(f"{body_indent}# 序列列表为空")
    for index, series in enumerate(spec.series, start=1):
        key = str(series.dataset or "")
        frame = table.get(key)
        variable = vars_map.get(key)
        if frame is None or variable is None:
            if opts.include_comments:
                reason = f"引用的数据集「{key}」不存在" if key else "没有指定数据集"
                draw_lines.append(f"{body_indent}# 序列 {index}：已跳过（{reason}）")
                draw_lines.append("")
            continue
        if opts.include_comments:
            draw_lines.append(body_indent + _series_comment(index, series, frame))
        drawing_spec = dataclasses.replace(spec)
        draw_lines += _series_lines(
            series, variable, frame, opts, drawing_spec, index,
            opts.include_comments, "ax", body_indent,
        )
        if opts.include_comments:
            draw_lines.append("")

    # 5) 坐标轴设置 + 保存
    axis_lines = _axis_lines(spec, opts, body_indent)

    if wrapper:
        lines += _function_wrapper_block()
        lines += data_lines
        lines += draw_lines
        lines += axis_lines
        lines += ["    return fig", "", ""]
        if opts.include_comments:
            lines.append("# ---- 直接运行本文件时保存并显示图片 ----")
        lines.append("if __name__ == \"__main__\":")
        lines.append("    fig = make_figure()")
        lines += _main_block(spec, opts)
    else:
        lines += data_lines
        lines += draw_lines
        lines += axis_lines
        if opts.include_comments:
            lines.append("# " + "=" * 68)
            lines.append("# 保存与显示")
            lines.append("# " + "=" * 68)
        lines.append("if __name__ == \"__main__\":")
        lines += _main_block(spec, opts)

    lines.append("")
    return "\n".join(lines)
