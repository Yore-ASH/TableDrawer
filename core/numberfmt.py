"""数值格式化工具：把 Python / NumPy 数值转换为可读文本与 LaTeX 数学片段。

本模块是 ``matrixops`` 与 LaTeX 渲染层之间唯一的数值表示契约：

* ``format_number``  -> 纯文本（用于剪贴板、报告、日志）
* ``to_latex``       -> 不带 ``$`` 的 LaTeX 数学体（交给 :mod:`core.latex` 渲染）
* ``matrix_to_latex``-> ``\\begin{pmatrix} ... \\end{pmatrix}`` 形式的矩阵
* ``matrix_to_text`` -> 等宽对齐的纯文本矩阵

所有函数都接受 ``np.ndarray`` / 嵌套列表 / 标量，并且不会修改输入。
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_PRECISION",
    "ZERO_TOL",
    "snap_zero",
    "format_number",
    "format_value",
    "to_latex",
    "vector_to_latex",
    "matrix_to_latex",
    "matrix_to_text",
    "MAX_FRACTION_DENOMINATOR",
]

DEFAULT_PRECISION = 6
ZERO_TOL = 1e-12
MAX_FRACTION_DENOMINATOR = 10_000

# 超过这些尺寸的矩阵在 LaTeX 中会被截断显示
_MAX_LATEX_ROWS = 24
_MAX_LATEX_COLS = 14


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _as_complex(x: Any) -> complex:
    """把任意标量安全转换为 ``complex``。"""
    if isinstance(x, complex):
        return x
    if isinstance(x, np.complexfloating):
        return complex(x)
    return complex(float(x), 0.0)


def snap_zero(x: Any, tol: float = ZERO_TOL, scale: float = 1.0) -> float | complex:
    """把浮点噪声（例如 ``1e-17``）吸附为精确的 ``0.0``。

    ``scale`` 用于按矩阵量级自适应：``abs(x) <= tol * max(1, abs(scale))`` 时返回 0。
    传复数时**实部与虚部分别处理**，返回复数（早期版本用 ``float(x)`` 会静默丢掉虚部）。
    """
    if isinstance(x, (complex, np.complexfloating)):
        z = complex(x)
        return complex(_snap_float(z.real, tol, scale), _snap_float(z.imag, tol, scale))
    return _snap_float(x, tol, scale)


def _snap_float(value: Any, tol: float, scale: float) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        return value_f
    if abs(value_f) <= tol * max(1.0, abs(float(scale))):
        return 0.0
    return value_f


def _trim_exponent(text: str) -> str:
    """``1e-05`` -> ``1e-5``；``1.5E+10`` -> ``1.5e10``。"""
    if "e" not in text and "E" not in text:
        return text
    mantissa, _, exponent = text.replace("E", "e").partition("e")
    sign = ""
    if exponent[:1] in "+-":
        sign, exponent = exponent[0], exponent[1:]
    exponent = exponent.lstrip("0") or "0"
    if exponent == "0":
        return mantissa
    return f"{mantissa}e{'-' if sign == '-' else ''}{exponent}"


def _trim_float(text: str) -> str:
    """去掉小数点后多余的 0：``1.500000`` -> ``1.5``，``2.000000`` -> ``2``。"""
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".") or "0"


def _split_sci(value: float, precision: int) -> tuple[str, int]:
    """返回 ``(尾数文本, 指数)``，尾数已按精度裁剪且不带指数记号。"""
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    mantissa = float(f"{mantissa:.{precision}g}")
    # 四舍五入后可能进位到 10.0（例如 9.9999999 -> 10）
    if abs(mantissa) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    return _trim_float(f"{mantissa:.{precision}g}"), exponent


# --------------------------------------------------------------------------- #
# 纯文本
# --------------------------------------------------------------------------- #
def format_number(
    x: Any,
    precision: int = DEFAULT_PRECISION,
    tol: float = ZERO_TOL,
    *,
    sci_lo: float = 1e-4,
    sci_hi: float = 1e7,
) -> str:
    """把一个标量格式化为简洁的纯文本。

    >>> format_number(0.1 + 0.2)
    '0.3'
    >>> format_number(3.0)
    '3'
    >>> format_number(float('nan'))
    'nan'
    """
    if isinstance(x, (bool, np.bool_)):
        return "True" if bool(x) else "False"
    if isinstance(x, (int, np.integer)):
        return str(int(x))

    z = _as_complex(x)
    if z.imag != 0.0:
        # 注意：这里必须自己拼装，不能再调用 format_value —— 那会绕回本函数造成无限递归
        return _text_complex(z, precision, tol)

    value = snap_zero(z.real, tol)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if value == 0.0:
        return "0"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))

    if sci_lo <= abs(value) < sci_hi:
        return _trim_float(f"{value:.{precision}g}")
    mantissa, exponent = _split_sci(value, precision)
    return f"{mantissa}e{exponent}"


def _text_complex(z: complex, precision: int, tol: float) -> str:
    """复数的纯文本形式：``3 + 4i`` / ``-i`` / ``2.5 - 1.5i``。"""
    real = float(snap_zero(z.real, tol))
    imag = float(snap_zero(z.imag, tol))
    imag_body = format_number(abs(imag), precision=precision, tol=tol)
    if abs(imag) == 1.0:
        imag_body = ""
    if real == 0.0:
        return f"{'-' if imag < 0 else ''}{imag_body}i"
    sign = "+" if imag > 0 else "-"
    return f"{format_number(real, precision=precision, tol=tol)} {sign} {imag_body}i"


def format_value(x: Any, precision: int = DEFAULT_PRECISION, tol: float = ZERO_TOL) -> str:
    """通用入口：标量 -> 文本，向量/矩阵 -> ``matrix_to_text``。"""
    arr = np.asarray(x) if not isinstance(x, (str, bytes)) else None
    if arr is None:
        return str(x)
    if arr.dtype.kind in "OUS":
        if arr.ndim == 0:
            return str(arr.item())
        return matrix_to_text(arr, precision=precision, tol=tol)
    if arr.ndim == 0:
        return format_number(arr.item(), precision=precision, tol=tol)
    if arr.ndim == 1:
        return "[" + ", ".join(format_number(v, precision=precision, tol=tol) for v in arr) + "]"
    return matrix_to_text(arr, precision=precision, tol=tol)


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def _latex_complex(z: complex, precision: int, fractions: bool, tol: float) -> str:
    real = snap_zero(z.real, tol)
    imag = snap_zero(z.imag, tol)
    if imag == 0.0:
        return to_latex(real, precision, fractions, tol=tol)
    imag_body = to_latex(abs(imag), precision, fractions, tol=tol)
    if abs(imag) == 1.0:
        imag_body = ""
    sign = "+" if imag > 0 else "-"
    if real == 0.0:
        return f"{'-' if imag < 0 else ''}{imag_body}i"
    return f"{to_latex(real, precision, fractions, tol=tol)} {sign} {imag_body}i"


def to_latex(
    x: Any,
    precision: int = DEFAULT_PRECISION,
    fractions: bool = False,
    tol: float = ZERO_TOL,
    *,
    sci_lo: float = 1e-4,
    sci_hi: float = 1e7,
) -> str:
    """把一个标量转换为 **不带 ``$``** 的 LaTeX 数学体。

    ``fractions=True`` 时会把接近的十进制数还原为分数（``0.333333`` -> ``\\frac{1}{3}``）。
    """
    if isinstance(x, (bool, np.bool_)):
        return r"\text{True}" if bool(x) else r"\text{False}"
    if isinstance(x, (int, np.integer)):
        return str(int(x))

    z = _as_complex(x)
    if z.imag != 0.0:
        return _latex_complex(z, precision, fractions, tol)

    value = snap_zero(z.real, tol)
    if math.isnan(value):
        return r"\text{NaN}"
    if math.isinf(value):
        return r"\infty" if value > 0 else r"-\infty"
    if value == 0.0:
        return "0"
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))

    if fractions:
        frac = Fraction(value).limit_denominator(MAX_FRACTION_DENOMINATOR)
        if abs(float(frac) - value) <= max(1e-12, abs(value) * 1e-10):
            if frac.denominator == 1:
                return str(frac.numerator)
            sign = "-" if frac.numerator < 0 else ""
            return f"{sign}\\frac{{{abs(frac.numerator)}}}{{{frac.denominator}}}"

    if sci_lo <= abs(value) < sci_hi:
        return _trim_float(f"{value:.{precision}g}")

    mantissa, exponent = _split_sci(value, precision)
    return f"{mantissa}\\times 10^{{{exponent}}}"


def vector_to_latex(
    v: Any,
    precision: int = DEFAULT_PRECISION,
    fractions: bool = False,
    tol: float = ZERO_TOL,
) -> str:
    """列向量：``\\begin{pmatrix} a \\\\ b \\end{pmatrix}``。"""
    arr = np.asarray(v)
    if arr.ndim == 0:
        return to_latex(arr.item(), precision, fractions, tol)
    flat = arr.reshape(-1)
    cells = [to_latex(val, precision, fractions, tol) for val in flat]
    body = r" \\ ".join(cells)
    return r"\begin{pmatrix}" + body + r"\end{pmatrix}"


def matrix_to_latex(
    m: Any,
    precision: int = DEFAULT_PRECISION,
    fractions: bool = False,
    tol: float = ZERO_TOL,
    *,
    env: str = "pmatrix",
    name: str | None = None,
    max_rows: int = _MAX_LATEX_ROWS,
    max_cols: int = _MAX_LATEX_COLS,
) -> str:
    """把矩阵转换为 LaTeX 矩阵环境字符串。

    ``name`` 非空时返回 ``name = \\begin{pmatrix}...``；超出行列上限时用
    ``\\cdots`` / ``\\vdots`` / ``\\ddots`` 省略中间部分。
    """
    arr = np.asarray(m)
    if arr.ndim == 0:
        body = to_latex(arr.item(), precision, fractions, tol)
        return f"{name} = {body}" if name else body
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    scale = _array_scale(arr)

    def cell(v: Any) -> str:
        if isinstance(v, (str, np.str_)):
            return r"\text{" + str(v) + "}"
        if v is None:
            return r"\cdot"
        try:
            return to_latex(snap_zero(v, tol, scale), precision, fractions, tol)
        except (TypeError, ValueError):
            return r"\text{" + str(v) + "}"

    n_rows, n_cols = arr.shape
    row_idx = list(range(n_rows))
    col_idx = list(range(n_cols))
    truncated_rows = n_rows > max_rows
    truncated_cols = n_cols > max_cols
    if truncated_rows:
        head, tail = max_rows // 2, max_rows - max_rows // 2 - 1
        row_idx = list(range(head)) + [-1] + list(range(n_rows - tail, n_rows))
    if truncated_cols:
        head, tail = max_cols // 2, max_cols - max_cols // 2 - 1
        col_idx = list(range(head)) + [-1] + list(range(n_cols - tail, n_cols))

    rows: list[str] = []
    for i in row_idx:
        cells: list[str] = []
        for j in col_idx:
            if i == -1 and j == -1:
                cells.append(r"\ddots")
            elif i == -1:
                cells.append(r"\vdots")
            elif j == -1:
                cells.append(r"\cdots")
            else:
                cells.append(cell(arr[i, j]))
        rows.append(" & ".join(cells))

    body = r" \\ ".join(rows)
    env_name = env if env in {"matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix"} else "pmatrix"
    result = rf"\begin{{{env_name}}}" + body + rf"\end{{{env_name}}}"
    return f"{name} = {result}" if name else result


def _array_scale(arr: np.ndarray) -> float:
    """矩阵的数值量级，用于零吸附；含非数值元素时返回 1.0。"""
    if arr.size == 0:
        return 1.0
    try:
        numeric = np.asarray(arr, dtype=complex)
    except (TypeError, ValueError):
        return 1.0
    magnitude = np.abs(numeric)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size == 0:
        return 1.0
    return float(np.max(finite))


def _cell_text(value: Any, precision: int, tol: float, scale: float) -> str:
    """单个矩阵单元格的纯文本；非数值内容原样字符串化。"""
    if isinstance(value, (str, np.str_)):
        return str(value)
    if value is None:
        return ""
    try:
        return format_number(snap_zero(value, tol, scale), precision, tol)
    except (TypeError, ValueError):
        return str(value)


def matrix_to_text(
    m: Any,
    precision: int = DEFAULT_PRECISION,
    tol: float = ZERO_TOL,
    *,
    name: str | None = None,
) -> str:
    """等宽对齐的纯文本矩阵，便于复制到终端或日志。"""
    arr = np.asarray(m)
    if arr.ndim == 0:
        body = format_number(arr.item(), precision, tol)
        return f"{name} = {body}" if name else body
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    scale = _array_scale(arr)
    texts = [[_cell_text(v, precision, tol, scale) for v in row] for row in arr]
    if not texts or not texts[0]:
        return f"{name} = []" if name else "[]"
    widths = [max(len(texts[i][j]) for i in range(len(texts))) for j in range(len(texts[0]))]
    lines = ["[ " + "  ".join(t.rjust(w) for t, w in zip(row, widths)) + " ]" for row in texts]
    body = "\n".join(lines)
    return f"{name} =\n{body}" if name else body
