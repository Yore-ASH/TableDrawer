"""文本与公式度量 —— 全部以「点」(1/72 英寸) 为单位。

设计要点
--------
* 数学部分复用 matplotlib 的 ``mathtext`` 解析器做**度量**，这样排版尺寸与
  最终 ``Axes.text`` 的渲染尺寸严格一致（两者走同一个解析器）。
* ``MathTextParser.parse(s, dpi=72, prop)`` 在 dpi=72 下返回的宽/高/深即为点值；
  ``TextToPath.get_text_width_height_descent`` 内部同样以 dpi=72 计算，单位一致。
* ``measure_math`` 返回 ``(w, h, d)``：``h`` 为基线以上高度，``d`` 为基线以下深度。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import matplotlib
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath, TextToPath

__all__ = [
    "MathParseError",
    "normalize_math",
    "math_fontset",
    "measure_math",
    "measure_text",
    "measure_space",
    "text_strut",
    "glyph_extents",
    "family_key",
    "clear_cache",
]

_MATH_PARSER = mathtext.MathTextParser("path")
_TTP = TextToPath()

#: 空宽度的兜底比例（相对字号）
_FALLBACK_SPACE = 0.30


class MathParseError(ValueError):
    """mathtext 无法解析给定公式时抛出（调用方应降级处理）。"""


# --------------------------------------------------------------------------- #
# 缓存清理
# --------------------------------------------------------------------------- #
def clear_cache() -> None:
    """清空本模块以及 mathtext 内部缓存（切换数学字体后必须调用）。"""
    for func in (
        _measure_math_cached,
        _measure_text_cached,
        _measure_text_runs,
        _space_width,
        _glyph_cached,
        _strut_cached,
        _font_for_char,
    ):
        clear = getattr(func, "cache_clear", None)
        if clear is not None:
            clear()
    parser_cls = getattr(mathtext, "MathTextParser", None)
    for holder in (parser_cls, mathtext):
        clear = getattr(getattr(holder, "_parse_cached", None), "cache_clear", None)
        if clear is not None:
            try:
                clear()
            except Exception:  # pragma: no cover - 版本差异兜底
                pass


# --------------------------------------------------------------------------- #
# 数学
# --------------------------------------------------------------------------- #
def math_fontset() -> str:
    """当前配置的 mathtext 字体集名称。"""
    return str(matplotlib.rcParams.get("mathtext.fontset", "dejavusans"))


def normalize_math(body: str) -> str:
    """把公式体包装成 ``$...$``（mathtext 要求美元符定界）。"""
    body = (body or "").strip()
    if not body:
        return "$$"
    if len(body) >= 2 and body.startswith("$") and body.endswith("$"):
        return body
    return f"${body}$"


@lru_cache(maxsize=32768)
def _measure_math_cached(text: str, fontsize: float, fontset: str) -> tuple[float, float, float]:
    prop = FontProperties(size=fontsize)
    width, height, depth, _glyphs, _rects = _MATH_PARSER.parse(text, dpi=72, prop=prop)
    return float(width), float(height), float(depth)


def measure_math(body: str, fontsize: float) -> tuple[float, float, float]:
    """度量一段数学公式体（不含 ``$``），返回 ``(w, h, d)`` 点值。

    解析失败抛出 :class:`MathParseError`。
    """
    body = (body or "").strip()
    if not body:
        return 0.0, 0.0, 0.0
    text = normalize_math(body)
    try:
        return _measure_math_cached(text, float(fontsize), math_fontset())
    except MathParseError:
        raise
    except Exception as exc:  # mathtext 抛出的 ValueError / ParseFatalException
        raise MathParseError(f"{exc}") from exc


# --------------------------------------------------------------------------- #
# 普通文本
# --------------------------------------------------------------------------- #
def family_key(family: Any) -> tuple[str, ...]:
    """把字体族规范化为可哈希的元组。

    支持传入 **字体族列表**（如 ``["Microsoft YaHei", "DejaVu Sans"]``）：
    matplotlib 3.6+ 会自动做字体回退，这样中文字体缺失的
    ``⁻ ᵢ ᵀ ⱼ`` 等上下标字符就能回退到 DejaVu Sans 正常显示，
    而不是渲染成豆腐块。
    """
    if family is None:
        return ("DejaVu Sans",)
    if isinstance(family, str):
        return (family,) if family.strip() else ("DejaVu Sans",)
    items = tuple(str(item) for item in family if str(item).strip())
    return items or ("DejaVu Sans",)


@lru_cache(maxsize=2048)
def _font_path(family: str) -> str:
    from matplotlib.font_manager import findfont

    return findfont(FontProperties(family=[family]), fallback_to_default=False)


@lru_cache(maxsize=256)
def _ft2font(path: str):
    from matplotlib.ft2font import FT2Font

    return FT2Font(path)


@lru_cache(maxsize=8192)
def _font_for_char(char: str, chain: tuple[str, ...]) -> str:
    """在字体回退链里找出第一个真正含有该字符的字体族名。

    这一步是**度量正确性的关键**：matplotlib 的 Agg 渲染器会做逐字符字体回退
    （CJK 落到宋体、``⁻ ᵢ`` 落到 DejaVu Sans），但 ``TextToPath``/``findfont``
    只会返回链里的**第一个**字体。若不做逐字符解析，中文的宽度会被算错，
    导致排版里中文与公式互相重叠。
    """
    code = ord(char)
    for name in chain:
        try:
            font = _ft2font(_font_path(name))
        except Exception:
            continue
        try:
            if code in font.get_charmap():
                return name
        except Exception:
            continue
    return chain[0] if chain else "DejaVu Sans"


def _runs_by_font(text: str, chain: tuple[str, ...]) -> list[tuple[str, tuple[str, ...]]]:
    """按「实际使用的字体」把文本切成若干连续片段。"""
    runs: list[tuple[list[str], str]] = []
    for char in text:
        name = _font_for_char(char, chain)
        if runs and runs[-1][1] == name:
            runs[-1][0].append(char)
        else:
            runs.append(([char], name))
    return [("".join(chars), (name,)) for chars, name in runs]


@lru_cache(maxsize=32768)
def _measure_text_cached(
    text: str, fontsize: float, family: tuple[str, ...], weight: str, style: str
) -> tuple[float, float, float]:
    prop = FontProperties(family=list(family), size=fontsize, weight=weight, style=style)
    width, height, depth = _TTP.get_text_width_height_descent(text, prop, ismath=False)
    return float(width), float(height), float(depth)


@lru_cache(maxsize=32768)
def _measure_text_runs(
    text: str, fontsize: float, chain: tuple[str, ...], weight: str, style: str
) -> tuple[float, float, float]:
    """逐字体片段度量后求和（宽度相加，高度/深度取最大）。"""
    total = 0.0
    height = 0.0
    depth = 0.0
    for run, run_family in _runs_by_font(text, chain):
        width, run_height, run_depth = _measure_text_cached(
            run, fontsize, run_family, weight, style
        )
        total += width
        height = max(height, run_height)
        depth = max(depth, run_depth)
    return total, height, depth


def measure_text(
    text: str,
    fontsize: float,
    family: Any,
    weight: str = "normal",
    style: str = "normal",
) -> tuple[float, float, float]:
    """度量普通文本（支持中文与逐字符字体回退），返回 ``(w, h, d)`` 点值。"""
    if not text:
        return 0.0, 0.0, 0.0
    chain = family_key(family)
    try:
        return _measure_text_runs(text, float(fontsize), chain, weight, style)
    except Exception:
        # 字体缺失等异常：按字符数粗略估算，保证不中断排版
        return len(text) * fontsize * 0.6, fontsize * 0.72, fontsize * 0.2


@lru_cache(maxsize=512)
def _space_width(
    fontsize: float, family: tuple[str, ...], weight: str, style: str
) -> float:
    one = _measure_text_cached("a", fontsize, family, weight, style)[0]
    two = _measure_text_cached("a a", fontsize, family, weight, style)[0]
    width = two - 2.0 * one
    if width <= 0.01:
        width = fontsize * _FALLBACK_SPACE
    return float(width)


def measure_space(
    fontsize: float, family: Any, weight: str = "normal", style: str = "normal"
) -> float:
    """空格宽度（点）。字体不提供空格度量时回退为 ``0.30em``。"""
    try:
        return _space_width(float(fontsize), family_key(family), weight, style)
    except Exception:
        return float(fontsize) * _FALLBACK_SPACE


@lru_cache(maxsize=1024)
def _strut_cached(
    fontsize: float, family: tuple[str, ...], weight: str, style: str
) -> tuple[float, float]:
    _w, height, depth = _measure_text_cached("Ag", fontsize, family, weight, style)
    if height <= 0:
        height = fontsize * 0.78
    if depth <= 0:
        depth = fontsize * 0.22
    return float(height), float(depth)


def text_strut(
    fontsize: float, family: Any, weight: str = "normal", style: str = "normal"
) -> tuple[float, float]:
    """一行文本的 ``(基线以上高度, 基线以下深度)``，用于行距计算。"""
    try:
        return _strut_cached(float(fontsize), family_key(family), weight, style)
    except Exception:
        return float(fontsize) * 0.78, float(fontsize) * 0.22


# --------------------------------------------------------------------------- #
# 字形轮廓（用于可伸缩定界符）
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1024)
def _glyph_cached(
    char: str, fontsize: float, family: tuple[str, ...]
) -> tuple[tuple[tuple[float, float], ...], tuple[int, ...], float, float, float, float]:
    prop = FontProperties(family=list(family), size=fontsize)
    path = TextPath((0.0, 0.0), char, prop=prop, usetex=False)
    verts = tuple((float(x), float(y)) for x, y in path.vertices)
    codes = tuple(int(c) for c in (path.codes if path.codes is not None else []))
    bbox = path.get_extents()
    return verts, codes, float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)


def glyph_extents(
    char: str, fontsize: float, family: Any
) -> tuple[list[tuple[float, float]], list[int] | None, float, float, float, float]:
    """返回字形的轮廓与包围盒 ``(verts, codes, x0, y0, x1, y1)``。

    坐标系为 *数学* 习惯：原点在基线左端，**y 轴向上**。
    调用方负责施加缩放/翻转变换。
    """
    verts, codes, x0, y0, x1, y1 = _glyph_cached(char, float(fontsize), family_key(family))
    return list(verts), (list(codes) if codes else None), x0, y0, x1, y1
