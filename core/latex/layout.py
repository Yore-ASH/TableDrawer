"""把 AST 排版为带绝对坐标的绘制指令。

坐标系
------
排版在「点」空间中进行，每个 :class:`Box` 记录::

    w : 宽度
    h : 基线**以上**高度
    d : 基线**以下**深度

绘制指令的坐标以 ``(0, 0) = (左端, 基线)`` 为原点，**y 轴向下**（与屏幕一致），
这样文档层只需顺序下推 y 即可，无需翻转。

所有几何量对字号都是线性的，因此「先按字号 fs 排一遍，发现太宽就按比例换
更小的字号重排」这种收缩策略一次即可收敛。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from matplotlib.transforms import Affine2D

from .measure import (
    MathParseError,
    glyph_extents,
    measure_math,
    measure_space,
    measure_text,
    normalize_math,
    text_strut,
)
from .nodes import (
    DELIM_CHARS,
    ENV_ALIGN,
    ENV_DELIMITERS,
    ENV_SCALE,
    Fence,
    Frac,
    Group,
    LineBreak,
    Matrix,
    Node,
    Plain,
    Run,
    Script,
    Space,
    Sqrt,
    as_group,
)
from .parser import atomize, latex_to_unicode, literal_node

__all__ = [
    "TextDraw",
    "PathDraw",
    "Box",
    "LayoutContext",
    "layout",
    "translate",
    "delimiter_width",
]


# --------------------------------------------------------------------------- #
# 绘制指令
# --------------------------------------------------------------------------- #
@dataclass
class TextDraw:
    """一段文本（``math=True`` 时 ``text`` 含 ``$...$``）。"""

    x: float
    y: float
    text: str
    fontsize: float
    family: str
    math: bool = False
    color: str = "#000000"
    weight: str = "normal"
    style: str = "normal"
    math_fontset: str = "cm"


@dataclass
class PathDraw:
    """一条路径；``fill=True`` 时填充（用于字形轮廓与分数线）。"""

    verts: list[tuple[float, float]]
    codes: list[int] | None = None
    width: float = 0.8
    color: str = "#000000"
    fill: bool = False


@dataclass
class Box:
    """排版结果：尺寸 + 绘制指令。"""

    w: float = 0.0
    h: float = 0.0
    d: float = 0.0
    draws: list[Any] = field(default_factory=list)

    @property
    def height(self) -> float:
        return self.h + self.d


def translate(draw: Any, dx: float, dy: float) -> None:
    """就地平移一条绘制指令。"""
    if isinstance(draw, TextDraw):
        draw.x += dx
        draw.y += dy
    elif isinstance(draw, PathDraw):
        draw.verts = [(x + dx, y + dy) for x, y in draw.verts]


def translate_box(box: Box, dx: float, dy: float) -> None:
    for draw in box.draws:
        translate(draw, dx, dy)


def hstack(boxes: Iterable[Box], gaps: Iterable[float] | None = None) -> Box:
    """按基线对齐水平拼接若干盒子。"""
    items = list(boxes)
    if not items:
        return Box()
    gap_list = list(gaps) if gaps is not None else []
    x = 0.0
    draws: list[Any] = []
    h = 0.0
    d = 0.0
    for index, box in enumerate(items):
        translate_box(box, x, 0.0)
        draws.extend(box.draws)
        x += box.w
        if index < len(gap_list):
            x += gap_list[index]
        h = max(h, box.h)
        d = max(d, box.d)
    return Box(x, h, d, draws)


# --------------------------------------------------------------------------- #
# 排版上下文
# --------------------------------------------------------------------------- #
@dataclass
class LayoutContext:
    """排版所需的字体与颜色设置。"""

    family: str = "Microsoft YaHei"
    math_family: str = "STIXGeneral"
    delim_family: str = "STIXGeneral"
    color: str = "#111111"
    math_fontset: str = "cm"


# --------------------------------------------------------------------------- #
# 定界符
# --------------------------------------------------------------------------- #
def delimiter_width(token: str, fs: float, ctx: LayoutContext) -> float:
    """定界符在给定字号下的自然宽度（点）。"""
    char = DELIM_CHARS.get(token)
    if char is None:
        return 0.0
    _verts, _codes, x0, _y0, x1, y1 = glyph_extents(char, fs, ctx.delim_family)
    if x1 <= x0:
        return 0.0
    return x1 - x0


def _delimiter_draws(
    token: str, x: float, top: float, bottom: float, fs: float, ctx: LayoutContext
) -> list[PathDraw]:
    """生成定界符轮廓；纵向拉伸以匹配内容高度，横向保持字形比例。"""
    char = DELIM_CHARS.get(token)
    if char is None:
        return []
    verts, codes, x0, y0, x1, y1 = glyph_extents(char, fs, ctx.delim_family)
    if not verts or x1 <= x0 or y1 <= y0:
        return []

    natural_h = y1 - y0
    want = max(bottom - top, 0.0)
    sy = want / natural_h if natural_h > 0 else 1.0
    if sy < 1.0:                      # 不缩小：内容矮时保留字形原大小
        sy = 1.0
    drawn_h = natural_h * sy
    y_top = top + (want - drawn_h) / 2.0

    transform = Affine2D().translate(-x0, -y1).scale(1.0, -sy).translate(x, y_top)
    new_verts = [(float(px), float(py)) for px, py in transform.transform(verts)]
    return [
        PathDraw(
            verts=new_verts,
            codes=list(codes) if codes else None,
            fill=True,
            color=ctx.color,
        )
    ]


def _rect(x0: float, y0: float, x1: float, y1: float, color: str) -> PathDraw:
    """填充矩形（分数线、根号上横线、表格竖线）。"""
    return PathDraw(
        verts=[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
        codes=None,
        fill=True,
        color=color,
    )


# --------------------------------------------------------------------------- #
# 各节点的排版
# --------------------------------------------------------------------------- #
def layout(node: Node, fs: float, ctx: LayoutContext) -> Box:
    """排版任意节点。"""
    if node is None:
        return Box()
    if isinstance(node, Run):
        return _layout_run(node, fs, ctx)
    if isinstance(node, Plain):
        return _layout_plain(node, fs, ctx)
    if isinstance(node, Space):
        return Box(node.em * fs, 0.0, 0.0, [])
    if isinstance(node, LineBreak):
        return Box(0.0, fs * 0.8, fs * 0.2, [])
    if isinstance(node, Group):
        return _layout_group(node, fs, ctx)
    if isinstance(node, Script):
        return _layout_script(node, fs, ctx)
    if isinstance(node, Frac):
        return _layout_frac(node, fs, ctx)
    if isinstance(node, Sqrt):
        return _layout_sqrt(node, fs, ctx)
    if isinstance(node, Fence):
        return _layout_fence(node, fs, ctx)
    if isinstance(node, Matrix):
        return _layout_matrix(node, fs, ctx)
    return Box()


def _layout_run(node: Run, fs: float, ctx: LayoutContext) -> Box:
    text = node.text
    if not text:
        return Box()
    if not text.strip():
        width = measure_space(fs, ctx.family) * max(1, len(text))
        return Box(width, 0.0, 0.0, [])

    try:
        w, h, d = measure_math(text, fs)
    except MathParseError:
        # mathtext 无法整体解析：先尝试拆成更小的原子，再退化为正体文本
        atoms = atomize(text)
        if len(atoms) > 1:
            return _layout_group(Group(items=atoms), fs, ctx)
        return _layout_plain(literal_node(text), fs, ctx)

    draw = TextDraw(
        x=0.0,
        y=0.0,
        text=normalize_math(text),
        fontsize=fs,
        family=ctx.family,
        math=True,
        color=ctx.color,
        math_fontset=ctx.math_fontset,
    )
    return Box(w, h, d, [draw])


def _layout_plain(node: Plain, fs: float, ctx: LayoutContext) -> Box:
    text = node.text
    if not text:
        return Box()
    weight = "bold" if node.bold else "normal"
    style = "italic" if node.italic else "normal"
    if not text.strip():
        width = measure_space(fs, ctx.family, weight, style) * max(1, len(text))
        return Box(width, 0.0, 0.0, [])
    w, h, d = measure_text(text, fs, ctx.family, weight, style)
    draw = TextDraw(
        x=0.0,
        y=0.0,
        text=text,
        fontsize=fs,
        family=ctx.family,
        math=False,
        color=ctx.color,
        weight=weight,
        style=style,
    )
    return Box(w, h, d, [draw])


def _layout_group(node: Group, fs: float, ctx: LayoutContext) -> Box:
    boxes = [layout(item, fs, ctx) for item in node.items]
    if not boxes:
        return Box()
    gaps = [g * fs for g in node.gaps_em] if node.gaps_em else []
    return hstack(boxes, gaps)


def _layout_script(node: Script, fs: float, ctx: LayoutContext) -> Box:
    base = layout(node.base, fs, ctx) if node.base is not None else Box()
    script_fs = fs * 0.72
    sup = layout(node.sup, script_fs, ctx) if node.sup is not None else None
    sub = layout(node.sub, script_fs, ctx) if node.sub is not None else None

    sup_shift = max(0.45 * fs, base.h - 0.20 * fs)
    sub_shift = max(0.18 * fs, base.d - 0.05 * fs)

    right = 0.0
    if sup is not None:
        right = max(right, sup.w)
    if sub is not None:
        right = max(right, sub.w)
    gap = 0.04 * fs if right else 0.0

    h = base.h
    d = base.d
    draws = list(base.draws)
    x_right = base.w + gap
    if sup is not None:
        translate_box(sup, x_right, -sup_shift)
        draws.extend(sup.draws)
        h = max(h, sup_shift + sup.h)
    if sub is not None:
        translate_box(sub, x_right, sub_shift)
        draws.extend(sub.draws)
        d = max(d, sub_shift + sub.d)
    return Box(base.w + gap + right, h, d, draws)


def _layout_frac(node: Frac, fs: float, ctx: LayoutContext) -> Box:
    num = layout(node.num, fs, ctx)
    den = layout(node.den, fs, ctx)
    gap = 0.22 * fs
    thickness = max(0.5, 0.045 * fs)
    axis = 0.24 * fs                     # 分数线相对基线的抬高量
    bar_y = -axis

    num_baseline = bar_y - thickness / 2.0 - gap - num.d
    den_baseline = bar_y + thickness / 2.0 + gap + den.h
    width = max(num.w, den.w) + 0.55 * fs

    draws: list[Any] = []
    translate_box(num, (width - num.w) / 2.0, num_baseline)
    translate_box(den, (width - den.w) / 2.0, den_baseline)
    draws.extend(num.draws)
    draws.extend(den.draws)
    draws.append(_rect(0.0, bar_y - thickness / 2.0, width, bar_y + thickness / 2.0, ctx.color))

    h = -(min(num_baseline - num.h, bar_y - thickness / 2.0))
    d = max(den_baseline + den.d, bar_y + thickness / 2.0)
    return Box(width, h, d, draws)


def _layout_sqrt(node: Sqrt, fs: float, ctx: LayoutContext) -> Box:
    body = layout(node.body, fs, ctx)
    index = layout(node.index, fs * 0.6, ctx) if node.index is not None else None

    pad = 0.22 * fs
    rad_w = 0.62 * fs
    thickness = max(0.5, 0.05 * fs)
    top = -(body.h + pad)
    bottom = body.d + 0.14 * fs
    mid = -(body.h * 0.45)

    body_x = rad_w + 0.06 * fs
    translate_box(body, body_x, 0.0)

    width = body_x + body.w + 0.12 * fs
    polyline = PathDraw(
        verts=[
            (0.0, mid),
            (rad_w * 0.32, bottom),
            (rad_w * 0.74, top),
            (width, top),
        ],
        codes=None,
        width=max(0.6, 0.055 * fs),
        color=ctx.color,
        fill=False,
    )
    draws: list[Any] = list(body.draws)
    draws.append(polyline)

    h = -top
    d = max(bottom, body.d)
    if index is not None:
        ix = min(rad_w * 0.4, 0.3 * fs)
        translate_box(index, ix, top - index.d - 0.08 * fs)
        draws.extend(index.draws)
        h = max(h, -(top - index.d - 0.08 * fs - index.h))

    return Box(width, h, d, draws)


def _layout_fence(node: Fence, fs: float, ctx: LayoutContext) -> Box:
    body = layout(node.body, fs, ctx)
    pad = 0.10 * fs
    top = -(body.h + pad)
    bottom = body.d + pad

    left_w = delimiter_width(node.left, fs, ctx)
    right_w = delimiter_width(node.right, fs, ctx)
    gap_l = 0.16 * fs if left_w else 0.0
    gap_r = 0.16 * fs if right_w else 0.0

    draws: list[Any] = []
    if left_w:
        draws.extend(_delimiter_draws(node.left, 0.0, top, bottom, fs, ctx))
    x_body = left_w + gap_l
    translate_box(body, x_body, 0.0)
    draws.extend(body.draws)
    if right_w:
        draws.extend(
            _delimiter_draws(node.right, x_body + body.w + gap_r, top, bottom, fs, ctx)
        )

    width = x_body + body.w + gap_r + right_w
    return Box(width, body.h + pad, body.d + pad, draws)


def _parse_col_spec(spec: str, n_cols: int) -> tuple[list[str], list[bool]]:
    """解析 ``array`` 的列格式串，返回 ``(对齐列表, 竖线位置)``。"""
    aligns: list[str] = []
    rules: list[bool] = []
    pending = False
    for ch in spec or "":
        if ch == "|":
            pending = True
        elif ch in "lcr":
            rules.append(pending)
            pending = False
            aligns.append({"l": "left", "c": "center", "r": "right"}[ch])
        elif ch in "pmbX":
            rules.append(pending)
            pending = False
            aligns.append("center")
    rules.append(pending)
    while len(aligns) < n_cols:
        aligns.append("center")
    while len(rules) < n_cols + 1:
        rules.append(False)
    return aligns[:n_cols], rules[: n_cols + 1]


def _layout_matrix(node: Matrix, fs: float, ctx: LayoutContext) -> Box:
    env = (node.env or "matrix").strip()
    cell_fs = fs * ENV_SCALE.get(env, 1.0)
    rows = [row for row in node.rows if row]
    if not rows:
        return Box(0.4 * fs, 0.6 * fs, 0.2 * fs, [])

    grid = [[layout(cell, cell_fs, ctx) for cell in row] for row in rows]
    n_rows = len(grid)
    n_cols = max(len(row) for row in grid)

    align_mode = ENV_ALIGN.get(env, "center")
    if align_mode == "spec":
        col_aligns, col_rules = _parse_col_spec(node.col_spec, n_cols)
    else:
        col_aligns = ["center"] * n_cols
        col_rules = [False] * (n_cols + 1)

    em = cell_fs
    row_gap = 0.50 * em if env in ("cases", "dcases") else 0.42 * em
    col_gap = 0.9 * em
    if env in ("cases", "dcases"):
        col_gap = 1.35 * em
    elif align_mode == "spec":
        col_gap = 0.62 * em
    elif align_mode == "align":
        col_gap = 0.7 * em

    col_w = [0.0] * n_cols
    for row in grid:
        for index, box in enumerate(row):
            col_w[index] = max(col_w[index], box.w)

    row_h: list[float] = []
    row_d: list[float] = []
    for row in grid:
        row_h.append(max((box.h for box in row), default=0.0) or 0.62 * em)
        row_d.append(max((box.d for box in row), default=0.0) or 0.18 * em)

    total_w = sum(col_w) + col_gap * max(0, n_cols - 1)
    total_h = sum(row_h) + sum(row_d) + row_gap * max(0, n_rows - 1)

    col_x: list[float] = []
    x = 0.0
    for index in range(n_cols):
        col_x.append(x)
        x += col_w[index] + col_gap

    content_offset_y = -total_h / 2.0
    content: list[Any] = []
    y = 0.0
    for r, row in enumerate(grid):
        baseline = y + row_h[r]
        for c, box in enumerate(row):
            if align_mode == "align":
                col_align = "right" if c % 2 == 0 else "left"
            else:
                col_align = col_aligns[c] if c < len(col_aligns) else "center"
            if col_align == "left":
                dx = col_x[c]
            elif col_align == "right":
                dx = col_x[c] + col_w[c] - box.w
            else:
                dx = col_x[c] + (col_w[c] - box.w) / 2.0
            translate_box(box, dx, content_offset_y + baseline)
            content.extend(box.draws)
        y += row_h[r] + row_d[r] + row_gap

    # array 的竖线
    if any(col_rules):
        top = content_offset_y - 0.06 * em
        bottom = content_offset_y + total_h + 0.06 * em
        thickness = max(0.5, 0.045 * em)
        for index, has_rule in enumerate(col_rules):
            if not has_rule:
                continue
            if index == 0:
                rx = -0.18 * em
            elif index >= n_cols:
                rx = col_x[-1] + col_w[-1] + 0.18 * em
            else:
                rx = col_x[index] - col_gap / 2.0
            content.append(
                _rect(rx - thickness / 2.0, top, rx + thickness / 2.0, bottom, ctx.color)
            )

    # 定界符（内容整体右移到左定界符之后）
    left_tok, right_tok = ENV_DELIMITERS.get(env, ("none", "none"))
    vpad = 0.10 * em
    d_top = content_offset_y - vpad
    d_bottom = content_offset_y + total_h + vpad
    left_w = delimiter_width(left_tok, cell_fs, ctx)
    right_w = delimiter_width(right_tok, cell_fs, ctx)
    gap_l = 0.14 * em if left_w else 0.0
    gap_r = 0.14 * em if right_w else 0.0

    for draw in content:
        translate(draw, left_w + gap_l, 0.0)

    draws: list[Any] = list(content)
    if left_w:
        draws.extend(_delimiter_draws(left_tok, 0.0, d_top, d_bottom, cell_fs, ctx))
    if right_w:
        draws.extend(
            _delimiter_draws(
                right_tok, left_w + gap_l + total_w + gap_r, d_top, d_bottom, cell_fs, ctx
            )
        )

    width = left_w + gap_l + total_w + gap_r + right_w
    half = total_h / 2.0 + vpad
    return Box(width, half, half, draws)
