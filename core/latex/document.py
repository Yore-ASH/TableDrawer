"""文档层：把若干「块」按给定宽度排版成整体。

支持的块类型
------------
* :class:`TextBlock` —— 段落，支持 ``$行内公式$``，可自动换行，中英文混排
* :class:`MathBlock` —— 独立公式行（可含 ``\\begin{pmatrix}`` 等环境），
  过宽时自动缩小字号以适应
* :class:`RuleBlock` —— 水平分隔线
* :class:`SpaceBlock` —— 垂直留白

用法::

    doc = LatexDocument()
    doc.add(TextBlock("矩阵的逆", bold=True))
    doc.add(MathBlock(r"A^{-1} = \\begin{pmatrix} -2 & 1 \\\\ 1.5 & -0.5 \\end{pmatrix}"))
    layout = layout_document(doc, width=600)
    fig = render_document(doc, width=600)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .layout import Box, LayoutContext, PathDraw, TextDraw, hstack, layout, translate, translate_box
from .measure import text_strut
from .nodes import Group, LineBreak, Node, Plain, Run, Space
from .parser import parse_math, strip_math_delimiters
from .render import attach_agg, figure_from_layout

__all__ = [
    "TextBlock",
    "MathBlock",
    "RuleBlock",
    "SpaceBlock",
    "LatexDocument",
    "DocumentLayout",
    "StripSlice",
    "layout_document",
    "render_document",
    "render_strip",
    "parse_paragraph",
]


# --------------------------------------------------------------------------- #
# 块定义
# --------------------------------------------------------------------------- #
@dataclass
class TextBlock:
    """一段可自动换行的文本，``$...$`` 之间按数学模式渲染。"""

    text: str = ""
    fontsize: float | None = None
    align: str = ""                 # "" -> 用文档默认；left / center / right
    color: str | None = None
    bold: bool = False
    italic: bool = False
    space_before: float = 7.0
    space_after: float = 1.0
    indent: float = 0.0
    line_spacing: float = 0.0       # 0 -> 用文档默认
    scale_min: float = 0.45
    wrap: bool = True


@dataclass
class MathBlock(TextBlock):
    """整块按数学模式解析，支持矩阵等环境；过宽时按比例缩小字号。"""

    space_before: float = 8.0
    space_after: float = 4.0


@dataclass
class RuleBlock:
    """水平分隔线。"""

    thickness: float = 0.9
    color: str | None = None
    space_before: float = 9.0
    space_after: float = 7.0
    inset: float = 0.0


@dataclass
class SpaceBlock:
    """垂直留白。"""

    height: float = 8.0


@dataclass
class LatexDocument:
    """一份可排版的 LaTeX 文档。"""

    blocks: list[Any] = field(default_factory=list)
    fontsize: float = 12.0
    family: str = ""
    math_family: str = ""
    delim_family: str = ""
    color: str = "#111111"
    math_fontset: str = "cm"
    line_spacing: float = 1.26
    padding: float = 12.0
    align: str = "left"
    background: str | None = None

    def add(self, block: Any) -> Any:
        self.blocks.append(block)
        return block

    def extend(self, blocks: Iterable[Any]) -> None:
        self.blocks.extend(blocks)

    def clear(self) -> None:
        self.blocks.clear()

    def __len__(self) -> int:
        return len(self.blocks)

    def context(self) -> LayoutContext:
        """构建排版上下文。

        字体默认取当前 matplotlib 生效的**完整回退链**（由
        :func:`core.mplsetup.apply_font_preset` 配置，形如
        Times New Roman → SimSun → DejaVu Sans）。这样：

        * 英文与数字用西文字体、中文自动落到宋体；
        * 中文字体缺失的上下标字符（``⁻ ᵢ ᵀ ⱼ`` 等）再回退到 DejaVu Sans，
          不会渲染成豆腐块。
        """
        import matplotlib

        from ..mplsetup import active_families
        from .measure import math_fontset

        try:
            fallback = tuple(active_families()) or tuple(
                str(name) for name in matplotlib.rcParams.get("font.sans-serif", ())
            )
        except Exception:
            fallback = tuple(str(name) for name in matplotlib.rcParams.get("font.sans-serif", ()))
        if not fallback:
            fallback = ("DejaVu Sans",)

        family = self.family or fallback
        math_family = self.math_family or fallback
        # 定界符优先用 STIX（数学括号字形最标准），缺失时回退
        delim_family = self.delim_family or ("STIXGeneral",) + tuple(fallback)
        return LayoutContext(
            family=family,
            math_family=math_family,
            delim_family=delim_family,
            color=self.color,
            math_fontset=self.math_fontset or math_fontset(),
        )


@dataclass
class DocumentLayout:
    """排版结果：整体尺寸与绝对坐标绘制指令。

    ``width`` 是排版时给定的可用宽度，``content_width`` 是**内容实际占据**的宽度
    （最右侧绘制指令的右边界 + 内边距）。渲染成紧凑位图（例如界面按钮上的
    LaTeX 图标）时应该用 ``content_width``，否则会得到一张大量留白的巨图。
    """

    width: float
    height: float
    draws: list[Any] = field(default_factory=list)
    context: LayoutContext | None = None
    content_width: float = 0.0


# --------------------------------------------------------------------------- #
# 段落解析
# --------------------------------------------------------------------------- #
def parse_paragraph(text: str) -> list[Node]:
    """把 ``$..$`` 混排的文本切分为 :class:`Plain` / 数学节点序列。"""
    nodes: list[Node] = []
    buffer: list[str] = []
    i = 0
    n = len(text or "")

    def flush() -> None:
        chunk = "".join(buffer)
        buffer.clear()
        if chunk:
            nodes.append(Plain(text=chunk))

    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] == "$":
            buffer.append("$")
            i += 2
            continue
        if ch == "$":
            j = i + 1
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if text[j] == "$":
                    break
                j += 1
            if j >= n:
                buffer.append(ch)
                i += 1
                continue
            body = text[i + 1 : j]
            flush()
            if body.strip():
                nodes.extend(parse_math(body))
            i = j + 1
            continue
        buffer.append(ch)
        i += 1

    flush()
    return nodes


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x2E80 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE4F
        or 0xFF00 <= code <= 0xFFEF
    )


_CJK_PUNCT = "，。、；：！？（）《》〈〉“”‘’【】—…·"


def _split_plain_units(text: str) -> list[str]:
    """把普通文本拆成可换行的最小单元（拉丁词整体、中文逐字）。"""
    units: list[str] = []
    buffer: list[str] = []
    for ch in text:
        if ch == "\n":
            if buffer:
                units.append("".join(buffer))
                buffer.clear()
            units.append("\n")
        elif ch in " \t":
            if buffer:
                units.append("".join(buffer))
                buffer.clear()
            units.append(" " if ch == " " else "\t")
        elif _is_cjk(ch) or ch in _CJK_PUNCT:
            if buffer:
                units.append("".join(buffer))
                buffer.clear()
            units.append(ch)
        else:
            buffer.append(ch)
    if buffer:
        units.append("".join(buffer))
    return units


def _units_from_nodes(nodes: list[Node]) -> list[Node]:
    """把段落节点展开成可换行的单元序列。"""
    units: list[Node] = []
    for node in nodes:
        if isinstance(node, Plain):
            for token in _split_plain_units(node.text):
                if token == "\n":
                    units.append(LineBreak())
                elif token.strip() == "":
                    units.append(Plain(text=token))
                else:
                    units.append(Plain(text=token, bold=node.bold, italic=node.italic))
        else:
            units.append(node)
    return units


# --------------------------------------------------------------------------- #
# 文档排版
# --------------------------------------------------------------------------- #
def _is_blank(node: Node) -> bool:
    return isinstance(node, Space) or (isinstance(node, Plain) and not node.text.strip())


def _wrap_units(
    units: list[Node], max_width: float, fs: float, ctx: LayoutContext
) -> list[Box]:
    """贪心折行，返回每行的 :class:`Box`。"""
    lines: list[list[Box]] = []
    current: list[Box] = []
    current_nodes: list[Node] = []
    current_w = 0.0

    def flush() -> None:
        nonlocal current, current_nodes, current_w
        while current and _is_blank(current_nodes[-1]):
            current.pop()
            current_nodes.pop()
        if current:
            lines.append(current)
        current = []
        current_nodes = []
        current_w = 0.0

    for node in units:
        if isinstance(node, LineBreak):
            flush()
            continue
        box = layout(node, fs, ctx)
        if current and current_w + box.w > max_width:
            flush()
            if _is_blank(node):
                continue
        current.append(box)
        current_nodes.append(node)
        current_w += box.w

    flush()
    if not lines:
        lines.append([])
    return [hstack(line) if line else Box(0.0, fs * 0.8, fs * 0.2, []) for line in lines]


def layout_document(document: LatexDocument, width: float) -> DocumentLayout:
    """把文档排版到指定宽度（点），返回带绝对坐标的绘制指令。"""
    ctx = document.context()
    padding = max(0.0, float(document.padding))
    width = max(float(width), 60.0)
    available = max(width - 2.0 * padding, 40.0)

    draws: list[Any] = []
    y = padding
    max_right = 0.0

    for block in document.blocks:
        if isinstance(block, SpaceBlock):
            y += block.height
            continue

        if isinstance(block, RuleBlock):
            y += block.space_before
            color = block.color or ctx.color
            x0 = padding + block.inset
            x1 = width - padding - block.inset
            thickness = max(block.thickness, 0.4)
            draws.append(
                PathDraw(
                    verts=[(x0, y), (x1, y), (x1, y + thickness), (x0, y + thickness), (x0, y)],
                    color=color,
                    fill=True,
                )
            )
            max_right = max(max_right, x1)
            y += thickness + block.space_after
            continue

        # ---- 文本块 ---------------------------------------------------- #
        fs = float(block.fontsize or document.fontsize)
        align = block.align or document.align or "left"
        block_ctx = LayoutContext(
            family=ctx.family,
            math_family=ctx.math_family,
            delim_family=ctx.delim_family,
            color=block.color or ctx.color,
            math_fontset=ctx.math_fontset,
        )
        spacing = float(block.line_spacing or document.line_spacing) or 1.26
        max_width = max(available - block.indent, 30.0)
        y += block.space_before

        if isinstance(block, MathBlock):
            body = strip_math_delimiters(block.text)
            box = layout(Group(items=parse_math(body)), fs, block_ctx)
            if box.w > max_width > 0:
                shrink = max(max_width / box.w, float(block.scale_min))
                fs = max(fs * shrink, 5.5)
                box = layout(Group(items=parse_math(body)), fs, block_ctx)
            line_boxes = [box]
        else:
            nodes = parse_paragraph(block.text)
            units = _units_from_nodes(nodes)
            line_boxes = (
                _wrap_units(units, max_width, fs, block_ctx)
                if block.wrap
                else [hstack([layout(u, fs, block_ctx) for u in units])]
            )

        strut_h, strut_d = text_strut(fs, block_ctx.family)
        trailing_leading = 0.0
        for line_box in line_boxes:
            ascent = max(line_box.h, strut_h)
            descent = max(line_box.d, strut_d)
            leading = (ascent + descent) * max(spacing - 1.0, 0.0)
            trailing_leading = leading
            baseline = y + ascent
            if align == "center":
                x = padding + block.indent + max(0.0, (max_width - line_box.w) / 2.0)
            elif align == "right":
                x = padding + block.indent + max(0.0, max_width - line_box.w)
            else:
                x = padding + block.indent
            translate_box(line_box, x, baseline)
            draws.extend(line_box.draws)
            max_right = max(max_right, x + line_box.w)
            y = baseline + descent + leading
        y -= trailing_leading
        y += block.space_after

    height = max(y + padding, 1.0)
    content_width = min(max(max_right + padding, 1.0), width)
    return DocumentLayout(
        width=width, height=height, draws=draws, context=ctx, content_width=content_width
    )


def render_document(
    document: LatexDocument,
    width: float,
    *,
    dpi: int = 110,
    background: str | None = None,
    figure: Any | None = None,
):
    """排版并渲染为 matplotlib ``Figure``。

    传入 ``figure`` 时复用同一个 Figure 对象（避免嵌入 Qt 时反复重建画布控件）。
    """
    result = layout_document(document, width)
    return figure_from_layout(
        result.width,
        result.height,
        result.draws,
        dpi=dpi,
        background=background if background is not None else document.background,
        figure=figure,
    )


@dataclass(frozen=True)
class StripSlice:
    """条带渲染中单个片段在整图里的像素区域与逻辑尺寸（点）。"""

    x: int
    y: int
    width: int
    height: int
    width_pt: float
    height_pt: float


def render_strip(
    documents: Sequence[LatexDocument],
    *,
    width: float = 8000.0,
    dpi: int = 192,
    background: str | None = None,
) -> tuple[Any, list[StripSlice]]:
    """把多个文档竖直拼成一条「胶片」，**只渲染一次**。

    逐张渲染时，每张都要新建 Figure + 画布 + 走一遍 Agg 绘制，单张成本
    50 ms 上下；而实际排版只要 0.5 ms。界面上一屏几十个 LaTeX 标签（例如
    55 个矩阵运算按钮）用这个函数一次画完，总耗时降到几百毫秒。

    返回 ``(RGBA 数组, 每个片段的裁剪框)``，数组第 0 行对应最上面的文档。
    """
    import numpy as np

    results = [layout_document(document, width=width) for document in documents]
    if not results:
        return np.zeros((0, 0, 4), dtype=np.uint8), []

    total_height = 0.0
    total_width = 0.0
    for result in results:
        total_height += result.height
        total_width = max(total_width, result.content_width)
    total_height = max(total_height, 1.0)
    total_width = max(total_width, 1.0)

    scale = dpi / 72.0
    draws: list[Any] = []
    slices: list[StripSlice] = []
    cursor = 0.0
    for result in results:
        for draw in result.draws:
            translate(draw, 0.0, cursor)
            draws.append(draw)
        slices.append(
            StripSlice(
                x=0,
                y=int(round(cursor * scale)),
                width=max(int(round(result.content_width * scale)), 1),
                height=max(int(round(result.height * scale)), 1),
                width_pt=result.content_width,
                height_pt=result.height,
            )
        )
        cursor += result.height

    figure = figure_from_layout(
        total_width, total_height, draws, dpi=dpi, background=background
    )
    canvas = attach_agg(figure)
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    return buffer, slices
