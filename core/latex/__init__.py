"""自研 LaTeX 排版引擎（matplotlib 之上）。

为什么不用 mathtext 直接渲染？
------------------------------
matplotlib 的 ``mathtext`` **不支持** ``\\begin{...}`` 环境（实测抛
``Unknown symbol: \\begin``），而矩阵、方程组、对齐公式恰恰是数学计算结果的
主要展示形式。因此本引擎只补上这一块：

* :mod:`core.latex.parser` —— 把 LaTeX 子集解析成 AST，遇到 ``\\begin{...}``
  走自研路径，其余原样交给 mathtext
* :mod:`core.latex.layout` —— 计算每个元素的宽/高/深与绝对坐标，
  可伸缩定界符用字体轮廓纵向拉伸得到（矢量、无锯齿）
* :mod:`core.latex.render` —— 输出 matplotlib ``Figure``，可导出 SVG/PNG/PDF
* :mod:`core.latex.document` —— 段落折行、中英文混排、整篇文档排版

快速上手::

    from core.latex import LatexDocument, MathBlock, TextBlock, render_document

    doc = LatexDocument()
    doc.add(TextBlock("方阵 $A$ 的逆矩阵：", bold=True))
    doc.add(MathBlock(r"A^{-1} = \\begin{pmatrix} -2 & 1 \\\\ 1.5 & -0.5 \\end{pmatrix}"))
    figure = render_document(doc, width=560)
    figure.savefig("out.svg")
"""

from __future__ import annotations

from .document import (
    DocumentLayout,
    LatexDocument,
    MathBlock,
    RuleBlock,
    SpaceBlock,
    StripSlice,
    TextBlock,
    layout_document,
    parse_paragraph,
    render_document,
    render_strip,
)
from .layout import Box, LayoutContext, PathDraw, TextDraw, layout
from .measure import MathParseError, clear_cache, measure_math, measure_text
from .nodes import (
    Fence,
    Frac,
    Group,
    Matrix,
    Node,
    Plain,
    Run,
    Script,
    Space,
    Sqrt,
    as_group,
)
from .parser import LatexSyntaxError, atomize, contains_env, parse_latex, parse_math
from .render import (
    figure_from_layout,
    figure_to_pdf_bytes,
    figure_to_png_bytes,
    figure_to_svg,
)
from .results import (
    DARK_RESULT_STYLE,
    LIGHT_RESULT_STYLE,
    STYLE_PRESETS,
    ResultStyle,
    blocks_for_sheet,
    document_for_sheet,
)

__all__ = [
    # 文档
    "LatexDocument",
    "TextBlock",
    "MathBlock",
    "RuleBlock",
    "SpaceBlock",
    "DocumentLayout",
    "StripSlice",
    "layout_document",
    "render_document",
    "render_strip",
    "parse_paragraph",
    # 底层
    "Box",
    "LayoutContext",
    "TextDraw",
    "PathDraw",
    "layout",
    "MathParseError",
    "LatexSyntaxError",
    "measure_math",
    "measure_text",
    "clear_cache",
    # AST / 解析
    "Node",
    "Run",
    "Plain",
    "Space",
    "Group",
    "Script",
    "Frac",
    "Sqrt",
    "Fence",
    "Matrix",
    "as_group",
    "parse_math",
    "parse_latex",
    "atomize",
    "contains_env",
    # 渲染
    "figure_from_layout",
    "figure_to_svg",
    "figure_to_png_bytes",
    "figure_to_pdf_bytes",
    # 结果适配
    "ResultStyle",
    "LIGHT_RESULT_STYLE",
    "DARK_RESULT_STYLE",
    "STYLE_PRESETS",
    "blocks_for_sheet",
    "document_for_sheet",
]
