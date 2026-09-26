"""把 :class:`core.spec.ResultSheet` 转换为可排版的 LaTeX 文档。

这一层是「计算结果」与「排版引擎」之间的适配器：矩阵运算只负责产生
``ResultItem``（标题 + LaTeX 数学体 + 备注），样式与排版全部在这里决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from ..spec import ResultItem, ResultSheet
from .document import LatexDocument, MathBlock, RuleBlock, SpaceBlock, TextBlock

__all__ = ["ResultStyle", "blocks_for_sheet", "document_for_sheet", "LIGHT_RESULT_STYLE", "DARK_RESULT_STYLE"]


@dataclass
class ResultStyle:
    """结果文档的排版样式。"""

    fontsize: float = 12.0
    heading_fontsize: float = 15.0
    title_fontsize: float = 11.5
    note_fontsize: float = 10.0
    text_color: str = "#111827"
    title_color: str = "#1d4ed8"
    note_color: str = "#6b7280"
    rule_color: str = "#d8dee9"
    show_rules: bool = True
    family: str = ""
    math_family: str = ""
    delim_family: str = ""
    math_fontset: str = "cm"
    line_spacing: float = 1.26
    padding: float = 14.0
    background: str | None = None

    def with_colors(self, *, text: str, title: str, note: str, rule: str) -> "ResultStyle":
        return replace(self, text_color=text, title_color=title, note_color=note, rule_color=rule)


LIGHT_RESULT_STYLE = ResultStyle()

DARK_RESULT_STYLE = ResultStyle(
    text_color="#e5e7eb",
    title_color="#60a5fa",
    note_color="#9ca3af",
    rule_color="#374151",
)

#: 样式预设名 -> 样式
STYLE_PRESETS: dict[str, ResultStyle] = {
    "light": LIGHT_RESULT_STYLE,
    "dark": DARK_RESULT_STYLE,
}


def blocks_for_sheet(
    sheet: ResultSheet | Sequence[ResultItem] | None,
    style: ResultStyle | None = None,
) -> list[Any]:
    """把结果集转换为文档块列表。"""
    style = style or LIGHT_RESULT_STYLE
    if sheet is None:
        return []
    items: Sequence[ResultItem] = sheet.items if isinstance(sheet, ResultSheet) else sheet
    title = sheet.title if isinstance(sheet, ResultSheet) else ""

    blocks: list[Any] = []
    if title:
        blocks.append(
            TextBlock(
                text=title,
                fontsize=style.heading_fontsize,
                bold=True,
                color=style.title_color,
                space_before=0.0,
                space_after=4.0,
            )
        )
        if style.show_rules:
            blocks.append(
                RuleBlock(thickness=1.2, color=style.rule_color, space_before=0.0, space_after=10.0)
            )

    for index, item in enumerate(items):
        if item.level == 1:
            blocks.append(
                TextBlock(
                    text=item.title or item.note,
                    fontsize=style.heading_fontsize,
                    bold=True,
                    color=style.title_color,
                    space_before=4.0 if index == 0 else 16.0,
                    space_after=4.0,
                )
            )
            if style.show_rules:
                blocks.append(
                    RuleBlock(thickness=0.9, color=style.rule_color, space_before=0.0, space_after=8.0)
                )
            continue

        if index > 0 and style.show_rules:
            blocks.append(
                RuleBlock(thickness=0.6, color=style.rule_color, space_before=2.0, space_after=10.0)
            )

        if item.title:
            blocks.append(
                TextBlock(
                    text=item.title,
                    fontsize=style.title_fontsize,
                    bold=True,
                    color=style.title_color,
                    space_before=0.0,
                    space_after=1.0,
                    line_spacing=1.1,
                )
            )

        if item.latex:
            blocks.append(
                MathBlock(
                    text=item.latex,
                    fontsize=style.fontsize,
                    color=style.text_color,
                    space_before=2.0 if item.title else 0.0,
                    space_after=2.0,
                )
            )
        elif item.plain:
            blocks.append(
                MathBlock(
                    text=item.plain,
                    fontsize=style.fontsize,
                    color=style.text_color,
                    space_before=2.0,
                    space_after=2.0,
                )
            )

        if item.note:
            blocks.append(
                TextBlock(
                    text=item.note,
                    fontsize=style.note_fontsize,
                    italic=True,
                    color=style.note_color,
                    indent=10.0,
                    space_before=1.0,
                    space_after=2.0,
                    line_spacing=1.2,
                )
            )

    if not blocks:
        blocks.append(
            TextBlock(
                text="（暂无结果）",
                fontsize=style.fontsize,
                color=style.note_color,
                space_before=0.0,
            )
        )
    return blocks


def document_for_sheet(
    sheet: ResultSheet | Sequence[ResultItem] | None,
    style: ResultStyle | None = None,
) -> LatexDocument:
    """构建一份可渲染的结果文档。"""
    style = style or LIGHT_RESULT_STYLE
    document = LatexDocument(
        fontsize=style.fontsize,
        family=style.family,
        math_family=style.math_family,
        delim_family=style.delim_family,
        color=style.text_color,
        math_fontset=style.math_fontset,
        line_spacing=style.line_spacing,
        padding=style.padding,
        background=style.background,
    )
    document.extend(blocks_for_sheet(sheet, style))
    return document
