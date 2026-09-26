"""把排版结果渲染为 matplotlib ``Figure``，并导出 SVG / PNG / PDF。

* 文档的「数据坐标」与「点」一一对应：图片物理尺寸取 ``宽/72 × 高/72`` 英寸，
  因此 ``fontsize=12`` 的文本在数据坐标里恰好占 12 个单位，无需任何缩放补偿。
* 导出 SVG 时用 ``svg.fonttype='path'``（在 :mod:`core.mplsetup` 中设置），
  文字会转成矢量轮廓，任何机器打开都不会缺字体。
"""

from __future__ import annotations

import io
from typing import Any, Iterable

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from .layout import PathDraw, TextDraw

__all__ = [
    "figure_from_layout",
    "figure_to_svg",
    "figure_to_png_bytes",
    "figure_to_pdf_bytes",
    "attach_agg",
]


def attach_agg(figure: Figure) -> FigureCanvasAgg:
    """确保 figure 拥有一个 Agg 画布（``savefig`` 需要）。"""
    canvas = getattr(figure, "canvas", None)
    if isinstance(canvas, FigureCanvasAgg):
        return canvas
    return FigureCanvasAgg(figure)


def figure_from_layout(
    width: float,
    height: float,
    draws: Iterable[Any],
    *,
    dpi: int = 110,
    background: str | None = None,
    figure: Figure | None = None,
) -> Figure:
    """按排版结果构建 Figure。

    ``background=None`` 时画布透明（便于叠加在 Qt 控件背景上）。

    传入 ``figure`` 时会**复用**同一个 Figure 对象（先 ``clear()`` 再重建内容）。
    这一点对嵌入 Qt 的界面很重要：复用 Figure 就不需要重建
    ``FigureCanvasQTAgg`` 控件，避免反复重排导致的界面抖动。
    """
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    if figure is None:
        figure = Figure(figsize=(width / 72.0, height / 72.0), dpi=dpi)
    else:
        figure.clear()
        figure.set_size_inches(width / 72.0, height / 72.0, forward=True)
        figure.set_dpi(dpi)
        figure.patch.set_visible(True)

    if background:
        figure.patch.set_facecolor(background)
        figure.patch.set_alpha(1.0)
    else:
        figure.patch.set_alpha(0.0)

    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.set_xlim(0.0, width)
    axes.set_ylim(height, 0.0)          # y 轴向下，与排版坐标一致
    axes.set_axis_off()
    axes.patch.set_alpha(0.0)

    text_items: list[TextDraw] = []
    path_items: list[PathDraw] = []
    for draw in draws:
        if isinstance(draw, TextDraw):
            text_items.append(draw)
        elif isinstance(draw, PathDraw):
            path_items.append(draw)

    for draw in path_items:
        path = MplPath(draw.verts, draw.codes) if draw.codes else MplPath(draw.verts)
        if draw.fill:
            patch = PathPatch(
                path,
                facecolor=draw.color,
                edgecolor="none",
                linewidth=0.0,
                clip_on=False,
                zorder=2,
            )
        else:
            patch = PathPatch(
                path,
                fill=False,
                edgecolor=draw.color,
                linewidth=max(draw.width, 0.1),
                capstyle="round",
                joinstyle="round",
                clip_on=False,
                zorder=2,
            )
        axes.add_patch(patch)

    for draw in text_items:
        kwargs: dict[str, Any] = {
            "fontsize": draw.fontsize,
            "color": draw.color,
            "ha": "left",
            "va": "baseline",
            "clip_on": False,
            "zorder": 3,
        }
        if draw.math:
            kwargs["math_fontfamily"] = draw.math_fontset
        else:
            # family 可能是回退列表（tuple），matplotlib 支持传列表做字体回退
            kwargs["family"] = list(draw.family) if not isinstance(draw.family, str) else draw.family
        if draw.weight != "normal":
            kwargs["weight"] = draw.weight
        if draw.style != "normal":
            kwargs["style"] = draw.style
        axes.text(draw.x, draw.y, draw.text, **kwargs)

    attach_agg(figure)
    return figure


def figure_to_svg(figure: Figure, **savefig_kwargs: Any) -> str:
    """把图渲染为 SVG 文本（不写磁盘）。"""
    attach_agg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", **savefig_kwargs)
    return buffer.getvalue().decode("utf-8", errors="replace")


def figure_to_png_bytes(figure: Figure, dpi: int | None = None, **savefig_kwargs: Any) -> bytes:
    """把图渲染为 PNG 字节串。"""
    attach_agg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, **savefig_kwargs)
    return buffer.getvalue()


def figure_to_pdf_bytes(figure: Figure, **savefig_kwargs: Any) -> bytes:
    """把图渲染为 PDF 字节串。"""
    attach_agg(figure)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf", **savefig_kwargs)
    return buffer.getvalue()
