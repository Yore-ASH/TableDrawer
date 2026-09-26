"""界面与计算/绘图引擎之间的**唯一数据契约**。

本模块只包含纯数据类与常量表，不依赖 PySide6 或 matplotlib，
因此可以被任意层安全导入。

数据流::

    DataSet(pandas.DataFrame)
        └─ SeriesSpec(引用数据集与列名)
              └─ PlotSpec(坐标轴/主题/画布设置)
                    ├─ core.plotting.build_figure()   -> matplotlib Figure
                    └─ core.codesync.generate_script() -> 独立可运行的 .py 源码

    NumPy 矩阵
        └─ core.matrixops.*  ->  list[ResultItem]  ->  LaTeX 结果视图
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "KINDS_2D",
    "KINDS_3D",
    "THEMES",
    "MARKERS",
    "LINESTYLES",
    "COLORMAPS",
    "SCALES",
    "SeriesSpec",
    "PlotSpec",
    "ResultItem",
    "ResultSheet",
    "kinds_for",
    "kind_label",
]


# --------------------------------------------------------------------------- #
# 常量表
# --------------------------------------------------------------------------- #
#: 二维图表类型：``键 -> 中文名``
KINDS_2D: dict[str, str] = {
    "line": "折线图",
    "scatter": "散点图",
    "bar": "柱形图",
    "barh": "条形图（横向）",
    "stem": "火柴杆图",
    "step": "阶梯图",
    "fill": "面积填充图",
    "errorbar": "误差棒图",
    "hist": "直方图",
    "box": "箱线图",
    "pie": "饼图",
    "quiver": "向量图（2D）",
    "contour": "等高线图",
    "heatmap": "热力图",
}

#: 三维图表类型
KINDS_3D: dict[str, str] = {
    "line3d": "三维折线图",
    "scatter3d": "三维散点图",
    "surface": "三维曲面图",
    "wireframe": "三维线框图",
    "bar3d": "三维柱形图",
    "quiver3d": "三维向量场",
    "contour3d": "三维等高线图",
    "trisurf": "三角剖分曲面",
}

#: matplotlib 样式名 -> 中文名（``light`` 表示使用 matplotlib 默认样式）
THEMES: dict[str, str] = {
    "light": "明亮（默认）",
    "dark_background": "暗色",
    "seaborn-v0_8-whitegrid": "Seaborn 白网格",
    "seaborn-v0_8-darkgrid": "Seaborn 暗网格",
    "ggplot": "ggplot",
    "bmh": "BMH",
    "classic": "经典",
    "Solarize_Light2": "Solarize Light",
    "fivethirtyeight": "FiveThirtyEight",
}

#: 标记符号 -> 说明
MARKERS: dict[str, str] = {
    "": "无",
    "o": "圆点",
    "s": "方块",
    "^": "上三角",
    "v": "下三角",
    "D": "菱形",
    "*": "星形",
    "+": "加号",
    "x": "叉号",
    ".": "小点",
    "P": "粗加号",
}

#: 线型 -> 说明
LINESTYLES: dict[str, str] = {
    "-": "实线",
    "--": "虚线",
    "-.": "点划线",
    ":": "点线",
    "": "无线条",
}

#: 常用色图
COLORMAPS: list[str] = [
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "jet", "rainbow", "coolwarm", "RdBu_r", "Spectral", "seismic",
    "twilight", "Greys", "Blues", "YlGnBu", "hot", "terrain",
]

#: 坐标轴缩放方式
SCALES: dict[str, str] = {"linear": "线性", "log": "对数", "symlog": "对称对数", "logit": "Logit"}


def kinds_for(projection: str) -> dict[str, str]:
    """返回指定投影（``"2d"`` / ``"3d"``）支持的图表类型表。"""
    return KINDS_3D if str(projection).lower() == "3d" else KINDS_2D


def kind_label(kind: str) -> str:
    """图表类型的中文名，找不到时返回原键。"""
    return KINDS_2D.get(kind) or KINDS_3D.get(kind) or kind


# --------------------------------------------------------------------------- #
# 绘图规格
# --------------------------------------------------------------------------- #
@dataclass
class SeriesSpec:
    """一条数据序列的绘图描述。

    列名约定（空字符串表示"不使用"）:

    * ``x`` 为空 -> 使用 DataFrame 的行索引
    * ``y`` 支持逗号分隔的多列（折线/散点/柱形等会画多条）
    * ``z`` 三维图的第三维；``surface``/``wireframe`` 需要 ``z`` 为网格列名
    * ``u``/``v``/``w`` 向量图（quiver）的分量
    * ``expr`` 非空时用它（NumPy 表达式，变量 ``x``/``y``/``i``）派生 y
    """

    dataset: str = ""
    kind: str = "line"
    x: str = ""
    y: str = ""
    z: str = ""
    u: str = ""
    v: str = ""
    w: str = ""
    yerr: str = ""
    label: str = ""
    color: str = ""
    cmap: str = "viridis"
    marker: str = ""
    linestyle: str = "-"
    linewidth: float = 2.0
    markersize: float = 6.0
    alpha: float = 1.0
    bar_width: float = 0.8
    bins: int = 20
    levels: int = 12
    scale: float = 1.0
    normalize: bool = False
    cumulative: bool = False
    density: bool = False
    horizontal: bool = False
    filled: bool = True
    smooth: bool = False
    expr: str = ""
    zorder: int = 2
    grid_shape: tuple[int, int] | None = None
    options: dict[str, Any] = field(default_factory=dict)

    # -- 便捷方法 ---------------------------------------------------------- #
    def y_columns(self) -> list[str]:
        """把 ``y`` 字段拆成列名列表。"""
        return [c.strip() for c in str(self.y).split(",") if c.strip()]

    def used_columns(self) -> list[str]:
        """该序列引用到的所有列名（去重，保持出现顺序）。"""
        out: list[str] = []
        for name in [self.x, self.z, self.u, self.v, self.w, self.yerr, *self.y_columns()]:
            name = str(name).strip()
            if name and name not in out:
                out.append(name)
        return out

    def is_vector(self) -> bool:
        return self.kind in {"quiver", "quiver3d"}

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeriesSpec":
        return _build(cls, data)


@dataclass
class PlotSpec:
    """整张图（画布 + 坐标轴 + 若干序列）的描述。"""

    projection: str = "2d"              # "2d" | "3d"
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    zlabel: str = ""
    series: list[SeriesSpec] = field(default_factory=list)

    theme: str = "light"
    figsize: tuple[float, float] = (8.0, 5.0)
    dpi: int = 110
    grid: bool = True
    legend: bool = True
    legend_loc: str = "best"
    legend_frame: bool = True
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    zlim: tuple[float, float] | None = None
    xscale: str = "linear"
    yscale: str = "linear"
    zscale: str = "linear"
    equal_aspect: bool = False
    tight_layout: bool = True
    transparent: bool = False
    font_family: str = ""
    font_size: float = 11.0
    colorcycle: list[str] = field(default_factory=list)
    view_elev: float = 30.0
    view_azim: float = -60.0
    colorbar: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    # -- 便捷方法 ---------------------------------------------------------- #
    def used_datasets(self) -> list[str]:
        """该图中引用到的数据集名（去重，保持出现顺序）。"""
        out: list[str] = []
        for series in self.series:
            if series.dataset and series.dataset not in out:
                out.append(series.dataset)
        return out

    def is_3d(self) -> bool:
        return str(self.projection).lower() == "3d"

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["series"] = [s.to_dict() for s in self.series]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlotSpec":
        payload = dict(data or {})
        raw_series = payload.pop("series", []) or []
        spec = _build(cls, payload)
        spec.series = [SeriesSpec.from_dict(item) for item in raw_series]
        return spec


# --------------------------------------------------------------------------- #
# 计算结果
# --------------------------------------------------------------------------- #
@dataclass
class ResultItem:
    """一条计算结果，供 LaTeX 结果视图渲染。

    * ``title``  : 中文小标题（纯文本，可含 ``$..$`` 行内公式）
    * ``latex``  : LaTeX 数学体（**不含** 首尾 ``$``），可包含
      ``\\begin{pmatrix}`` 等环境
    * ``note``   : 补充说明（纯文本）
    * ``level``  : 0 普通 / 1 章节标题 / 2 次要说明
    * ``kind``   : ``value`` | ``matrix`` | ``vector`` | ``text``
    """

    title: str = ""
    latex: str = ""
    note: str = ""
    level: int = 0
    kind: str = "value"
    plain: str = ""          # 纯文本版本，用于「复制文本结果」
    payload: Any = None      # 原始 NumPy 结果，便于二次利用
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResultSheet:
    """一组结果的集合。"""

    title: str = ""
    items: list[ResultItem] = field(default_factory=list)
    created: str = ""

    def add(self, item: ResultItem) -> ResultItem:
        self.items.append(item)
        return item

    def extend(self, items: Iterable[ResultItem]) -> None:
        self.items.extend(items)

    def plain_text(self) -> str:
        blocks: list[str] = []
        if self.title:
            blocks.append(self.title)
            blocks.append("=" * max(6, len(self.title) * 2))
        for item in self.items:
            if item.title:
                blocks.append(item.title)
            body = item.plain or item.latex
            if body:
                blocks.append(body)
            if item.note:
                blocks.append(f"# {item.note}")
            blocks.append("")
        return "\n".join(blocks).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# 内部：容忍未知字段的反序列化
# --------------------------------------------------------------------------- #
def _build(cls: type, data: dict[str, Any]) -> Any:
    """按数据类的字段名过滤未知键后构造实例（保证配置向后兼容）。"""
    if not isinstance(data, dict):
        return cls()
    names = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in names}
    for key in ("xlim", "ylim", "zlim", "figsize", "grid_shape"):
        if key in kwargs and isinstance(kwargs[key], list):
            kwargs[key] = tuple(kwargs[key])
    return cls(**kwargs)
