"""LaTeX 子集的 AST 节点与查表常量。

渲染管线::

    LaTeX 字符串
        └─ parser.parse_math()      -> list[Node]
              └─ layout.layout()    -> Box(宽/高/深 + 绘制指令)
                    └─ render.build_figure() -> matplotlib Figure -> SVG/PNG

节点只描述结构，不携带尺寸；尺寸在 :mod:`core.latex.layout` 中计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Node",
    "Run",
    "Plain",
    "Space",
    "LineBreak",
    "Group",
    "Script",
    "Frac",
    "Sqrt",
    "Fence",
    "Matrix",
    "ENV_DELIMITERS",
    "ENV_ALIGN",
    "ENV_SCALE",
    "DELIM_CHARS",
    "MATRIX_ENVS",
    "as_group",
]


class Node:
    """所有节点的基类。"""


@dataclass
class Run(Node):
    """交给 mathtext 渲染的数学原子（原样保留 LaTeX 命令）。"""

    text: str = ""


@dataclass
class Plain(Node):
    """普通文本原子（正体、支持中文）。"""

    text: str = ""
    bold: bool = False
    italic: bool = False


@dataclass
class Space(Node):
    """水平空白，宽度以 em 为单位。"""

    em: float = 0.35


@dataclass
class LineBreak(Node):
    """强制换行（用于段落内手工折行）。"""


@dataclass
class Group(Node):
    """水平排列的一组节点；``gaps_em`` 给出相邻节点之间的额外间距。"""

    items: list[Node] = field(default_factory=list)
    gaps_em: list[float] = field(default_factory=list)


@dataclass
class Script(Node):
    """上标 / 下标。"""

    base: Node | None = None
    sup: Group | None = None
    sub: Group | None = None


@dataclass
class Frac(Node):
    """分式。"""

    num: Group = field(default_factory=Group)
    den: Group = field(default_factory=Group)


@dataclass
class Sqrt(Node):
    """根式，``index`` 非空表示 n 次根。"""

    body: Group = field(default_factory=Group)
    index: Group | None = None


@dataclass
class Fence(Node):
    """由 ``\\left`` / ``\\right`` 指定的可伸缩定界符包裹。"""

    left: str = "none"
    body: Group = field(default_factory=Group)
    right: str = "none"


@dataclass
class Matrix(Node):
    """矩阵 / 方程组 / 对齐等「环境」，行为单元格网格。"""

    env: str = "matrix"
    rows: list[list[Group]] = field(default_factory=list)
    col_spec: str = ""


def as_group(nodes: list[Node] | None) -> Group:
    """把节点列表包装为 :class:`Group`。"""
    return Group(items=list(nodes or []))


# --------------------------------------------------------------------------- #
# 环境查表
# --------------------------------------------------------------------------- #
#: 环境名 -> (左定界符, 右定界符)，``"none"`` 表示无
ENV_DELIMITERS: dict[str, tuple[str, str]] = {
    "matrix": ("none", "none"),
    "matrix*": ("none", "none"),
    "smallmatrix": ("none", "none"),
    "pmatrix": ("(", ")"),
    "pmatrix*": ("(", ")"),
    "bmatrix": ("[", "]"),
    "bmatrix*": ("[", "]"),
    "Bmatrix": (r"\{", r"\}"),
    "Bmatrix*": (r"\{", r"\}"),
    "vmatrix": ("|", "|"),
    "vmatrix*": ("|", "|"),
    "Vmatrix": (r"\|", r"\|"),
    "Vmatrix*": (r"\|", r"\|"),
    "cases": (r"\{", "none"),
    "dcases": (r"\{", "none"),
    "array": ("none", "none"),
    "tabular": ("none", "none"),
    "aligned": ("none", "none"),
    "align": ("none", "none"),
    "align*": ("none", "none"),
    "split": ("none", "none"),
    "gathered": ("none", "none"),
    "gather": ("none", "none"),
}

#: 环境名 -> 单元格对齐方式
#: ``center`` 居中；``left`` 左对齐；``align`` 偶数列右对齐、奇数列左对齐；``spec`` 用列格式串
ENV_ALIGN: dict[str, str] = {
    "cases": "left",
    "dcases": "left",
    "array": "spec",
    "tabular": "spec",
    "aligned": "align",
    "align": "align",
    "align*": "align",
    "split": "align",
    "gathered": "center",
    "gather": "center",
}

#: 环境名 -> 单元格字号缩放
ENV_SCALE: dict[str, float] = {
    "smallmatrix": 0.78,
    "cases": 1.0,
}

#: 支持的环境集合
MATRIX_ENVS: frozenset[str] = frozenset(ENV_DELIMITERS)

#: 定界符记号 -> 用于取字形的 Unicode 字符（``None`` 表示空定界符）
DELIM_CHARS: dict[str, str | None] = {
    "(": "(",
    ")": ")",
    "[": "[",
    "]": "]",
    "|": "|",
    ".": None,
    "<": "\u27e8",
    ">": "\u27e9",
    r"\{": "{",
    r"\}": "}",
    r"\|": "\u2016",
    r"\vert": "|",
    r"\Vert": "\u2016",
    r"\lvert": "|",
    r"\rvert": "|",
    r"\lVert": "\u2016",
    r"\rVert": "\u2016",
    r"\langle": "\u27e8",
    r"\rangle": "\u27e9",
    r"\lceil": "\u2308",
    r"\rceil": "\u2309",
    r"\lfloor": "\u230a",
    r"\rfloor": "\u230b",
    r"\lgroup": "(",
    r"\rgroup": ")",
}

#: 常见 LaTeX 命令 -> Unicode，用于 mathtext 解析失败时的降级显示
UNICODE_FALLBACK: dict[str, str] = {
    r"\alpha": "\u03b1", r"\beta": "\u03b2", r"\gamma": "\u03b3", r"\delta": "\u03b4",
    r"\epsilon": "\u03b5", r"\varepsilon": "\u03b5", r"\zeta": "\u03b6", r"\eta": "\u03b7",
    r"\theta": "\u03b8", r"\vartheta": "\u03b8", r"\iota": "\u03b9", r"\kappa": "\u03ba",
    r"\lambda": "\u03bb", r"\mu": "\u03bc", r"\nu": "\u03bd", r"\xi": "\u03be",
    r"\pi": "\u03c0", r"\rho": "\u03c1", r"\sigma": "\u03c3", r"\tau": "\u03c4",
    r"\upsilon": "\u03c5", r"\phi": "\u03c6", r"\varphi": "\u03c6", r"\chi": "\u03c7",
    r"\psi": "\u03c8", r"\omega": "\u03c9",
    r"\Gamma": "\u0393", r"\Delta": "\u0394", r"\Theta": "\u0398", r"\Lambda": "\u039b",
    r"\Xi": "\u039e", r"\Pi": "\u03a0", r"\Sigma": "\u03a3", r"\Phi": "\u03a6",
    r"\Psi": "\u03a8", r"\Omega": "\u03a9",
    r"\times": "\u00d7", r"\cdot": "\u00b7", r"\ast": "\u2217", r"\star": "\u22c6",
    r"\pm": "\u00b1", r"\mp": "\u2213", r"\div": "\u00f7", r"\otimes": "\u2297",
    r"\oplus": "\u2295", r"\odot": "\u2299",
    r"\leq": "\u2264", r"\le": "\u2264", r"\geq": "\u2265", r"\ge": "\u2265",
    r"\neq": "\u2260", r"\ne": "\u2260", r"\approx": "\u2248", r"\equiv": "\u2261",
    r"\sim": "\u223c", r"\propto": "\u221d", r"\ll": "\u226a", r"\gg": "\u226b",
    r"\infty": "\u221e", r"\partial": "\u2202", r"\nabla": "\u2207",
    r"\to": "\u2192", r"\rightarrow": "\u2192", r"\leftarrow": "\u2190",
    r"\Rightarrow": "\u21d2", r"\Leftrightarrow": "\u21d4", r"\mapsto": "\u21a6",
    r"\in": "\u2208", r"\notin": "\u2209", r"\subset": "\u2282", r"\subseteq": "\u2286",
    r"\supset": "\u2283", r"\cup": "\u222a", r"\cap": "\u2229", r"\emptyset": "\u2205",
    r"\forall": "\u2200", r"\exists": "\u2203", r"\neg": "\u00ac",
    r"\sum": "\u2211", r"\prod": "\u220f", r"\int": "\u222b", r"\oint": "\u222e",
    r"\sqrt": "\u221a", r"\angle": "\u2220", r"\perp": "\u22a5", r"\parallel": "\u2225",
    r"\cdots": "\u22ef", r"\ldots": "\u2026", r"\dots": "\u2026",
    r"\vdots": "\u22ee", r"\ddots": "\u22f1",
    r"\quad": "\u2003", r"\qquad": "\u2003\u2003", r"\,": "\u2009", r"\;": "\u2009",
    r"\:": "\u2009", r"\!": "",
    r"\langle": "\u27e8", r"\rangle": "\u27e9", r"\|": "\u2016", r"\vert": "|",
    r"\{": "{", r"\}": "}", r"\_": "_", r"\%": "%", r"\&": "&", r"\#": "#",
    r"\left": "", r"\right": "", r"\bigl": "", r"\bigr": "", r"\Bigl": "", r"\Bigr": "",
    r"\text": "", r"\operatorname": "", r"\mathrm": "", r"\mathbf": "", r"\mathit": "",
    r"\det": "det", r"\tr": "tr", r"\rank": "rank", r"\diag": "diag",
}
