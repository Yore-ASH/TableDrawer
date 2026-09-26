"""matplotlib 全局初始化：字体回退链、数学字体、输出后端。

字体策略
--------
只把**系统里真实存在**的字体放进回退链（matplotlib 对每个找不到的字体都会
打印 ``findfont: Font family ... not found`` 警告），并且让「拉丁字体 → 中文字体
→ 兜底字体」构成一条链，这样：

* 英文与数字使用指定的西文字体（默认 Times New Roman）
* 中文自动回退到宋体
* 中文字体里缺失的 ``⁻ ᵢ ᵀ ⱼ`` 等上下标再回退到 DejaVu Sans，不会出现豆腐块

:data:`FONT_PRESETS` 提供可切换的预设；界面层通过 :func:`apply_font_preset`
切换，切换后需要调用 :func:`core.latex.clear_cache` 丢弃旧的度量缓存。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Iterable, Sequence

import matplotlib
from matplotlib import font_manager

__all__ = [
    "LATIN_CANDIDATES",
    "CJK_CANDIDATES",
    "MONO_CANDIDATES",
    "MATH_FONTSETS",
    "FONT_PRESETS",
    "DEFAULT_FONT_PRESET",
    "available_fonts",
    "setup_matplotlib",
    "apply_font_preset",
    "font_config",
    "active_families",
    "cjk_font_family",
    "latin_font_family",
    "mono_font_family",
    "math_font_family",
]

#: 西文候选（按优先级）
LATIN_CANDIDATES: tuple[str, ...] = (
    "Times New Roman",
    "Segoe UI",
    "Calibri",
    "Arial",
    "Helvetica",
    "DejaVu Sans",
)

#: 中文候选（按优先级）
CJK_CANDIDATES: tuple[str, ...] = (
    "SimSun",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimHei",
    "NSimSun",
    "KaiTi",
    "DengXian",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Arial Unicode MS",
)

#: 等宽候选（代码区域）
MONO_CANDIDATES: tuple[str, ...] = (
    "Consolas",
    "Cascadia Mono",
    "Cascadia Code",
    "JetBrains Mono",
    "Fira Code",
    "DejaVu Sans Mono",
    "Courier New",
)

#: 数学字体集 -> 中文说明
MATH_FONTSETS: dict[str, str] = {
    "cm": "Computer Modern（最像 LaTeX）",
    "stix": "STIX（与 Times 搭配）",
    "stixsans": "STIX Sans",
    "dejavusans": "DejaVu Sans",
    "dejavuserif": "DejaVu Serif",
}

#: 字体预设：``generic`` 决定挂到 matplotlib 的 serif 还是 sans-serif 列表
FONT_PRESETS: dict[str, dict[str, Any]] = {
    "paper": {
        "name": "论文风格（Times New Roman + 宋体）",
        "latin": ("Times New Roman", "Arial", "DejaVu Sans"),
        "cjk": ("SimSun", "Microsoft YaHei", "SimHei", "DejaVu Sans"),
        "mono": ("Consolas", "Cascadia Mono", "Courier New", "DejaVu Sans Mono"),
        "generic": "serif",
        "mathtext": "stix",
    },
    "modern": {
        "name": "现代风格（Segoe UI + 微软雅黑）",
        "latin": ("Segoe UI", "Calibri", "Arial", "DejaVu Sans"),
        "cjk": ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "SimSun"),
        "mono": ("Consolas", "Cascadia Mono", "Courier New", "DejaVu Sans Mono"),
        "generic": "sans-serif",
        "mathtext": "cm",
    },
    "songti": {
        "name": "全宋体（Times New Roman + 宋体 + 楷体）",
        "latin": ("Times New Roman", "DejaVu Serif"),
        "cjk": ("SimSun", "NSimSun", "KaiTi", "SimHei"),
        "mono": ("Consolas", "Courier New", "DejaVu Sans Mono"),
        "generic": "serif",
        "mathtext": "stix",
    },
}

DEFAULT_FONT_PRESET = "paper"

#: matplotlib 的通用字体族名（不参与「具体字体链」的判断）
_GENERIC_FAMILIES = frozenset(
    {"serif", "sans-serif", "sans", "monospace", "mono", "cursive", "fantasy"}
)

_initialised = False
_config: dict[str, Any] = {
    "preset": DEFAULT_FONT_PRESET,
    "latin": [],
    "cjk": [],
    "mono": [],
    "generic": "serif",
    "mathtext": "stix",
}


# --------------------------------------------------------------------------- #
# 字体探测
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def available_fonts() -> frozenset[str]:
    """当前 matplotlib 能识别的所有字体族名。"""
    return frozenset(f.name for f in font_manager.fontManager.ttflist)


def _resolve(candidates: Iterable[str], *, fallback: str | None = None) -> list[str]:
    """按优先级筛出系统里真实存在的字体名。"""
    names = available_fonts()
    picked = [name for name in candidates if name in names]
    if not picked and fallback and fallback in names:
        picked = [fallback]
    return picked


@lru_cache(maxsize=1)
def cjk_font_family() -> str:
    """首个可用的中文字体族名（用于需要单一字体名的场合）。"""
    for candidate in CJK_CANDIDATES:
        if candidate in available_fonts():
            return candidate
    return "DejaVu Sans"


@lru_cache(maxsize=1)
def latin_font_family() -> str:
    """首个可用的西文字体族名。"""
    for candidate in LATIN_CANDIDATES:
        if candidate in available_fonts():
            return candidate
    return "DejaVu Sans"


@lru_cache(maxsize=1)
def mono_font_family() -> str:
    """首个可用的等宽字体族名。"""
    for candidate in MONO_CANDIDATES:
        if candidate in available_fonts():
            return candidate
    return "DejaVu Sans Mono"


@lru_cache(maxsize=1)
def math_font_family() -> str:
    """适合渲染纯数学文本的字体族名（数学字体集不可用时的兜底）。"""
    for candidate in ("STIXGeneral", "Cambria Math", "DejaVu Sans"):
        if candidate in available_fonts():
            return candidate
    return "DejaVu Sans"


def font_config() -> dict[str, Any]:
    """返回当前生效的字体配置副本。"""
    return {
        "preset": _config["preset"],
        "latin": list(_config["latin"]),
        "cjk": list(_config["cjk"]),
        "mono": list(_config["mono"]),
        "generic": _config["generic"],
        "mathtext": _config["mathtext"],
    }


def active_families() -> list[str]:
    """当前 matplotlib 生效的完整字体回退链（西文 → 中文 → 兜底）。"""
    family = matplotlib.rcParams.get("font.family", [])
    if isinstance(family, str):
        family = [family]
    concrete = [str(name) for name in family if str(name) not in _GENERIC_FAMILIES]
    if concrete:
        return concrete
    generic = _config.get("generic", "serif")
    try:
        return list(matplotlib.rcParams.get(f"font.{generic}", []))
    except Exception:
        return list(_config["latin"]) + list(_config["cjk"])


# --------------------------------------------------------------------------- #
# 应用配置
# --------------------------------------------------------------------------- #
def apply_font_preset(preset: str = DEFAULT_FONT_PRESET, *, clear_latex_cache: bool = True) -> dict[str, Any]:
    """切换到指定字体预设并写入 matplotlib rcParams。

    返回解析后的配置（``latin`` / ``cjk`` / ``mono`` 均为**实际存在**的字体名）。
    """
    global _initialised
    spec = FONT_PRESETS.get(preset) or FONT_PRESETS[DEFAULT_FONT_PRESET]
    latin = _resolve(spec["latin"], fallback="DejaVu Sans")
    cjk = _resolve(spec["cjk"], fallback="DejaVu Sans")
    mono = _resolve(spec["mono"], fallback="DejaVu Sans Mono")
    generic = spec.get("generic", "serif")
    mathtext_fontset = spec.get("mathtext", "stix")

    # 回退链：西文 → 中文 →（数字/上下标兜底）
    chain: list[str] = []
    for name in (*latin, *cjk, "DejaVu Sans"):
        if name in available_fonts() and name not in chain:
            chain.append(name)

    _config.update(
        {
            "preset": preset if preset in FONT_PRESETS else DEFAULT_FONT_PRESET,
            "latin": latin,
            "cjk": cjk,
            "mono": mono,
            "generic": generic,
            "mathtext": mathtext_fontset,
        }
    )

    #: matplotlib 的多字体回退只在 ``font.family`` **本身**是具体字体名列表时生效：
    #: 若写成 ``font.family = ["serif"]``，回退链里只会剩下 ``font.serif`` 解析出的
    #: **第一个**字体，中文就会变成豆腐块。因此这里把完整链条直接写进 font.family。
    rc: dict[str, Any] = {
        "font.family": chain,
        "font.serif": chain,
        "font.sans-serif": chain,
        "font.monospace": list(dict.fromkeys([*mono, *cjk, "DejaVu Sans Mono"])),
        "axes.unicode_minus": False,
        "mathtext.fontset": mathtext_fontset,
        "mathtext.default": "it",
        "axes.formatter.use_mathtext": True,
    }
    matplotlib.rcParams.update(rc)

    _initialised = True
    if clear_latex_cache:
        try:
            from .latex.measure import clear_cache

            clear_cache()
        except Exception:  # pragma: no cover - 循环导入或字体缺失兜底
            pass
    return font_config()


def setup_matplotlib(
    *,
    preset: str = DEFAULT_FONT_PRESET,
    mathtext_fontset: str | None = None,
    font_size: float = 11.0,
    dpi: int = 110,
    interactive: bool = True,
    clear_latex_cache: bool = False,
) -> str:
    """初始化 matplotlib 全局配置，返回选中的中文字体族名。"""
    if not interactive:
        matplotlib.use("Agg", force=True)
    elif os.environ.get("TABLEDRAWER_BACKEND"):
        matplotlib.use(os.environ["TABLEDRAWER_BACKEND"], force=True)

    config = apply_font_preset(preset, clear_latex_cache=clear_latex_cache)
    if mathtext_fontset:
        matplotlib.rcParams["mathtext.fontset"] = mathtext_fontset
        config["mathtext"] = mathtext_fontset
        _config["mathtext"] = mathtext_fontset

    matplotlib.rcParams.update(
        {
            "figure.dpi": dpi,
            "figure.max_open_warning": 0,
            "savefig.bbox": "standard",
            "axes.titlesize": font_size + 1.5,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1.5,
            "ytick.labelsize": font_size - 1.5,
            "legend.fontsize": font_size - 1.5,
            "axes.grid": False,
            "axes.formatter.limits": (-4, 5),
            "pdf.fonttype": 42,       # 导出 PDF 时嵌入 TrueType
            "ps.fonttype": 42,
            "svg.fonttype": "path",   # SVG 用路径，避免缺字体
        }
    )
    return cjk_font_family()


def ensure_setup() -> str:
    """幂等地完成初始化。"""
    if not _initialised:
        return setup_matplotlib()
    return cjk_font_family()
