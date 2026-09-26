"""界面主题：配色板、字体与 Qt 样式表。

* :func:`palette` 返回一套命名颜色，界面各处（含 matplotlib 画布底色、
  LaTeX 结果视图背景）统一从这里取色，保证换肤时全局一致。
* :func:`ui_fonts` 把 :data:`core.mplsetup.FONT_PRESETS` 解析成
  「正文族列表 / 等宽族列表」两个 CSS 字体栈：正文默认
  **Times New Roman → 宋体 → 兜底**（西文走 Times、中文走宋体），
  代码区域使用 **Consolas**（VS Code 在 Windows 上的默认等宽字体）。
* :func:`stylesheet` 根据配色板与字体生成 QSS。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from core.mplsetup import DEFAULT_FONT_PRESET, FONT_PRESETS, font_config

__all__ = [
    "DARK",
    "LIGHT",
    "PALETTES",
    "FONT_PRESETS",
    "DEFAULT_FONT_PRESET",
    "palette",
    "ui_fonts",
    "stylesheet",
    "font_css_stack",
]

#: 深色配色
DARK: dict[str, str] = {
    "name": "dark",
    "bg": "#1b1d21",
    "surface": "#24262b",
    "surface_alt": "#2c2f35",
    "surface_hi": "#33373e",
    "border": "#3a3e46",
    "text": "#e6e8ec",
    "text_muted": "#9aa1ab",
    "text_disabled": "#5f6670",
    "accent": "#4c8dff",
    "accent_hover": "#6aa0ff",
    "accent_pressed": "#3a7ae8",
    "accent_text": "#ffffff",
    "danger": "#f0616d",
    "success": "#3fb950",
    "warning": "#d29922",
    "canvas_bg": "#191b1f",
    "plot_grid": "#3d424b",
    "plot_axis": "#6b7280",
    "plot_text": "#e6e8ec",
    "latex_bg": "#202226",
    "latex_text": "#e6e8ec",
    "latex_title": "#6aa0ff",
    "latex_note": "#9aa1ab",
    "latex_rule": "#3a3e46",
    "selection": "#2f4a7a",
}

#: 浅色配色
LIGHT: dict[str, str] = {
    "name": "light",
    "bg": "#f2f4f7",
    "surface": "#ffffff",
    "surface_alt": "#f6f8fa",
    "surface_hi": "#eceff3",
    "border": "#d6dae0",
    "text": "#1f2328",
    "text_muted": "#656d76",
    "text_disabled": "#a0a6ad",
    "accent": "#2563eb",
    "accent_hover": "#3b82f6",
    "accent_pressed": "#1d4ed8",
    "accent_text": "#ffffff",
    "danger": "#cf222e",
    "success": "#1a7f37",
    "warning": "#9a6700",
    "canvas_bg": "#ffffff",
    "plot_grid": "#ccd2da",
    "plot_axis": "#57606a",
    "plot_text": "#1f2328",
    "latex_bg": "#ffffff",
    "latex_text": "#1f2328",
    "latex_title": "#1d4ed8",
    "latex_note": "#656d76",
    "latex_rule": "#d6dae0",
    "selection": "#cfe0ff",
}

PALETTES: dict[str, dict[str, str]] = {"dark": DARK, "light": LIGHT}


def palette(name: str | dict[str, str]) -> dict[str, str]:
    """按名称取配色板（未知名称回退到浅色）。"""
    if isinstance(name, dict):
        return name
    return PALETTES.get(str(name).lower(), LIGHT)


def font_css_stack(names: Iterable[str], extra: Iterable[str] = ()) -> str:
    """把字体名列表拼成 CSS/QSS 的 ``font-family`` 值。

    含空格的字体名要加引号；末尾会补一个通用族（``serif`` / ``monospace``）兜底。
    """
    parts: list[str] = []
    for name in (*names, *extra):
        name = str(name or "").strip()
        if not name:
            continue
        parts.append(f'"{name}"' if " " in name else name)
    return ", ".join(dict.fromkeys(parts))


def ui_fonts(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """把字体配置解析成界面可直接使用的字体栈。

    返回 ``{"body": CSS 族列表, "mono": CSS 族列表, "latin": …, "cjk": …, "mono_name": …, "preset": …}``。
    """
    cfg = dict(config or font_config())
    latin = list(cfg.get("latin") or ["Times New Roman", "DejaVu Sans"])
    cjk = list(cfg.get("cjk") or ["SimSun", "DejaVu Sans"])
    mono = list(cfg.get("mono") or ["Consolas", "DejaVu Sans Mono"])
    generic = "serif" if str(cfg.get("generic", "serif")) == "serif" else "sans-serif"
    return {
        "preset": str(cfg.get("preset", DEFAULT_FONT_PRESET)),
        "latin": latin[0] if latin else "DejaVu Sans",
        "cjk": cjk[0] if cjk else "DejaVu Sans",
        "mono_name": mono[0] if mono else "DejaVu Sans Mono",
        "body": font_css_stack([*latin, *cjk], extra=[generic]),
        "heading": font_css_stack([*cjk, *latin], extra=[generic]),
        "mono": font_css_stack(mono, extra=[*cjk[:1], "monospace"]),
    }


def stylesheet(p: dict[str, str], fonts: Mapping[str, Any] | None = None) -> str:
    """根据配色板与字体生成 QSS。"""
    f = dict(fonts or ui_fonts())
    body = f["body"]
    mono = f["mono"]
    return f"""
* {{
    font-family: {body};
    font-size: 13px;
    outline: none;
}}
QWidget {{
    background-color: {p['bg']};
    color: {p['text']};
}}
QMainWindow, QDialog {{ background-color: {p['bg']}; }}

/* 代码 / 数据粘贴区域使用等宽字体（VS Code 同款：Consolas） */
QPlainTextEdit, QTextEdit, QLineEdit[code="true"], QLabel[code="true"] {{
    font-family: {mono};
}}
QTableWidget[code="true"] {{ font-family: {mono}; }}

QMenuBar {{ background-color: {p['bg']}; border-bottom: 1px solid {p['border']}; }}
QMenuBar::item {{ padding: 5px 11px; background: transparent; border-radius: 5px; }}
QMenuBar::item:selected {{ background-color: {p['surface_hi']}; }}
QMenu {{ background-color: {p['surface']}; border: 1px solid {p['border']}; border-radius: 8px; padding: 5px; }}
QMenu::item {{ padding: 6px 26px 6px 22px; border-radius: 5px; }}
QMenu::item:selected {{ background-color: {p['accent']}; color: {p['accent_text']}; }}
QMenu::separator {{ height: 1px; background: {p['border']}; margin: 5px 8px; }}

QToolBar {{ background-color: {p['bg']}; border-bottom: 1px solid {p['border']}; spacing: 4px; padding: 4px; }}
QToolBar::separator {{ width: 1px; background: {p['border']}; margin: 4px 6px; }}
QToolButton {{ padding: 5px 9px; border-radius: 6px; background: transparent; }}
QToolButton:hover {{ background-color: {p['surface_hi']}; }}
QToolButton:pressed, QToolButton:checked {{ background-color: {p['accent']}; color: {p['accent_text']}; }}

QStatusBar {{ background-color: {p['surface']}; border-top: 1px solid {p['border']}; color: {p['text_muted']}; }}
QStatusBar::item {{ border: none; }}

QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 8px; background: {p['surface']}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {p['text_muted']};
    padding: 7px 16px; margin-right: 3px;
    border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{ background: {p['surface']}; color: {p['text']}; border-color: {p['border']}; border-bottom-color: {p['surface']}; }}
QTabBar::tab:hover:!selected {{ background: {p['surface_hi']}; color: {p['text']}; }}

QGroupBox {{
    border: 1px solid {p['border']}; border-radius: 8px;
    margin-top: 11px; padding: 10px 8px 8px 8px; background: {p['surface']};
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 10px; padding: 1px 6px; color: {p['accent']}; font-weight: 600;
}}

QPushButton {{
    background-color: {p['surface_alt']}; color: {p['text']};
    border: 1px solid {p['border']}; border-radius: 7px; padding: 5px 12px; min-height: 20px;
}}
QPushButton:hover {{ background-color: {p['surface_hi']}; border-color: {p['accent']}; }}
QPushButton:pressed {{ background-color: {p['accent']}; color: {p['accent_text']}; }}
QPushButton:disabled {{ color: {p['text_disabled']}; border-color: {p['border']}; background: {p['surface']}; }}
QPushButton[accent="true"] {{ background-color: {p['accent']}; color: {p['accent_text']}; border-color: {p['accent']}; }}
QPushButton[accent="true"]:hover {{ background-color: {p['accent_hover']}; }}
QPushButton[danger="true"]:hover {{ background-color: {p['danger']}; color: #ffffff; border-color: {p['danger']}; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {p['surface_alt']}; color: {p['text']};
    border: 1px solid {p['border']}; border-radius: 6px; padding: 4px 7px;
    selection-background-color: {p['accent']}; selection-color: {p['accent_text']};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {p['accent']};
}}
QLineEdit:read-only {{ color: {p['text_muted']}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {p['surface']}; border: 1px solid {p['border']};
    selection-background-color: {p['accent']}; selection-color: {p['accent_text']}; padding: 4px;
}}

QTableWidget, QTableView, QListWidget, QTreeWidget {{
    background-color: {p['surface']}; alternate-background-color: {p['surface_alt']};
    border: 1px solid {p['border']}; border-radius: 8px;
    gridline-color: {p['border']};
    selection-background-color: {p['selection']}; selection-color: {p['text']};
}}
QTableWidget::item, QListWidget::item {{ padding: 3px 5px; }}
QListWidget::item:selected {{ background-color: {p['accent']}; color: {p['accent_text']}; }}
QHeaderView::section {{
    background-color: {p['surface_hi']}; color: {p['text_muted']};
    padding: 5px 7px; border: none; border-right: 1px solid {p['border']}; border-bottom: 1px solid {p['border']};
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: {p['surface_hi']}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p['surface_hi']}; border-radius: 5px; min-height: 26px; }}
QScrollBar::handle:vertical:hover {{ background: {p['accent']}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p['surface_hi']}; border-radius: 5px; min-width: 26px; }}
QScrollBar::handle:horizontal:hover {{ background: {p['accent']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {p['border']}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}

QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
QDockWidget::title {{ background: {p['surface_hi']}; padding: 6px 9px; border-bottom: 1px solid {p['border']}; }}

QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {p['border']};
    border-radius: 4px; background: {p['surface_alt']};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {p['accent']}; border-color: {p['accent']};
}}

QSlider::groove:horizontal {{ height: 5px; background: {p['surface_hi']}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {p['accent']}; width: 13px; margin: -5px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {p['accent']}; border-radius: 3px; }}

QProgressBar {{ border: 1px solid {p['border']}; border-radius: 6px; text-align: center; background: {p['surface_alt']}; }}
QProgressBar::chunk {{ background: {p['accent']}; border-radius: 5px; }}

QToolTip {{
    background-color: {p['surface']}; color: {p['text']};
    border: 1px solid {p['accent']}; border-radius: 6px; padding: 5px 8px;
}}
QLabel[heading="true"] {{ font-size: 16px; font-weight: 700; color: {p['text']}; }}
QLabel[muted="true"] {{ color: {p['text_muted']}; }}
QLabel[accent="true"] {{ color: {p['accent']}; font-weight: 600; }}
QFrame[role="card"] {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 8px; }}
"""
