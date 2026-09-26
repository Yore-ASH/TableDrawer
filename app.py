#!/usr/bin/env python3
"""TableDrawer 启动入口。

用法::

    python app.py                      # 正常启动图形界面
    python app.py data.csv other.csv   # 启动并自动导入指定的 CSV
    python app.py --theme light        # 指定主题（dark / light）
    python app.py --selftest           # 离屏自检（构建窗口、画一张图后退出）
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _pop_option(argv: list[str], name: str, default: str = "") -> str:
    """从参数表里取出 ``--name value`` 并返回其值。"""
    if name in argv:
        index = argv.index(name)
        if index + 1 < len(argv):
            value = argv[index + 1]
            del argv[index : index + 2]
            return value
        del argv[index]
    return default


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    selftest = "--selftest" in argv
    if selftest:
        argv.remove("--selftest")
    theme = _pop_option(argv, "--theme", "dark")

    if selftest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from core.mplsetup import setup_matplotlib

    family = setup_matplotlib(mathtext_fontset="cm")

    from PySide6.QtWidgets import QApplication

    app = QApplication(argv)
    app.setApplicationName("TableDrawer")
    app.setApplicationDisplayName("TableDrawer")
    app.setOrganizationName("TableDrawer")
    app.setStyle("Fusion")

    from ui.main_window import MainWindow

    window = MainWindow(theme=theme)
    window.show()

    csv_paths = [p for p in argv[1:] if p.lower().endswith((".csv", ".tsv", ".txt", ".dat"))]
    if csv_paths:
        window._import_paths(csv_paths)

    if selftest:
        return _run_selftest(app, window, family)
    return app.exec()


def _run_selftest(app, window, family: str) -> int:
    """离屏自检：构建窗口 -> 载入示例 -> 绘图 -> 导出 SVG -> 矩阵计算 -> 干净退出。"""
    from PySide6.QtCore import QTimer

    problems: list[str] = []

    def check(label: str, condition: bool, extra: str = "") -> None:
        print(f"{'OK  ' if condition else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")
        if not condition:
            problems.append(label)

    window.load_samples()
    check("载入示例数据", len(window.store) > 0, f"{len(window.store)} 个数据集")

    names = window.store.names()
    window.tabs.setCurrentIndex(1)
    window.plot_tab.new_default_plot(names[0])
    window.plot_tab.render()
    figure = window.plot_tab.canvas.figure
    check("二维绘图", len(figure.axes) > 0)

    from core.plotting import figure_to_svg

    svg = figure_to_svg(figure)
    check("导出 SVG", "<svg" in svg and "</svg>" in svg, f"{len(svg)} 字节")

    window.plot_tab.projection_combo.setCurrentIndex(1)
    window.plot_tab.new_default_plot(names[0])
    window.plot_tab.render()
    check("三维绘图", len(window.plot_tab.canvas.figure.axes) > 0)

    window.tabs.setCurrentIndex(2)
    matrix_tab = window.matrix_tab
    matrix_tab.editor_a.set_matrix([[1, 2], [3, 4]])
    matrix_tab.run_operation("determinant")
    matrix_tab.run_operation("inverse")
    matrix_tab.run_operation("eigen")
    check("矩阵计算", len(matrix_tab._sheet.items) > 3, f"{len(matrix_tab._sheet.items)} 条结果")
    check("LaTeX 渲染", len(matrix_tab.view.to_svg()) > 400)

    window.tabs.setCurrentIndex(0)
    from ui.data_tab import build_summary_sheet

    summary = build_summary_sheet(window.store.require(names[0]))
    check("统计摘要 LaTeX", any(item.latex for item in summary.items), f"{len(summary.items)} 条")
    window.data_tab.table.setCurrentCell(0, 0)
    check("数据页就绪", window.data_tab.table.columnCount() > 0)

    window.apply_theme("light")
    check("浅色主题切换", window._theme_name == "light")

    def finish() -> None:
        window.close()
        app.quit()

    QTimer.singleShot(0, finish)
    app.exec()          # 真正跑一遍事件循环，顺便验证关闭流程不会崩

    print()
    print(f"中文字体：{family}")
    print(f"数据集  ：{', '.join(names)}")
    if problems:
        print(f"自检失败：{', '.join(problems)}")
        return 1
    print("自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
