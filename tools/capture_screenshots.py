"""离屏截取界面各页，生成 ``docs/screenshots`` 里的配图。

用法::

    .venv\\Scripts\\python.exe tools\\capture_screenshots.py

要点：使用**真实平台插件 + WA_DontShowOnScreen**——窗口不会出现在屏幕上，
但 Qt 能访问系统字体库。若改用 ``QT_QPA_PLATFORM=offscreen``，Qt 的字体库是空的，
中文会全部显示成方框，截出来的图没有参考价值。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_API", "pyside6")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from core.mplsetup import setup_matplotlib  # noqa: E402

setup_matplotlib()

app = QApplication([])
app.setStyle("Fusion")

from ui.main_window import MainWindow  # noqa: E402

OUT = os.path.join(ROOT, "docs", "screenshots")


def settle(ms: int = 250) -> None:
    """跑一段事件循环，让延迟渲染（LaTeX 位图队列）完成。"""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def shot(widget, name: str) -> None:
    app.processEvents()
    pixmap = widget.grab()
    path = os.path.join(OUT, name)
    pixmap.save(path)
    print(f"{name:22s} {pixmap.width()}x{pixmap.height()}  {os.path.getsize(path)} bytes")


def main() -> int:
    print("platform:", app.platformName(), "| 字体族:", len(QFontDatabase.families()))
    os.makedirs(OUT, exist_ok=True)

    win = MainWindow(theme="dark")
    win.resize(1560, 980)
    win.setAttribute(Qt.WA_DontShowOnScreen, True)   # 不映射到屏幕，但保留真实字体库
    win.show()
    app.processEvents()

    win.load_samples()
    names = win.store.names()

    # ---- 数据页 ---------------------------------------------------------- #
    win.tabs.setCurrentIndex(0)
    win.data_tab.select_dataset(names[3])            # 学生成绩
    app.processEvents()
    shot(win, "data-tab.png")

    # ---- 绘图页 2D ------------------------------------------------------- #
    win.tabs.setCurrentIndex(1)
    win.plot_tab.projection_combo.setCurrentIndex(0)
    win.plot_tab.new_default_plot(names[2])          # 传感器时序
    win.plot_tab.render()
    settle(400)
    shot(win, "plot-2d.png")

    # ---- 绘图页：中文 + 公式混排标题 -------------------------------------- #
    win.plot_tab.title_edit.setText(r"温度与湿度 $y=e^{-t/6}\sin t$ 对比")
    win.plot_tab.xlabel_edit.setText(r"时间 $t$ / h")
    win.plot_tab.ylabel_edit.setText(r"数值 $y$")
    win.plot_tab.render()
    settle(400)
    shot(win, "plot-mixed-labels.png")

    # ---- 绘图页 3D ------------------------------------------------------- #
    win.plot_tab.projection_combo.setCurrentIndex(1)
    win.plot_tab.new_default_plot(names[0])          # 三维曲面
    win.plot_tab.render()
    settle(400)
    shot(win, "plot-3d.png")

    # ---- 矩阵页（等 LaTeX 按钮渲染完） ------------------------------------ #
    win.tabs.setCurrentIndex(2)
    matrix_tab = win.matrix_tab
    matrix_tab.editor_a.set_matrix([[4, -2, 1], [3, 6, -1], [2, 1, 8]])
    matrix_tab.run_common()
    for _ in range(40):                              # 队列里是一个一个渲染的
        app.processEvents()
        settle(120)
        if __import__("ui.widgets", fromlist=["latex_pixmaps"]).latex_pixmaps().pending() == 0:
            break
    settle(600)
    shot(win, "matrix-tab.png")

    # ---- 放大编辑矩阵窗口 ------------------------------------------------- #
    from ui.widgets import MatrixEditorDialog

    matrix_tab.editor_a.set_cells([["1", "2", "3", "4"], ["5", "6", "", "8"], ["9", "10", "11", ""]])
    dialog = MatrixEditorDialog(win, editor=matrix_tab.editor_a, label="A")
    dialog.resize(1180, 760)
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    dialog.show()
    settle(700)
    shot(dialog, "matrix-editor.png")
    dialog.close()

    # ---- manim 动画对话框 ------------------------------------------------- #
    from ui.dialogs import ManimDialog

    matrix_tab.editor_a.set_matrix([[4, -2, 1], [3, 6, -1], [2, 1, 8]])
    manim_dialog = ManimDialog(
        win,
        matrix=matrix_tab.editor_a.matrix(),
        other=matrix_tab.editor_b.matrix(),
        label="A",
        palette=win._palette,
    )
    manim_dialog.scene_combo.setCurrentIndex(manim_dialog.scene_combo.findData("elimination"))
    manim_dialog.resize(1180, 800)
    manim_dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    manim_dialog.show()
    settle(900)
    shot(manim_dialog, "manim-storyboard.png")
    manim_dialog.tabs.setCurrentIndex(1)
    settle(300)
    shot(manim_dialog, "manim-source.png")
    manim_dialog.close()

    # ---- 浅色主题 -------------------------------------------------------- #
    win.apply_theme("light")
    win.tabs.setCurrentIndex(1)
    win.plot_tab.projection_combo.setCurrentIndex(0)
    win.plot_tab.new_default_plot(names[1])          # 月度销售
    win.plot_tab.render()
    settle(400)
    shot(win, "plot-light.png")

    win.tabs.setCurrentIndex(2)
    matrix_tab.view._render()
    for _ in range(40):
        app.processEvents()
        settle(120)
        if __import__("ui.widgets", fromlist=["latex_pixmaps"]).latex_pixmaps().pending() == 0:
            break
    settle(600)
    shot(win, "matrix-tab-light.png")

    # ---- 对话框 ---------------------------------------------------------- #
    from ui.dialogs import CsvImportDialog, ExportSourceDialog, ManualDataDialog

    win.apply_theme("dark")
    for dialog, name, size in (
        (CsvImportDialog(win, path=win.store.require(names[3]).path, palette=win._palette), "csv-import.png", (940, 620)),
        (ManualDataDialog(win), "manual-data.png", (880, 600)),
    ):
        dialog.resize(*size)
        dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
        dialog.show()
        settle(350 if name.startswith("csv") else 200)
        shot(dialog, name)
        dialog.close()

    win.tabs.setCurrentIndex(1)
    source_dialog = ExportSourceDialog(
        win, spec=win.plot_tab._collect_spec(), datasets=win.store.frames()
    )
    source_dialog.resize(960, 680)
    source_dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    source_dialog.show()
    settle(500)
    shot(source_dialog, "export-source.png")
    source_dialog.close()

    print("done ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
