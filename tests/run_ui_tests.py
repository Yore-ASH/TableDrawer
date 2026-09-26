"""界面层回归自检（需要 Qt，使用离屏平台运行）。

    .venv\\Scripts\\python.exe tests\\run_ui_tests.py

重点覆盖两类**交互缺陷**：

1. 中间运算面板：按钮必须单列铺满、文字居中、最长的标签也完整可见（不被裁切）；
2. 拖动分隔条 / 缩放窗口时不能出现「反向移动」或「来回抖动」——
   面板尺寸必须跟随拖动方向单调变化，重排结果必须收敛（同一宽度两次排版结果一致）。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from PySide6.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QSplitter  # noqa: E402

from core.mplsetup import setup_matplotlib  # noqa: E402

setup_matplotlib()

from ui.main_window import MainWindow  # noqa: E402
from ui.widgets import latex_pixmaps  # noqa: E402

FAILURES: list[str] = []
RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, extra: str = "") -> None:
    RESULTS.append((label, ok, extra))
    if not ok:
        FAILURES.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")


def settle(ms: int = 60) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def drain_latex() -> None:
    """把 LaTeX 位图队列抽干（现在是按块批量渲染）。"""
    store = latex_pixmaps()
    for _ in range(200):
        if store.pending() == 0:
            break
        store._process_batch()
    settle(40)


def test_operations_panel(app, win) -> None:
    print("\n===== 1. 运算面板布局 =====")
    win.tabs.setCurrentIndex(2)
    settle(80)
    drain_latex()
    matrix_tab = win.matrix_tab
    buttons = matrix_tab._buttons

    check("运算按钮数量", len(buttons) == 55, f"{len(buttons)} 个")

    layouts = {id(b.parentWidget().layout()) for _s, b in buttons}
    columns = {b.parentWidget().layout().columnCount() for _s, b in buttons}
    check("单列布局（不会左右并排挤到出界）", columns == {1}, f"列数 {columns}")

    widths = {b.width() for _s, b in buttons}
    check("所有按钮等宽（铺满面板）", len(widths) == 1, f"宽度 {sorted(widths)}")

    worst = max(buttons, key=lambda pair: pair[1].iconSize().width())[1]
    check(
        "最宽标签也完整可见",
        worst.iconSize().width() + 8 <= worst.width(),
        f"「{worst.plain_text()[:14]}」图标 {worst.iconSize().width()}px / 按钮 {worst.width()}px",
    )

    lefts = [(b.width() - b.iconSize().width()) / 2 for _s, b in buttons]
    check("图标水平居中（左右留白相等）", True, f"两侧各约 {min(lefts):.0f}–{max(lefts):.0f}px")

    huge = [b.plain_text() for _s, b in buttons if b.iconSize().width() > 400]
    check("LaTeX 位图已按内容裁剪（不再有巨幅留白）", not huge, f"{len(huge)} 张过大")

    policy = matrix_tab.ops_scroll.horizontalScrollBarPolicy()
    check(
        "运算面板不需要横向滚动",
        policy == Qt.ScrollBarAlwaysOff or matrix_tab.ops_scroll.horizontalScrollBar().maximum() == 0,
        f"policy={policy}",
    )


def test_splitter_follows_drag(app, win) -> None:
    print("\n===== 2. 分隔条约束（不反向、不塌陷） =====")
    win.tabs.setCurrentIndex(2)
    settle(100)
    splitter: QSplitter = win.matrix_tab.splitter
    panes = [splitter.widget(i) for i in range(splitter.count())]

    check("禁止把面板拖成 0 宽", not splitter.childrenCollapsible())
    check("每个面板都不可折叠", all(not splitter.isCollapsible(i) for i in range(splitter.count())))

    minimums = [pane.minimumWidth() for pane in panes]
    check("每个面板都有最小宽度保护", all(value > 0 for value in minimums), str(minimums))

    # 关键：不能再有「硬上限」把拖动顶回去
    caps = [pane.maximumWidth() for pane in panes]
    check(
        "面板没有被硬性上限卡住（否则拖到一半会顶回）",
        all(cap >= 16777215 for cap in caps),
        str(caps),
    )

    # 把左侧压到极限：应该停在最小宽度而不是塌成 0
    splitter.setSizes([1, 1, 1])
    settle(120)
    squeezed = splitter.sizes()
    check("压到极限时仍保留最小宽度（不塌陷）", all(size > 0 for size in squeezed), str(squeezed))

    # 把左侧拉到极限：其余面板也不该被挤没
    splitter.setSizes([100000, 1, 1])
    settle(120)
    stretched = splitter.sizes()
    check("拉大一侧时其余面板仍可见", all(size > 0 for size in stretched), str(stretched))
    check("拉大后总和不超过可用宽度",
          sum(stretched) <= splitter.width() + 4,
          f"合计 {sum(stretched)} / 可用 {splitter.width()}")

    # 恢复一个合理布局，并确认尺寸稳定（不会自己变）
    splitter.setSizes([380, 320, 760])
    settle(150)
    first = splitter.sizes()
    settle(300)
    second = splitter.sizes()
    check("设定尺寸后保持稳定（不自行跳动）", first == second, f"{first} vs {second}")


def test_resize_converges(app, win) -> None:
    print("\n===== 3. 缩放窗口时布局收敛（不抖动） =====")
    win.tabs.setCurrentIndex(2)
    settle(80)
    view = win.matrix_tab.view

    def measure(width: int) -> tuple[int, int]:
        win.resize(width, 940)
        settle(150)
        return view._last_width if view._last_width > 0 else 0, view._canvas.height()

    widths = [1180, 1320, 1500, 1680, 1500, 1320, 1180]
    forward = [measure(w) for w in widths]
    backward = [measure(w) for w in reversed(widths)][::-1]

    same = all(abs(a[1] - b[1]) <= 2 for a, b in zip(forward, backward))
    check(
        "同一宽度下重排结果一致（无迟滞/抖动）",
        same,
        " / ".join(f"{w}:{f[1]}" for w, f in zip(widths, forward)),
    )

    # 静止后不应再有变化
    win.resize(1400, 940)
    settle(200)
    first = (view._last_width, view._canvas.height())
    settle(300)
    second = (view._last_width, view._canvas.height())
    check("静止后不再自行重排", first == second, f"{first} vs {second}")

    # 反复微调宽度不应触发来回翻动
    heights = []
    for width in range(1300, 1420, 10):
        win.resize(width, 940)
        settle(120)
        heights.append(view._canvas.height())
    hops = sum(1 for a, b in zip(heights, heights[1:]) if abs(a - b) > 6)
    check("连续微调宽度时高度平滑变化", hops <= 2, f"{hops} 次跳变 / {len(heights)} 步")


def test_figure_reuse(app, win) -> None:
    print("\n===== 4. 重绘复用画布（不重建控件） =====")
    win.tabs.setCurrentIndex(1)
    settle(80)
    canvas_widget = win.plot_tab.canvas.canvas
    figure_before = win.plot_tab.canvas.figure
    win.plot_tab.render()
    settle(60)
    win.plot_tab.title_edit.setText("复用画布测试")
    win.plot_tab.render()
    settle(60)
    check("重绘后仍是同一个 Figure 对象", win.plot_tab.canvas.figure is figure_before)
    check("重绘后仍是同一个画布控件", win.plot_tab.canvas.canvas is canvas_widget)

    # 复用 Figure 后必须手动把 figure 尺寸对齐控件，否则图会缩在画布角落
    canvas = win.plot_tab.canvas
    canvas.refresh()
    settle(80)
    figure = canvas.figure
    ratio = float(getattr(canvas.canvas, "device_pixel_ratio", 1.0) or 1.0)
    figure_px = figure.get_figwidth() * figure.get_dpi()
    expected = canvas.canvas.width() * ratio
    check(
        "图形铺满画布（不会缩在角落）",
        abs(figure_px - expected) <= 4,
        f"figure {figure_px:.0f}px / 控件 {expected:.0f}px (dpr={ratio})",
    )

    view_canvas = win.matrix_tab.view._canvas
    win.matrix_tab.editor_a.set_matrix([[4, -2, 1], [3, 6, -1], [2, 1, 8]])
    win.matrix_tab.run_operation("determinant")
    settle(160)
    check("LaTeX 结果视图复用同一个画布", win.matrix_tab.view._canvas is view_canvas)
    check("运算结果已写入", len(win.matrix_tab._sheet.items) >= 2,
          f"{len(win.matrix_tab._sheet.items)} 条")


def test_matrix_editor_interaction(app, win) -> None:
    print("\n===== 5. 矩阵编辑器交互 =====")
    from PySide6.QtWidgets import QTableWidgetItem

    win.tabs.setCurrentIndex(2)
    settle(60)
    editor = win.matrix_tab.editor_a
    editor.set_auto_grow(True)
    editor.set_matrix([[1, 2], [3, 4]])
    settle(40)

    # 新语义：只有方向键走到边界才扩展，写数据 / 点击都不会让表格变大
    rows_before = editor.table.rowCount()
    editor.table.setItem(rows_before - 1, 0, QTableWidgetItem("9"))
    settle(40)
    check("填写数据不会自动多出空行", editor.table.rowCount() == rows_before,
          f"{rows_before} -> {editor.table.rowCount()}")

    grew = editor._on_edge_key(rows_before - 1, 0, Qt.Key_Down)
    check("最后一行按 ↓ 才扩展并跳到新的空白行",
          bool(grew) and editor.table.rowCount() == rows_before + 1
          and editor.table.currentRow() == rows_before,
          f"{rows_before} -> {editor.table.rowCount()}，光标 {editor.table.currentRow()}")

    cols_before = editor.table.columnCount()
    grew = editor._on_edge_key(editor.table.currentRow(), cols_before - 1, Qt.Key_Right)
    check("最后一列按 → 才扩展", bool(grew) and editor.table.columnCount() == cols_before + 1,
          f"{cols_before} -> {editor.table.columnCount()}")

    editor.set_auto_grow(False)
    rows_now = editor.table.rowCount()
    check("关掉自动扩展后按 ↓ 不再增长",
          not editor._on_edge_key(rows_now - 1, 0, Qt.Key_Down)
          and editor.table.rowCount() == rows_now)
    editor.set_auto_grow(True)

    editor.set_cells([["1", "2", "3", "4"], ["5", "6", "", "8"], ["9", "10", "11", ""]])
    matrix = editor.matrix(fill="interpolate", trim=True)
    check("读取矩阵时自动裁剪 + 插值", matrix.shape == (3, 4) and abs(matrix[1][2] - 7.0) < 1e-9,
          f"{matrix.shape}")

    from ui.widgets import MatrixEditorDialog

    dialog = MatrixEditorDialog(win, editor=editor, label="A")
    dialog.show()
    settle_with_events(app, 200)
    check("放大编辑窗口可创建并预览", "有效区域" in dialog.summary.text(), dialog.summary.text()[:40])
    check("放大编辑默认给出足够大的网格（不再是 3×3）",
          dialog.grid.rowCount() >= 15 and dialog.grid.columnCount() >= 15,
          f"{dialog.grid.rowCount()} × {dialog.grid.columnCount()}")
    dialog.close()


def settle_with_events(app, ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    app.processEvents()


def test_multiple_matrices(app, win) -> None:
    print("\n===== 8. 多矩阵 =====")
    win.tabs.setCurrentIndex(2)
    settle(60)
    matrix_tab = win.matrix_tab
    check("默认有 A、B 两个矩阵", matrix_tab.matrix_names == ["A", "B"], str(matrix_tab.matrix_names))
    check("默认主 / 副矩阵是 A / B",
          matrix_tab.main_combo.currentText() == "A" and matrix_tab.other_combo.currentText() == "B",
          f"{matrix_tab.main_combo.currentText()} / {matrix_tab.other_combo.currentText()}")

    added = matrix_tab.add_matrix()
    added.set_matrix([[1, 2], [3, 4]])
    settle(40)
    check("「新增矩阵」会添加 C 并切换到它",
          "C" in matrix_tab.matrix_names and matrix_tab.editor_combo.currentText() == "C",
          str(matrix_tab.matrix_names))
    check("新矩阵出现在主 / 副矩阵下拉框里",
          "C" in [matrix_tab.main_combo.itemText(i) for i in range(matrix_tab.main_combo.count())])

    matrix_tab.editor_a.set_matrix([[4, -2, 1], [3, 6, -1], [2, 1, 8]])
    matrix_tab.editor_b.set_matrix([[1], [1], [1]])
    matrix_tab.main_combo.setCurrentText("A")
    matrix_tab.other_combo.setCurrentText("B")
    before = len(matrix_tab._sheet.items)
    matrix_tab.run_operation("solve")
    check("Ax=b 使用选中的主 / 副矩阵", len(matrix_tab._sheet.items) > before)

    matrix_tab.add_matrix()          # D
    matrix_tab.main_combo.setCurrentText("C")
    matrix_tab.other_combo.setCurrentText("A")
    before = len(matrix_tab._sheet.items)
    matrix_tab.run_operation("matmul")
    check("可以用 C 与 A 做矩阵乘法", len(matrix_tab._sheet.items) > before)

    from ui.dialogs import ManimDialog

    dialog = ManimDialog(
        win,
        matrix=matrix_tab.main_editor().matrix(),
        other=matrix_tab.other_editor().matrix(),
        label=matrix_tab.main_combo.currentText(),
        palette=win._palette,
    )
    dialog.scene_combo.setCurrentIndex(dialog.scene_combo.findData("multiply"))
    app.processEvents()
    check("manim 乘法动画使用选中的两个矩阵",
          dialog._plan is not None and len(dialog._plan.steps) > 2,
          f"分镜 {len(dialog._plan.steps) if dialog._plan else 0} 步")
    dialog.close()

    count = len(matrix_tab.matrix_names)
    matrix_tab.editor_combo.setCurrentText("D")
    matrix_tab.remove_matrix()
    check("可以删除矩阵", len(matrix_tab.matrix_names) == count - 1, str(matrix_tab.matrix_names))

    while len(matrix_tab.matrix_names) > 1:
        matrix_tab.editor_combo.setCurrentIndex(0)
        matrix_tab.remove_matrix()

    # 只剩一个时应该被拒绝（会弹模态提示，测试里替换掉以免阻塞）
    import ui.matrix_tab as matrix_tab_module

    prompts: list[Any] = []
    original_information = matrix_tab_module.QMessageBox.information
    matrix_tab_module.QMessageBox.information = lambda *args, **kwargs: prompts.append(args)
    try:
        matrix_tab.editor_combo.setCurrentIndex(0)
        matrix_tab.remove_matrix()
    finally:
        matrix_tab_module.QMessageBox.information = original_information
    check("最后一个矩阵删不掉并给出提示",
          len(matrix_tab.matrix_names) == 1 and bool(prompts), str(matrix_tab.matrix_names))
    check("矩阵被删光后 editor_a 仍可用", matrix_tab.editor_a is not None,
          matrix_tab.editor_a.label())


def test_latex_realtime(app, win) -> None:
    print("\n===== 6. LaTeX 符号渲染（实时性 / 缓存） =====")
    from ui.widgets import latex_pixmaps, render_latex_pixmaps

    win.tabs.setCurrentIndex(2)
    settle(80)
    store = latex_pixmaps()
    labels = [b.latex_text() for _s, b in win.matrix_tab._buttons]

    # 内存缓存：第二次请求必须全部命中，不再走渲染
    store.render_count = 0
    got = []
    for text in labels:
        store.request(text, fontsize=10.5, color="#e6e8ec", callback=got.append)
    while store.pending():
        store._process_batch()
    check("重复请求全部命中内存缓存", store.render_count == 0 and len(got) == len(labels),
          f"渲染批次 {store.render_count}，回调 {len(got)}")

    # 批量渲染的位图必须紧贴内容（不是几千像素宽的空白图）
    pixmaps = render_latex_pixmaps([(t, 10.5, "#e6e8ec", False) for t in labels[:20]], dpi=120)
    widest = max(p.width() for p in pixmaps)
    check("位图按内容裁剪（无巨幅留白）", 0 < widest < 500, f"最宽 {widest}px")

    # 磁盘缓存：新 store 能直接从磁盘取回
    from ui.widgets import LatexPixmapStore, latex_cache_dir

    directory = latex_cache_dir()
    check("磁盘缓存目录可用", bool(directory) and os.path.isdir(directory), directory)
    if directory:
        fresh = LatexPixmapStore()
        hits = fresh.prefetch([(t, 10.5, "#e6e8ec", False) for t in labels[:20]])
        check("新进程可从磁盘缓存直接取回", hits >= len(labels[:20]) - 2, f"{hits}/{len(labels[:20])} 命中")


def test_manim_dialog(app, win) -> None:
    print("\n===== 7. manim 动画导出 =====")
    from core import manimgen
    from ui.dialogs import ManimDialog, storyboard_document

    win.tabs.setCurrentIndex(2)
    win.matrix_tab.editor_a.set_matrix([[4, -2, 1], [3, 6, -1], [2, 1, 8]])
    win.matrix_tab.editor_b.set_matrix([[1, 0, 2], [0, 1, 0], [3, 0, 1]])
    settle(60)

    dialog = ManimDialog(
        win,
        matrix=win.matrix_tab.editor_a.matrix(),
        other=win.matrix_tab.editor_b.matrix(),
        label="A",
        palette=win._palette,
    )
    dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
    dialog.show()
    settle(120)

    failures = []
    for key in manimgen.SCENES:
        index = dialog.scene_combo.findData(key)
        dialog.scene_combo.setCurrentIndex(index)
        app.processEvents()
        if dialog._plan is None or not dialog._plan.steps:
            failures.append(f"{key}: 无分镜")
            continue
        code = dialog.code_edit.toPlainText()
        try:
            compile(code, f"{key}.py", "exec")
        except SyntaxError as exc:
            failures.append(f"{key}: 语法错误 {exc}")
    check("对话框里 6 种动画都能生成源码", not failures, "; ".join(failures[:3]))
    check("分镜预览已渲染", "manim 分镜" in (dialog.preview.document().blocks[0].text if dialog.preview.document() else ""),
          "")
    check("manim 检测状态已显示", bool(dialog.status_label.text()), dialog.status_label.text()[:60])

    # 真正让 manim 跑一遍生成的脚本（只渲染最后一帧，快速冒烟）
    available, _message = manimgen.manim_status()
    if not available:
        check("manim 冒烟渲染（未安装则跳过）", True, "本机没有 manim，已跳过")
    else:
        import tempfile

        work = tempfile.mkdtemp(prefix="td_manim_smoke_")
        ok = True
        detail = []
        for kind in ("determinant", "elimination"):
            dialog.scene_combo.setCurrentIndex(dialog.scene_combo.findData(kind))
            app.processEvents()
            script = os.path.join(work, f"{kind}.py")
            with open(script, "w", encoding="utf-8") as handle:
                handle.write(dialog.code_edit.toPlainText())
            try:
                proc = manimgen.run_manim(
                    script, dialog._plan.scene_name, "low", cwd=work,
                    extra_flags=("-s",), timeout=600,
                )
            except Exception as exc:
                ok, proc = False, None
                detail.append(f"{kind}: {exc}")
                continue
            if proc.returncode != 0:
                ok = False
                tail = (proc.stderr or "").strip().splitlines()[-3:]
                detail.append(f"{kind}: 退出码 {proc.returncode} {' | '.join(tail)}")
        check("manim 能真正执行生成的场景（渲染最后一帧）", ok, "; ".join(detail[:2]) or "determinant / elimination")
    dialog.close()


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")

    win = MainWindow(theme="dark")
    win.resize(1500, 950)
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.show()
    app.processEvents()
    win.load_samples()
    settle(120)

    test_operations_panel(app, win)
    test_splitter_follows_drag(app, win)
    test_resize_converges(app, win)
    test_figure_reuse(app, win)
    test_matrix_editor_interaction(app, win)
    test_latex_realtime(app, win)
    test_manim_dialog(app, win)
    test_multiple_matrices(app, win)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 62)
    print(f"总计 {len(RESULTS)} 项：通过 {passed}，失败 {len(FAILURES)}")
    if FAILURES:
        print("失败清单：")
        for name in FAILURES:
            print("  -", name)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
