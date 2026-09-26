"""端到端集成自检（不依赖 pytest，直接运行即可）。

    .venv\\Scripts\\python.exe tests\\run_tests.py

覆盖：
  1. 全部 22 种图表类型在 2D/3D 下都能画出来并导出 SVG
  2. 全部 55 个矩阵/向量组运算都能执行且产出可渲染的 LaTeX
  3. LaTeX 引擎对矩阵环境、中文、定界符的稳定渲染
  4. 数据层：CSV 读取、表达式求值、统计摘要
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from core.mplsetup import setup_matplotlib

setup_matplotlib(interactive=False)

from core import matrixops, plotting
from core.dataset import DataSet, DataSetStore, evaluate_expression, read_csv_frame
from core.spec import KINDS_2D, KINDS_3D, PlotSpec, SeriesSpec

FAILURES: list[str] = []
RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, extra: str = "") -> None:
    RESULTS.append((label, ok, extra))
    if not ok:
        FAILURES.append(label)
    print(f"{'OK  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")


# --------------------------------------------------------------------------- #
# 测试数据
# --------------------------------------------------------------------------- #
def build_datasets() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    t = np.linspace(0.0, 6.0, 80)

    flat = pd.DataFrame(
        {
            "x": np.round(t, 4),
            "y": np.round(np.sin(t) * np.exp(-t / 8) * 3 + 5, 4),
            "y2": np.round(np.cos(t) * 2 + 5, 4),
            "err": np.round(np.full(t.size, 0.25), 4),
            "类别": [f"G{i % 4 + 1}" for i in range(t.size)],
        }
    )

    gx, gy = np.meshgrid(np.linspace(-2.0, 2.0, 30), np.linspace(-2.0, 2.0, 30))
    r = np.hypot(gx, gy)
    grid = pd.DataFrame(
        {
            "x": np.round(gx.ravel(), 4),
            "y": np.round(gy.ravel(), 4),
            "z": np.round((np.sin(r * 2) / (r + 0.4) * 3).ravel(), 6),
            "u": np.round(-gy.ravel(), 4),
            "v": np.round(gx.ravel(), 4),
            "w": np.round(np.sin(r).ravel(), 4),
        }
    )
    return {"平面": flat, "网格": grid}


DATASETS = build_datasets()

#: 每种图表类型的 SeriesSpec 配置
SERIES_BY_KIND: dict[str, dict] = {
    "line": dict(dataset="平面", x="x", y="y,y2", marker="o"),
    "scatter": dict(dataset="平面", x="x", y="y", marker="s"),
    "bar": dict(dataset="平面", x="x", y="y"),
    "barh": dict(dataset="平面", x="x", y="y"),
    "stem": dict(dataset="平面", x="x", y="y"),
    "step": dict(dataset="平面", x="x", y="y"),
    "fill": dict(dataset="平面", x="x", y="y"),
    "errorbar": dict(dataset="平面", x="x", y="y", yerr="err"),
    "hist": dict(dataset="平面", y="y", bins=15),
    "box": dict(dataset="平面", y="y,y2"),
    "pie": dict(dataset="平面", x="类别", y="y"),
    "quiver": dict(dataset="网格", x="x", y="y", u="u", v="v"),
    "contour": dict(dataset="网格", x="x", y="y", z="z"),
    "heatmap": dict(dataset="网格", x="x", y="y", z="z"),
    "line3d": dict(dataset="网格", x="x", y="y", z="z"),
    "scatter3d": dict(dataset="网格", x="x", y="y", z="z"),
    "surface": dict(dataset="网格", x="x", y="y", z="z"),
    "wireframe": dict(dataset="网格", x="x", y="y", z="z"),
    "contour3d": dict(dataset="网格", x="x", y="y", z="z"),
    "trisurf": dict(dataset="网格", x="x", y="y", z="z"),
    "bar3d": dict(dataset="网格", x="x", y="y", z="z"),
    "quiver3d": dict(dataset="网格", x="x", y="y", z="z", u="u", v="v", w="w"),
}


def test_all_plot_kinds() -> None:
    print("\n===== 1. 全部图表类型 =====")
    for kind, kwargs in SERIES_BY_KIND.items():
        for projection in ("2d", "3d"):
            table = KINDS_3D if projection == "3d" else KINDS_2D
            if kind not in table:
                continue
            spec = PlotSpec(
                projection=projection,
                title=f"{table[kind]} $y=f(x)$",
                xlabel="X 轴",
                ylabel="Y 轴",
                zlabel="Z 轴",
                series=[SeriesSpec(kind=kind, **kwargs)],
            )
            label = f"绘图 {projection} / {kind}"
            try:
                figure, warnings = plotting.build_figure(spec, DATASETS)
                svg = plotting.figure_to_svg(figure)
                ok = "<svg" in svg and len(svg) > 800 and len(figure.axes) > 0
                check(label, ok, f"{len(svg)}B" + (f" 警告:{len(warnings)}" if warnings else ""))
            except Exception as exc:
                check(label, False, f"{type(exc).__name__}: {exc}")


def test_missing_columns_are_safe() -> None:
    print("\n===== 2. 缺列/空数据的容错 =====")
    spec = PlotSpec(series=[SeriesSpec(dataset="平面", kind="line", x="不存在", y="也没有")])
    try:
        figure, warnings = plotting.build_figure(spec, DATASETS)
        check("缺列不抛异常且给出警告", bool(warnings) and len(figure.axes) > 0, f"{len(warnings)} 条警告")
    except Exception as exc:
        check("缺列不抛异常且给出警告", False, str(exc))

    spec2 = PlotSpec(series=[SeriesSpec(dataset="空数据", kind="line", x="a", y="b")])
    try:
        figure, warnings = plotting.build_figure(spec2, {"空数据": pd.DataFrame({"a": [], "b": []})})
        check("空数据集不抛异常", len(figure.axes) > 0, f"{len(warnings)} 条警告")
    except Exception as exc:
        check("空数据集不抛异常", False, str(exc))

    try:
        figure, _ = plotting.build_figure(PlotSpec(), {})
        check("无序列时画出占位图", len(figure.axes) > 0)
    except Exception as exc:
        check("无序列时画出占位图", False, str(exc))


def test_all_matrix_ops() -> None:
    print("\n===== 3. 全部矩阵 / 向量组运算 =====")
    matrix_a = np.array([[4.0, -2.0, 1.0], [3.0, 6.0, -1.0], [2.0, 1.0, 8.0]])
    matrix_b = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [3.0, 0.0, 1.0]])
    vector_a = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])

    ok_count = 0
    latex_bad: list[str] = []
    for spec in matrixops.REGISTRY:
        a = vector_a if spec.vector_input else (matrix_a if spec.arity >= 1 else None)
        b = matrix_b if spec.arity >= 2 else (matrix_b if spec.vector_input and spec.arity >= 2 else None)
        if spec.arity >= 2 and not spec.vector_input:
            b = matrix_b
        try:
            sheet = matrixops.run(spec.key, a, b, 3)
        except Exception as exc:
            check(f"运算 {spec.key}", False, f"{type(exc).__name__}: {exc}")
            continue
        problems = []
        if not isinstance(sheet, type(matrixops.run("rank", matrix_a, None, None))):
            problems.append("返回类型错误")
        if not sheet.items:
            problems.append("没有结果项")
        for item in sheet.items:
            if item.latex:
                if item.latex.count("{") != item.latex.count("}"):
                    problems.append("花括号不平衡")
                if any("\u4e00" <= ch <= "\u9fff" for ch in item.latex):
                    problems.append("latex 含中文")
        if problems:
            latex_bad.append(f"{spec.key}: {'; '.join(sorted(set(problems)))}")
        else:
            ok_count += 1
    check(f"55 个运算全部执行且 LaTeX 合规（{ok_count}/{len(matrixops.REGISTRY)}）",
          not latex_bad, "; ".join(latex_bad[:4]))


def test_latex_render_of_all_results() -> None:
    print("\n===== 4. 所有运算结果都能被 LaTeX 引擎渲染 =====")
    from core.latex import document_for_sheet, layout_document

    matrix_a = np.array([[4.0, -2.0, 1.0], [3.0, 6.0, -1.0], [2.0, 1.0, 8.0]])
    combined = []
    for key in ("determinant", "inverse", "eigen", "svd", "rref", "lu", "norms", "nullspace", "charpoly"):
        sheet = matrixops.run(key, matrix_a, None, None)
        combined.extend(sheet.items)
    sheet = matrixops.run("transpose", matrix_a, None, None)
    combined.extend(sheet.items)

    try:
        document = document_for_sheet(combined)
        result = layout_document(document, width=680)
        check("结果文档排版", result.height > 100 and len(result.draws) > 20,
              f"{result.width:.0f}x{result.height:.0f}, {len(result.draws)} 个绘制指令")
    except Exception as exc:
        check("结果文档排版", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


def test_latex_engine_robustness() -> None:
    print("\n===== 5. LaTeX 引擎边界情况 =====")
    from core.latex import LatexDocument, MathBlock, TextBlock, layout_document, render_document

    cases = {
        "空文档": LatexDocument(),
        "空公式": LatexDocument(blocks=[MathBlock(text="")]),
        "残缺环境": LatexDocument(blocks=[MathBlock(text=r"\begin{pmatrix} 1 & 2")]),
        "不配对括号": LatexDocument(blocks=[MathBlock(text=r"A^{-1} = \begin{pmatrix} 1 & 2 \\ 3")]),
        "中文混排": LatexDocument(blocks=[TextBlock(text="中文 $a^2+b^2=c^2$ 混排" * 3)]),
        "中文在公式内": LatexDocument(blocks=[MathBlock(text=r"P(A) = \begin{cases} 1 & A \text{可逆} \\ 0 & \text{其他} \end{cases}")]),
        "超长公式": LatexDocument(blocks=[MathBlock(text=r"\sum_{i=1}^{n} " + " + ".join(f"x_{{{i}}}^2" for i in range(60)))]),
        "嵌套矩阵": LatexDocument(blocks=[MathBlock(text=r"\begin{pmatrix} \begin{matrix} a & b \\ c & d \end{matrix} & 0 \\ 0 & I \end{pmatrix}")]),
        "未知命令": LatexDocument(blocks=[MathBlock(text=r"\foo{bar} \baz")]),
        "深度嵌套": LatexDocument(blocks=[MathBlock(text=r"\frac{\frac{\frac{1}{2}}{3}}{4}")]),
    }
    for name, document in cases.items():
        try:
            result = layout_document(document, width=420)
            figure = render_document(document, width=420, dpi=72)
            svg = __import__("core.latex", fromlist=["figure_to_svg"]).figure_to_svg(figure)
            check(f"LaTeX 边界：{name}", result.height >= 1 and "<svg" in svg,
                  f"h={result.height:.0f}")
        except Exception as exc:
            check(f"LaTeX 边界：{name}", False, f"{type(exc).__name__}: {exc}")

    # 各种宽度下重排（模拟窗口缩放）
    document = LatexDocument(blocks=[TextBlock(text="中文段落折行测试。" * 20),
                                     MathBlock(text=r"A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}")])
    heights = []
    for width in (900, 600, 400, 260, 140, 80):
        result = layout_document(document, width=width)
        heights.append(result.height)
    check("随宽度重排且高度单调不减", all(b >= a - 1 for a, b in zip(heights, heights[1:])),
          " -> ".join(f"{h:.0f}" for h in heights))


def test_data_layer() -> None:
    print("\n===== 6. 数据层 =====")
    samples = os.path.join(ROOT, "samples")
    csv_files = sorted(f for f in os.listdir(samples) if f.endswith(".csv")) if os.path.isdir(samples) else []
    check("samples 目录存在示例 CSV", len(csv_files) >= 5, f"{len(csv_files)} 个")

    for name in csv_files:
        path = os.path.join(samples, name)
        try:
            preview = read_csv_frame(path)
            ok = preview.frame.shape[0] > 0 and preview.frame.shape[1] > 0
            check(f"读取 {name}", ok,
                  f"{preview.frame.shape[0]}×{preview.frame.shape[1]} 编码={preview.encoding} 分隔符={preview.delimiter!r}")
        except Exception as exc:
            check(f"读取 {name}", False, f"{type(exc).__name__}: {exc}")

    frame = DATASETS["平面"]
    for expr, note in (
        ("sin(x) * exp(-x/5)", "基本函数"),
        ("linspace(0, 2*pi, n)", "linspace"),
        ("y + y2", "列引用"),
        ("np.where(x > 3, 1, 0)", "np 命名空间"),
        ("x**2 - 3*x + 1", "幂运算"),
    ):
        try:
            values = evaluate_expression(expr, frame)
            check(f"表达式：{note}", len(values) == len(frame), f"{expr}")
        except Exception as exc:
            check(f"表达式：{note}", False, f"{expr}: {exc}")

    for bad in ("__import__('os').system('echo hi')", "open('x')", "1/0", "nosuchname"):
        try:
            evaluate_expression(bad, frame)
            check(f"拒绝/报错：{bad[:24]}", False, "本应报错却成功了")
        except ValueError:
            check(f"拒绝/报错：{bad[:24]}", True)

    try:
        from ui.data_tab import build_summary_sheet

        store = DataSetStore()
        store.add(DataSet(name="测试", frame=frame))
        sheet = build_summary_sheet(store.require("测试"))
        check("统计摘要生成", len(sheet.items) >= 2 and sheet.plain_text().strip() != "")
    except Exception as exc:
        check("统计摘要生成", False, f"{type(exc).__name__}: {exc}")


def test_export_roundtrip() -> None:
    print("\n===== 7. 导出 =====")
    import tempfile

    spec = PlotSpec(title="导出测试", xlabel="x", ylabel="y",
                    series=[SeriesSpec(dataset="平面", kind="line", x="x", y="y")])
    figure, _ = plotting.build_figure(spec, DATASETS)
    tmp = tempfile.mkdtemp(prefix="td_export_")
    for ext in ("svg", "png", "pdf"):
        path = os.path.join(tmp, f"figure.{ext}")
        try:
            plotting.save_figure(figure, path)
            size = os.path.getsize(path)
            check(f"导出 {ext.upper()}", size > 500, f"{size}B")
        except Exception as exc:
            check(f"导出 {ext.upper()}", False, f"{type(exc).__name__}: {exc}")

    from core import codesync

    code = codesync.generate_script(spec, DATASETS, codesync.CodeGenOptions())
    check("生成源码", "matplotlib" in code and len(code.splitlines()) > 20, f"{len(code.splitlines())} 行")
    try:
        compile(code, "<generated>", "exec")
        check("生成源码语法正确", True)
    except SyntaxError as exc:
        check("生成源码语法正确", False, str(exc))

    from core.latex import LatexDocument, MathBlock, figure_to_svg

    document = LatexDocument(blocks=[MathBlock(text=r"A^{-1} = \begin{pmatrix} -2 & 1 \\ 1.5 & -0.5 \end{pmatrix}")])
    from core.latex import render_document

    svg = figure_to_svg(render_document(document, width=500))
    check("LaTeX 导出 SVG", "<svg" in svg and "</svg>" in svg, f"{len(svg)}B")


def test_rich_labels_and_fonts() -> None:
    print("\n===== 8. 混排标签 / 字体 / 矩阵输入策略 =====")
    import logging

    from core.mplsetup import FONT_PRESETS, apply_font_preset, font_config
    from core.spec import PlotSpec as _Spec
    from core.spec import SeriesSpec as _Series

    # 字体预设
    for preset in FONT_PRESETS:
        try:
            config = apply_font_preset(preset)
            check(f"字体预设 {preset}", bool(config["latin"]) and bool(config["cjk"]) and bool(config["mono"]),
                  f"{config['latin'][0]} / {config['cjk'][0]} / {config['mono'][0]}")
        except Exception as exc:
            check(f"字体预设 {preset}", False, str(exc))
    apply_font_preset("paper")

    # 字体回退链
    import matplotlib

    chain = list(matplotlib.rcParams["font.family"])
    check("font.family 是具体字体链（否则中文回退失效）",
          len(chain) > 1 and "serif" not in chain and "sans-serif" not in chain, " → ".join(chain[:4]))

    # 逐字符字体回退的度量
    from core.latex.measure import measure_text

    chain_tuple = tuple(chain)
    width_chain = measure_text("时间", 11.0, chain_tuple)[0]
    width_cjk = measure_text("时间", 11.0, ("SimSun",))[0]
    check("中文度量走 CJK 回退（不是 Times 的缺字宽度）",
          abs(width_chain - width_cjk) < 0.5, f"{width_chain:.2f} vs {width_cjk:.2f}")

    # 混排标签：中文 + 公式
    from core import plotting as _plotting

    check("split_cjk_math 识别混排", _plotting.split_cjk_math(r"信号 $y=\sin t$") is not None)
    check("split_cjk_math 纯中文返回 None", _plotting.split_cjk_math("温度曲线") is None)
    check("split_cjk_math 纯公式返回 None", _plotting.split_cjk_math(r"$y=x^2$") is None)

    frame = pd.DataFrame({"x": np.arange(30.0), "y": np.sin(np.arange(30.0) / 4)})
    spec = _Spec(
        title=r"衰减振荡 $y=e^{-\zeta\omega t}\cos(\omega_d t)$ 曲线",
        xlabel=r"时间 $t$ / s",
        ylabel=r"幅值 $A$",
        series=[_Series(dataset="d", kind="line", x="x", y="y")],
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("matplotlib")
    logger.addHandler(handler)
    try:
        stream.truncate(0)
        stream.seek(0)
        figure, _notes = _plotting.build_figure(spec, {"d": frame})
        figure.savefig(os.path.join(tempfile.gettempdir(), "td_rich.png"), dpi=110)
        glyph_warnings = [line for line in stream.getvalue().splitlines() if "glyph" in line.lower()]
    finally:
        logger.removeHandler(handler)
    check("中文+公式混排标题不再出现缺字（豆腐块）", not glyph_warnings,
          f"{len(glyph_warnings)} 条" + (f"：{glyph_warnings[0][:70]}" if glyph_warnings else ""))

    # 片段的实际包围盒不应互相重叠
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    boxes = sorted(
        (text.get_window_extent(renderer).extents for text in figure.axes[0].texts),
        key=lambda box: box[0],
    )
    overlaps = [
        (round(boxes[i][2], 1), round(boxes[i + 1][0], 1))
        for i in range(len(boxes) - 1)
        if boxes[i][2] > boxes[i + 1][0] + 1.0 and abs(boxes[i][1] - boxes[i + 1][1]) < 5
    ]
    check("混排标签片段不重叠", not overlaps, str(overlaps[:3]))

    # 矩阵输入策略
    from ui.widgets import FILL_POLICIES, grid_to_matrix

    cells = [["1", "2", "3", "4"], ["5", "6", "", "8"], ["9", "10", "11", ""]]
    matrix_zero, info = grid_to_matrix(cells, fill="zero", trim=True)
    check("自动识别有效区域", matrix_zero.shape == (3, 4), f"{matrix_zero.shape}, 空白 {info['blanks']} 处")
    matrix_interp, _ = grid_to_matrix(cells, fill="interpolate", trim=True)
    check("行内线性插值填充", abs(matrix_interp[1][2] - 7.0) < 1e-9, str(matrix_interp.tolist()[1]))
    matrix_prev, _ = grid_to_matrix(cells, fill="previous", trim=True)
    check("向左复制填充", abs(matrix_prev[2][3] - 11.0) < 1e-9, str(matrix_prev.tolist()[2]))
    matrix_nan, _ = grid_to_matrix(cells, fill="nan", trim=True)
    check("NaN 填充", np.isnan(matrix_nan[1][2]))
    try:
        grid_to_matrix(cells, fill="strict", trim=True)
        check("strict 模式报错并指出位置", False, "本应报错")
    except ValueError as exc:
        check("strict 模式报错并指出位置", "2,3" in str(exc) or "(2,3)" in str(exc), str(exc)[:60])
    trimmed, info2 = grid_to_matrix([["1", "2", "", ""], ["3", "", "", ""], ["", "", "", ""]], fill="zero", trim=True)
    check("右下空白被裁掉", trimmed.shape == (2, 2) and info2["trimmed"], f"{trimmed.shape}")
    check("填充策略齐全", set(FILL_POLICIES) >= {"zero", "one", "nan", "previous", "interpolate", "strict"})

    # LaTeX 按钮文案
    from ui.matrix_tab import _LATEX_LABELS

    missing = [spec.key for spec in matrixops.REGISTRY if spec.key not in _LATEX_LABELS]
    check("每个运算都有 LaTeX 按钮文案", not missing, str(missing[:5]))
    from core.latex import LatexDocument, MathBlock, layout_document, TextBlock

    bad_labels = []
    for key, label in _LATEX_LABELS.items():
        document = LatexDocument(fontsize=10.5)
        document.add(TextBlock(text=label, fontsize=10.5, space_before=0.0, space_after=0.0))
        try:
            result = layout_document(document, width=4000)
            if result.width < 5 or result.height < 5:
                bad_labels.append(f"{key}(空)")
        except Exception as exc:
            bad_labels.append(f"{key}({exc})")
    check("全部 LaTeX 按钮文案可排版", not bad_labels, str(bad_labels[:4]))


def test_manim_export() -> None:
    print("\n===== 9. manim 动画导出 =====")
    from core import manimgen

    matrix = [[4.0, -2.0, 1.0], [3.0, 6.0, -1.0], [2.0, 1.0, 8.0]]
    other = [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [3.0, 0.0, 1.0]]

    generated = 0
    bad: list[str] = []
    for key, name in manimgen.SCENES.items():
        try:
            plan = manimgen.plan_scene(key, matrix, other)
            code = manimgen.generate_scene(plan, manimgen.ManimOptions())
            compile(code, f"{key}.py", "exec")
            if not plan.steps:
                bad.append(f"{key}: 没有分镜")
                continue
            if "DARK" in code or "LIGHT" in code:
                bad.append(f"{key}: 使用了 manim 不存在的颜色常量 DARK/LIGHT")
                continue
            if "def construct" not in code or "from manim import *" not in code:
                bad.append(f"{key}: 缺少场景骨架")
                continue
            generated += 1
        except Exception as exc:
            bad.append(f"{key}: {type(exc).__name__}: {exc}")
    check(f"{len(manimgen.SCENES)} 种 manim 动画都能生成且语法正确（{generated}）",
          not bad, "; ".join(bad[:3]))

    # 边界输入
    edge = {
        "奇异矩阵求逆": ("inverse", [[1, 2], [2, 4]], None),
        "非方阵转置": ("transpose", [[1, 2, 3], [4, 5, 6]], None),
        "非方阵特征值": ("eigen", [[1, 2, 3], [4, 5, 6]], None),
        "维度不匹配乘法": ("multiply", [[1, 2]], [[1, 2]]),
        "复数特征值": ("eigen", [[0, -1], [1, 0]], None),
        "二阶行列式": ("determinant", [[1, 2], [3, 4]], None),
        "1x1 矩阵": ("elimination", [[5.0]], None),
    }
    edge_bad: list[str] = []
    for label, (kind, value, second) in edge.items():
        try:
            plan = manimgen.plan_scene(kind, value, second)
            compile(manimgen.generate_scene(plan), "edge.py", "exec")
        except Exception as exc:
            edge_bad.append(f"{label}: {type(exc).__name__}: {exc}")
    check("manim 导出的边界输入不崩", not edge_bad, "; ".join(edge_bad[:3]))

    # 消元分镜必须真的得到行简化阶梯形
    plan = manimgen.plan_scene("elimination", matrix)
    final = plan.steps[-1].rows
    identity = [[str(int(i == j)) for j in range(3)] for i in range(3)]
    check("消元分镜最后一步是单位阵", final == identity, str(final))

    # 分镜预览用的 LaTeX 必须能被自研引擎排版
    from core.latex import layout_document

    preview_bad: list[str] = []
    for key in manimgen.SCENES:
        try:
            plan = manimgen.plan_scene(key, matrix, other)
            from ui.dialogs import storyboard_document

            result = layout_document(storyboard_document(plan), width=700)
            if result.height < 50 or not result.draws:
                preview_bad.append(f"{key}: 预览为空")
        except Exception as exc:
            preview_bad.append(f"{key}: {type(exc).__name__}: {exc}")
    check("分镜预览可排版（不需要装 manim）", not preview_bad, "; ".join(preview_bad[:3]))

    # 渲染命令
    command = manimgen.render_command("scene.py", "DeterminantScene", "high")
    check("渲染命令带正确的画质旗标", command[-3] == "-qh" and command[-1] == "DeterminantScene",
          " ".join(command[-3:]))
    smoke = manimgen.render_command("scene.py", "X", "low", extra_flags=("-s",))
    check("冒烟渲染旗标 -s 可用", "-s" in smoke, " ".join(smoke))

    available, message = manimgen.manim_status()
    has_latex, latex_message = manimgen.latex_status()
    check("manim / LaTeX 探测不抛异常", isinstance(available, bool) and isinstance(has_latex, bool),
          f"manim={available} latex={has_latex}")


def main() -> int:
    test_all_plot_kinds()
    test_missing_columns_are_safe()
    test_all_matrix_ops()
    test_latex_render_of_all_results()
    test_latex_engine_robustness()
    test_data_layer()
    test_export_roundtrip()
    test_rich_labels_and_fonts()
    test_manim_export()

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
