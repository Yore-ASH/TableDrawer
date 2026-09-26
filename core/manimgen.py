"""把矩阵计算过程导出成 **manim** 动画源码（并可在应用内预览分镜）。

设计取舍
--------
manim 依赖 ``moderngl → glcontext``，在 Windows 上需要 MSVC 编译 C 扩展；
本机（Python 3.14，无 MSVC）装不上。因此本模块做成**源码生成器 + 分镜预览**：

* :func:`plan_scene` 先按真实数据算出每一步（消元、转置、求逆、特征值…），
  产出 :class:`ManimStep` 列表 —— 界面用自研 LaTeX 引擎把它渲染成「分镜脚本」，
  **不需要装 manim 也能看到动画会怎么演**；
* :func:`generate_scene` 把分镜翻译成完整、可直接运行的 manim 场景文件；
* :func:`manim_status` 检测本机有没有 manim，有的话界面直接提供
  「渲染成 MP4」按钮（跑 ``manim`` 子进程）。

消元过程用 :class:`fractions.Fraction` 精确计算，所以生成的画面里是
``\\frac{3}{2}`` 这样的精确分数，而不是 ``1.5000000001``。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Sequence

import numpy as np

from .numberfmt import format_number, to_latex

__all__ = [
    "ManimStep",
    "ManimScenePlan",
    "ManimOptions",
    "SCENES",
    "QUALITIES",
    "INSTALL_HINT",
    "NO_LATEX_HINT",
    "available_scenes",
    "plan_scene",
    "generate_scene",
    "manim_status",
    "manim_executable",
    "latex_status",
    "render_command",
    "run_manim",
]

#: 可生成的动画：``键 -> 中文名``
SCENES: dict[str, str] = {
    "elimination": "高斯消元 → 行简化阶梯形",
    "transpose": "转置 Aᵀ",
    "inverse": "求逆 A⁻¹（增广矩阵法）",
    "determinant": "行列式的展开",
    "multiply": "矩阵乘法 A·B（行×列）",
    "eigen": "特征值 A v = λ v",
}

#: manim 画质档位：``键 -> (中文名, 分辨率, 帧率, 命令行旗标)``
QUALITIES: dict[str, tuple[str, str, int, str]] = {
    "low": ("低（480p15，最快）", "854x480", 15, "l"),
    "medium": ("中（720p30，推荐）", "1280x720", 30, "m"),
    "high": ("高（1080p60）", "1920x1080", 60, "h"),
    "production": ("超高（4K60，很慢）", "3840x2160", 60, "k"),
}

INSTALL_HINT = (
    "未检测到 manim。可以先生成源码与分镜预览，装好后再渲染：\n\n"
    "    python -m pip install manim\n\n"
    "注意 Windows 上 moderngl 依赖的 glcontext 需要 C++ 编译器\n"
    "（Microsoft C++ Build Tools），否则会编译失败。\n"
    "另外若想得到最漂亮的排版，建议再装一个 LaTeX 发行版\n"
    "（MiKTeX 或 TeX Live）；没有 LaTeX 也能渲染，\n"
    "生成的脚本会自动降级成用 Text 拼矩阵。"
)

#: 未装 LaTeX 时的说明
NO_LATEX_HINT = (
    "已检测到 manim，但**没有找到 LaTeX**（latex / pdflatex / xelatex）。\n"
    "生成的脚本会自动降级：矩阵改用 Text 拼出的等价格子，仍然可以正常渲染出视频，\n"
    "只是分数会显示成 3/2 这种写法。装上 MiKTeX 或 TeX Live 后可获得完整排版。"
)

_MAX_STEPS = 60          # 分镜上限，避免超大矩阵生成上千步


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class ManimStep:
    """动画的一帧：一个矩阵（或公式）+ 说明 + 高亮。"""

    rows: list[list[str]] = field(default_factory=list)   # 矩阵内容（字符串，可直接给 manim）
    text_note: str = ""                                   # 中文说明（manim 的 Text）
    math_note: str = ""                                   # 数学注释（manim 的 MathTex）
    highlight_rows: tuple[int, ...] = ()                  # 需要高亮的行（0 基）
    extra_lines: list[str] = field(default_factory=list)  # 该帧附加的展示行

    def latex(self, *, name: str = "A", fractions: bool = True) -> str:
        """本帧的 LaTeX（交给自研引擎做分镜预览）。"""
        if not self.rows:
            return ""
        env = _matrix_latex(self.rows)
        return f"{name} = {env}" if name else env


@dataclass
class ManimScenePlan:
    """一部动画的完整分镜。"""

    key: str
    title: str
    subtitle: str
    scene_name: str
    steps: list[ManimStep] = field(default_factory=list)
    matrix_name: str = "A"
    show_axes: bool = False
    hero_latex: str = ""          # 结尾定格展示的公式


@dataclass
class ManimOptions:
    """生成 manim 源码的选项。"""

    quality: str = "medium"
    background: str = "dark"          # dark | light
    step_duration: float = 1.2        # 每步动画时长
    hold_duration: float = 0.8        # 每步停留时长
    scale: float = 1.15               # 矩阵整体缩放
    show_index: bool = True           # 显示「第 n 步 / 共 N 步」
    loop: bool = False                # 结尾回到第一步
    extra_header: str = ""


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _matrix_latex(rows: Sequence[Sequence[str]]) -> str:
    body = r" \\ ".join(" & ".join(str(cell) for cell in row) for row in rows)
    return r"\begin{pmatrix}" + body + r"\end{pmatrix}"


def _frac_to_latex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    sign = "-" if value.numerator < 0 else ""
    return rf"{sign}\frac{{{abs(value.numerator)}}}{{{value.denominator}}}"


def _to_fractions(matrix: Any) -> list[list[Fraction]]:
    array = np.atleast_2d(np.asarray(matrix, dtype=float))
    out: list[list[Fraction]] = []
    for row in array:
        line: list[Fraction] = []
        for value in row:
            if not np.isfinite(value):
                line.append(Fraction(0))
            else:
                line.append(Fraction(float(value)).limit_denominator(1000))
        out.append(line)
    return out


def _numbers_to_rows(matrix: Any, *, fractions: bool = True) -> list[list[str]]:
    """把数值矩阵转成「每格一个 LaTeX 字符串」的行列表（喂给 manim 的 Matrix）。"""
    array = np.atleast_2d(np.asarray(matrix))
    rows: list[list[str]] = []
    for row in array:
        line: list[str] = []
        for value in row:
            if isinstance(value, Fraction):
                line.append(_frac_to_latex(value) if fractions else format_number(float(value)))
            elif np.iscomplexobj(array) and abs(complex(value).imag) > 1e-12:
                line.append(to_latex(complex(value)))
            else:
                line.append(to_latex(float(np.real(value)), fractions=fractions))
        rows.append(line)
    return rows


def _fracs_to_rows(fractions: Sequence[Sequence[Fraction]]) -> list[list[str]]:
    return [[_frac_to_latex(value) for value in row] for row in fractions]


def _row_operation_latex(kind: str, target: int, source: int | None, factor: Fraction | None) -> str:
    """行变换的数学写法，例如 ``R_2 \\leftarrow R_2 - \\frac{3}{2} R_1``。"""
    target_name = f"R_{{{target + 1}}}"
    if kind == "swap" and source is not None:
        return rf"{target_name} \leftrightarrow R_{{{source + 1}}}"
    if kind == "scale" and factor is not None:
        return rf"{target_name} \leftarrow {_frac_to_latex(factor)} {target_name}"
    if kind == "add" and source is not None and factor is not None:
        if factor == 1:
            return rf"{target_name} \leftarrow {target_name} + R_{{{source + 1}}}"
        if factor == -1:
            return rf"{target_name} \leftarrow {target_name} - R_{{{source + 1}}}"
        sign = "+" if factor > 0 else "-"
        magnitude = _frac_to_latex(abs(factor))
        return rf"{target_name} \leftarrow {target_name} {sign} {magnitude} R_{{{source + 1}}}"
    return target_name


def _row_operation_text(kind: str, target: int, source: int | None, factor: Fraction | None) -> str:
    """行变换的中文说明。"""
    if kind == "swap" and source is not None:
        return f"交换第 {target + 1} 行与第 {source + 1} 行"
    if kind == "scale" and factor is not None:
        return f"第 {target + 1} 行整体除以 {format_number(1 / float(factor))}"
    if kind == "add" and source is not None and factor is not None:
        return f"第 {target + 1} 行减去第 {source + 1} 行的 {format_number(float(abs(factor)))} 倍"
    return f"处理第 {target + 1} 行"


# --------------------------------------------------------------------------- #
# 分镜规划
# --------------------------------------------------------------------------- #
def _plan_elimination(matrix: Any) -> ManimScenePlan:
    """高斯–若尔当消元，逐步记录行变换（精确分数）。"""
    grid = _to_fractions(matrix)
    n_rows = len(grid)
    n_cols = len(grid[0]) if grid else 0
    plan = ManimScenePlan(
        key="elimination",
        title="高斯消元 → 行简化阶梯形",
        subtitle="用初等行变换把矩阵化成 RREF",
        scene_name="GaussianElimination",
        matrix_name="A",
    )
    plan.steps.append(
        ManimStep(rows=_fracs_to_rows(grid), text_note="原始矩阵", math_note="A")
    )

    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows or len(plan.steps) > _MAX_STEPS:
            break
        # 选主元（部分主元法：取绝对值最大的行，数值更稳，也更有教学意义）
        best = max(
            range(pivot_row, n_rows),
            key=lambda r: abs(grid[r][col]) if col < len(grid[r]) else Fraction(0),
        )
        if grid[best][col] == 0:
            continue
        if best != pivot_row:
            grid[pivot_row], grid[best] = grid[best], grid[pivot_row]
            plan.steps.append(
                ManimStep(
                    rows=_fracs_to_rows(grid),
                    text_note=_row_operation_text("swap", pivot_row, best, None),
                    math_note=_row_operation_latex("swap", pivot_row, best, None),
                    highlight_rows=(pivot_row, best),
                )
            )
        # 主元归一
        pivot = grid[pivot_row][col]
        if pivot != 1:
            grid[pivot_row] = [value / pivot for value in grid[pivot_row]]
            plan.steps.append(
                ManimStep(
                    rows=_fracs_to_rows(grid),
                    text_note=_row_operation_text("scale", pivot_row, None, 1 / pivot),
                    math_note=_row_operation_latex("scale", pivot_row, None, 1 / pivot),
                    highlight_rows=(pivot_row,),
                )
            )
        # 其余行消元
        for row in range(n_rows):
            if row == pivot_row:
                continue
            factor = grid[row][col]
            if factor == 0:
                continue
            grid[row] = [a - factor * b for a, b in zip(grid[row], grid[pivot_row])]
            plan.steps.append(
                ManimStep(
                    rows=_fracs_to_rows(grid),
                    text_note=_row_operation_text("add", row, pivot_row, -factor),
                    math_note=_row_operation_latex("add", row, pivot_row, -factor),
                    highlight_rows=(row,),
                )
            )
        pivot_row += 1

    plan.steps.append(
        ManimStep(
            rows=_fracs_to_rows(grid),
            text_note="完成：得到行简化阶梯形",
            math_note=r"\mathrm{RREF}(A)",
        )
    )
    return plan


def _plan_transpose(matrix: Any) -> ManimScenePlan:
    array = np.atleast_2d(np.asarray(matrix, dtype=float))
    plan = ManimScenePlan(
        key="transpose",
        title="转置 Aᵀ",
        subtitle="把第 i 行第 j 列的元素搬到第 j 行第 i 列",
        scene_name="TransposeScene",
    )
    original = _numbers_to_rows(array)
    plan.steps.append(ManimStep(rows=original, text_note="原始矩阵 A", math_note="A"))
    transposed = _numbers_to_rows(array.T)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if len(plan.steps) > _MAX_STEPS:
                break
            plan.steps.append(
                ManimStep(
                    rows=original,
                    text_note=f"a({i + 1},{j + 1}) = {format_number(float(array[i, j]))} 移到 ({j + 1},{i + 1})",
                    math_note=rf"a_{{{i + 1}{j + 1}}} \rightarrow a_{{{j + 1}{i + 1}}}",
                    highlight_rows=(i,),
                )
            )
    plan.steps.append(ManimStep(rows=transposed, text_note="转置完成", math_note=r"A^{\mathrm{T}}"))
    plan.hero_latex = rf"A^{{\mathrm{{T}}}} = {_matrix_latex(transposed)}"
    return plan


def _plan_inverse(matrix: Any) -> ManimScenePlan:
    """增广矩阵 [A | I] 消元成 [I | A⁻¹]。"""
    array = np.atleast_2d(np.asarray(matrix, dtype=float))
    if array.shape[0] != array.shape[1]:
        # 非方阵：给出提示分镜
        plan = ManimScenePlan(
            key="inverse",
            title="求逆 A⁻¹",
            subtitle="只有方阵才谈得上逆矩阵",
            scene_name="InverseScene",
        )
        plan.steps.append(
            ManimStep(rows=_numbers_to_rows(array), text_note="当前矩阵不是方阵，无法求逆")
        )
        return plan

    size = array.shape[0]
    grid = _to_fractions(array)
    identity = [[Fraction(int(i == j)) for j in range(size)] for i in range(size)]
    augmented = [grid[i] + identity[i] for i in range(size)]

    plan = ManimScenePlan(
        key="inverse",
        title="求逆 A⁻¹（增广矩阵法）",
        subtitle="把 [A | I] 用行变换化成 [I | A⁻¹]",
        scene_name="InverseScene",
    )
    plan.steps.append(
        ManimStep(rows=_fracs_to_rows(augmented), text_note="构造增广矩阵 [A | I]", math_note=r"[\,A \mid I\,]")
    )

    pivot_row = 0
    for col in range(size):
        if len(plan.steps) > _MAX_STEPS:
            break
        best = max(range(pivot_row, size), key=lambda r: abs(augmented[r][col]))
        if augmented[best][col] == 0:
            continue
        if best != pivot_row:
            augmented[pivot_row], augmented[best] = augmented[best], augmented[pivot_row]
            plan.steps.append(
                ManimStep(
                    rows=_fracs_to_rows(augmented),
                    text_note=_row_operation_text("swap", pivot_row, best, None),
                    math_note=_row_operation_latex("swap", pivot_row, best, None),
                    highlight_rows=(pivot_row, best),
                )
            )
        pivot = augmented[pivot_row][col]
        if pivot != 1:
            augmented[pivot_row] = [value / pivot for value in augmented[pivot_row]]
            plan.steps.append(
                ManimStep(
                    rows=_fracs_to_rows(augmented),
                    text_note=_row_operation_text("scale", pivot_row, None, 1 / pivot),
                    math_note=_row_operation_latex("scale", pivot_row, None, 1 / pivot),
                    highlight_rows=(pivot_row,),
                )
            )
        for row in range(size):
            if row == pivot_row:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[pivot_row])]
            plan.steps.append(
                ManimStep(
                    rows=_fracs_to_rows(augmented),
                    text_note=_row_operation_text("add", row, pivot_row, -factor),
                    math_note=_row_operation_latex("add", row, pivot_row, -factor),
                    highlight_rows=(row,),
                )
            )
        pivot_row += 1

    left = [row[:size] for row in augmented]
    if left == identity:
        inverse = [row[size:] for row in augmented]
        plan.steps.append(
            ManimStep(rows=_fracs_to_rows(inverse), text_note="右半边就是逆矩阵", math_note=r"A^{-1}")
        )
        plan.hero_latex = rf"A^{{-1}} = {_matrix_latex(_fracs_to_rows(inverse))}"
    else:
        plan.steps.append(
            ManimStep(rows=_fracs_to_rows(augmented), text_note="左半边无法化成单位阵 ⇒ A 不可逆（奇异矩阵）")
        )
    return plan


def _plan_determinant(matrix: Any) -> ManimScenePlan:
    array = np.atleast_2d(np.asarray(matrix, dtype=float))
    plan = ManimScenePlan(
        key="determinant",
        title="行列式的展开",
        subtitle="按第一行代数余子式展开",
        scene_name="DeterminantScene",
    )
    rows = _numbers_to_rows(array)
    plan.steps.append(ManimStep(rows=rows, text_note="原始矩阵", math_note="A"))

    if array.shape[0] == array.shape[1] and array.shape[0] == 2:
        a, b = array[0]
        c, d = array[1]
        plan.steps.append(
            ManimStep(rows=rows, text_note="二阶行列式：主对角线之积减去副对角线之积",
                      math_note=rf"\det(A) = {to_latex(a)} \cdot {to_latex(d)} - {to_latex(b)} \cdot {to_latex(c)}")
        )
    elif array.shape[0] == array.shape[1]:
        size = array.shape[0]
        terms = []
        for j in range(size):
            minor = np.delete(np.delete(array, 0, axis=0), j, axis=1)
            minor_det = float(np.linalg.det(minor)) if minor.size else 1.0
            cofactor = ((-1) ** j) * array[0, j]
            terms.append(f"{to_latex(cofactor)} \\cdot {to_latex(minor_det)}")
        plan.steps.append(
            ManimStep(rows=rows, text_note=f"按第一行展开成 {size} 项代数余子式",
                      math_note=r"\det(A) = " + " + ".join(terms))
        )

    try:
        value = float(np.linalg.det(array))
        plan.steps.append(ManimStep(rows=rows, text_note="行列式的值", math_note=rf"\det(A) = {to_latex(value)}"))
        plan.hero_latex = rf"\det(A) = {to_latex(value)}"
    except Exception:
        pass
    return plan


def _plan_multiply(matrix: Any, other: Any) -> ManimScenePlan:
    left = np.atleast_2d(np.asarray(matrix, dtype=float))
    right = np.atleast_2d(np.asarray(other, dtype=float))
    plan = ManimScenePlan(
        key="multiply",
        title="矩阵乘法 A·B",
        subtitle="结果的第 i 行第 j 列 = A 的第 i 行与 B 的第 j 列的内积",
        scene_name="MatrixMultiplyScene",
    )
    plan.steps.append(ManimStep(rows=_numbers_to_rows(left), text_note="左矩阵 A", math_note="A"))
    plan.steps.append(ManimStep(rows=_numbers_to_rows(right), text_note="右矩阵 B", math_note="B"))

    if left.shape[1] != right.shape[0]:
        plan.steps.append(
            ManimStep(rows=_numbers_to_rows(left), text_note="两矩阵维度不匹配，无法相乘")
        )
        return plan

    result = left @ right
    for i in range(left.shape[0]):
        for j in range(right.shape[1]):
            if len(plan.steps) > _MAX_STEPS:
                break
            terms = " + ".join(
                f"{to_latex(left[i, k])} \\cdot {to_latex(right[k, j])}" for k in range(left.shape[1])
            )
            plan.steps.append(
                ManimStep(
                    rows=_numbers_to_rows(result),
                    text_note=f"计算结果的第 {i + 1} 行第 {j + 1} 列",
                    math_note=rf"c_{{{i + 1}{j + 1}}} = {terms} = {to_latex(result[i, j])}",
                    highlight_rows=(i,),
                )
            )
    plan.steps.append(ManimStep(rows=_numbers_to_rows(result), text_note="乘法完成", math_note=r"A\,B"))
    plan.hero_latex = rf"A\,B = {_matrix_latex(_numbers_to_rows(result))}"
    return plan


def _plan_eigen(matrix: Any) -> ManimScenePlan:
    array = np.atleast_2d(np.asarray(matrix, dtype=float))
    plan = ManimScenePlan(
        key="eigen",
        title="特征值 A v = λ v",
        subtitle="特征向量被 A 作用后只改变长度、不改变方向",
        scene_name="EigenScene",
    )
    plan.steps.append(ManimStep(rows=_numbers_to_rows(array), text_note="原始矩阵", math_note="A"))

    if array.shape[0] != array.shape[1]:
        plan.steps.append(ManimStep(rows=_numbers_to_rows(array), text_note="非方阵没有特征值"))
        return plan

    try:
        values, vectors = np.linalg.eig(array)
    except Exception:
        return plan

    order = np.argsort(-np.abs(values))
    for rank, index in enumerate(order):
        lam = values[index]
        vec = vectors[:, index]
        lam_text = to_latex(complex(lam))
        vec_rows = [[to_latex(complex(component))] for component in vec]
        plan.steps.append(
            ManimStep(
                rows=_numbers_to_rows(array),
                text_note=f"第 {rank + 1} 个特征对：特征向量被 A 作用后只伸缩 {format_number(abs(complex(lam)))} 倍",
                math_note=rf"\lambda_{{{rank + 1}}} = {lam_text}",
                extra_lines=[rf"v_{{{rank + 1}}} = {_matrix_latex(vec_rows)}"],
            )
        )
    plan.steps.append(
        ManimStep(
            rows=_numbers_to_rows(array),
            text_note="所有特征值按模长从大到小排列",
            math_note=r"\lambda_i",
            extra_lines=[
                r"\lambda_{" + str(i + 1) + r"} = " + to_latex(complex(values[order[i]]))
                for i in range(len(order))
            ],
        )
    )
    # 实特征值的 2x2 情形可以画出向量被拉伸的过程
    if array.shape == (2, 2) and np.allclose(values.imag, 0):
        plan.show_axes = True
    return plan


def plan_scene(kind: str, matrix: Any, other: Any = None) -> ManimScenePlan:
    """按真实数据算出整部动画的分镜。"""
    if kind == "elimination":
        return _plan_elimination(matrix)
    if kind == "transpose":
        return _plan_transpose(matrix)
    if kind == "inverse":
        return _plan_inverse(matrix)
    if kind == "determinant":
        return _plan_determinant(matrix)
    if kind == "multiply":
        return _plan_multiply(matrix, other if other is not None else matrix)
    if kind == "eigen":
        return _plan_eigen(matrix)
    raise ValueError(f"未知的动画类型：{kind}")


def available_scenes() -> dict[str, str]:
    """可用的动画类型。"""
    return dict(SCENES)


# --------------------------------------------------------------------------- #
# manim 源码生成
# --------------------------------------------------------------------------- #
def _py_repr_rows(rows: Sequence[Sequence[str]]) -> str:
    """把矩阵行渲染成 Python 字面量（每行一个 LaTeX 字符串）。"""
    parts = []
    for row in rows:
        cells = ", ".join(repr(str(cell)) for cell in row)
        parts.append(f"[{cells}]")
    return "[" + ", ".join(parts) + "]"


_SCENE_TEMPLATE = '''# -*- coding: utf-8 -*-
"""由 TableDrawer 生成的 manim 动画：$title

$subtitle

运行方式：
    manim -q$quality_flag -p $scene_file $scene_name
    # -p 渲染后自动播放；生成时选择的画质是「$quality_label」

兼容性说明（重要）：
  · 若本机装了 LaTeX 发行版（MiKTeX / TeX Live），矩阵用 manim 的 ``Matrix``、
    公式用 ``MathTex``，排版最漂亮；
  · 若**没装 LaTeX**，脚本会自动降级成用 ``Text`` 拼出的等价格子，
    依旧能正常渲染出视频（只是分数会显示成 ``3/2`` 这种写法）；
  · 中文说明一律用 ``Text``（走 Pango，不需要 LaTeX 支持）。
  · 每一步的矩阵与公式都是生成时按**真实计算**得到的，不是模板占位符。
"""
import re
import shutil

from manim import *

#: 本机是否有 LaTeX（决定用 MathTex 还是 Text 降级）
HAS_LATEX = bool(
    shutil.which("latex") or shutil.which("pdflatex") or shutil.which("xelatex")
)

_PLAIN_REPLACEMENTS = (
    (r"\\circ", "\\u2218"), (r"\\otimes", "\\u2297"), (r"\\times", "\\u00d7"),
    (r"\\cdot", "\\u00b7"), (r"\\approx", "\\u2248"), (r"\\leftarrow", "\\u2190"),
    (r"\\leftrightarrow", "\\u2194"), (r"\\lambda", "\\u03bb"), (r"\\sigma", "\\u03c3"),
    (r"\\rho", "\\u03c1"), (r"\\theta", "\\u03b8"), (r"\\infty", "\\u221e"),
)


def to_plain(latex: str) -> str:
    """把简单 LaTeX 降级成可读文本（无 LaTeX 环境时用）。"""
    text = str(latex)
    for command, char in _PLAIN_REPLACEMENTS:
        text = text.replace(command, char)
    text = re.sub(r"\\\\frac\\{([^{}]*)\\}\\{([^{}]*)\\}", r"\\1/\\2", text)
    text = re.sub(r"\\\\(?:mathrm|operatorname|mathbf|text)\\{([^{}]*)\\}", r"\\1", text)
    text = re.sub(r"\\^\\{([^{}]*)\\}", r"^\\1", text)
    text = re.sub(r"_\\{([^{}]*)\\}", r"_\\1", text)
    text = text.replace("\\\\", " ").replace("{", "").replace("}", "")
    return text.strip()


def make_math(latex: str, font_size: float = 34):
    """公式：优先 MathTex，没有 LaTeX 就退回 Text。"""
    if HAS_LATEX:
        try:
            return MathTex(latex, font_size=font_size)
        except Exception:
            pass
    return Text(to_plain(latex), font_size=font_size * 0.8)


def make_matrix(rows, font_size: float = 38, h_buff: float = 0.9, v_buff: float = 0.55):
    """矩阵：优先 manim 的 Matrix，没有 LaTeX 就用 Text 拼一个等价格子。"""
    if HAS_LATEX:
        try:
            return Matrix(
                rows,
                h_buff=h_buff,
                v_buff=v_buff,
                element_to_mobject_config={"font_size": font_size},
            )
        except Exception:
            pass
    grid = VGroup(
        *[
            VGroup(*[Text(to_plain(cell), font_size=font_size * 0.8) for cell in row]).arrange(
                RIGHT, buff=h_buff
            )
            for row in rows
        ]
    ).arrange(DOWN, buff=v_buff)
    try:
        body = Bracketed(grid, buff=0.25)
    except Exception:
        body = grid
    return FallbackMatrix(grid, body)


class FallbackMatrix(VGroup):
    """没有 LaTeX 时用 Text 拼出的矩阵。

    依旧提供 ``get_rows()``，这样「高亮某一行」的动画在降级模式下同样有效。
    """

    def __init__(self, grid, body, **kwargs):
        super().__init__(**kwargs)
        self._grid = grid
        self.add(body)

    def get_rows(self):
        return list(self._grid)


class $scene_name(Scene):
    """$title"""

    def construct(self):
        self.camera.background_color = $background

        title = Text($title_repr, font_size=34).to_edge(UP)
        subtitle = Text($subtitle_repr, font_size=20, color=GRAY).next_to(title, DOWN, buff=0.18)
        self.play(Write(title), FadeIn(subtitle, shift=DOWN * 0.2))
        self.wait(0.4)
        self.play(FadeOut(subtitle))

$steps_code

        total = len(steps)
        current = None
        for index, step in enumerate(steps):
            matrix = make_matrix(step["rows"]).scale($scale)

            notes = VGroup()
            if step.get("math"):
                notes.add(make_math(step["math"], font_size=32))
            if step.get("note"):
                notes.add(Text(step["note"], font_size=22, color=YELLOW))
            if len(notes):
                notes.arrange(DOWN, buff=0.22).next_to(matrix, DOWN, buff=0.42)

            group = VGroup(matrix, notes)
            extra = VGroup()
            for line in step.get("extra", []):
                extra.add(make_math(line, font_size=28))
            if len(extra):
                extra.arrange(DOWN, buff=0.2).to_edge(RIGHT).shift(LEFT * 0.6)
                group.add(extra)

            group.move_to(ORIGIN).shift(DOWN * 0.35)
$counter_code
            if current is None:
                self.play(FadeIn(group, shift=UP * 0.3), FadeIn(counter, shift=UP * 0.3))
            else:
                self.play(ReplacementTransform(current, group), FadeIn(counter))

            # 高亮本次操作涉及的行
            rows = matrix.get_rows() if hasattr(matrix, "get_rows") else []
            for row_index in step.get("rows_hl", []):
                if 0 <= row_index < len(rows):
                    self.play(rows[row_index].animate.set_color(YELLOW), run_time=0.22)
                    self.play(rows[row_index].animate.set_color(WHITE), run_time=0.22)

            self.wait($hold)
            current = group
$tail
'''


def _emit_steps(plan: ManimScenePlan) -> str:
    lines: list[str] = ["        steps = ["]
    for step in plan.steps:
        lines.append("            {")
        lines.append(f"                'rows': {_py_repr_rows(step.rows)},")
        if step.text_note:
            lines.append(f"                'note': {step.text_note!r},")
        if step.math_note:
            lines.append(f"                'math': {step.math_note!r},")
        if step.highlight_rows:
            lines.append(f"                'rows_hl': {list(step.highlight_rows)!r},")
        if step.extra_lines:
            lines.append(f"                'extra': {list(step.extra_lines)!r},")
        lines.append("            },")
    lines.append("        ]")
    return "\n".join(lines)


def generate_scene(
    plan: ManimScenePlan,
    options: ManimOptions | None = None,
    *,
    source_matrix: Any = None,
) -> str:
    """把分镜翻译成完整、可直接运行的 manim 源码。

    生成的脚本会**自行探测 LaTeX**：有就用 ``Matrix`` / ``MathTex`` 得到最佳排版，
    没有就降级成 ``Text`` 拼的等价格子，因此在没装 LaTeX 的机器上同样能渲染出视频。
    """
    from string import Template

    options = options or ManimOptions()
    quality_label = QUALITIES.get(options.quality, QUALITIES["medium"])[0]
    quality_flag = QUALITIES.get(options.quality, QUALITIES["medium"])[3]
    # manim 没有 DARK / LIGHT 这两个颜色常量，必须直接给颜色值
    background = 'ManimColor("#1c1c1e")' if options.background == "dark" else 'ManimColor("#ffffff")'

    counter_code = (
        '            counter = Text(f"{index + 1} / {total}", font_size=20, color=GRAY).to_corner(DL)'
        if options.show_index
        else "            counter = VGroup()"
    )

    tail_parts = ["        self.wait(1.5)"]
    if options.loop:
        tail_parts.append(
            "        # 回到第一步循环播放\n"
            "        first = make_matrix(steps[0]['rows']).scale(1.15)\n"
            "        first.move_to(ORIGIN).shift(DOWN * 0.35)\n"
            "        self.play(ReplacementTransform(current, first))\n"
            "        self.wait(1)"
        )
    if plan.hero_latex:
        tail_parts.append(
            "        # 结尾定格展示关键结论\n"
            "        try:\n"
            f"            hero = make_math(r'{plan.hero_latex}', font_size=42)\n"
            "            box = SurroundingRectangle(hero, color=GREEN, buff=0.3)\n"
            "            self.play(FadeOut(current), Write(hero), Create(box))\n"
            "            self.wait(2)\n"
            "        except Exception:\n"
            "            pass"
        )

    template = Template(_SCENE_TEMPLATE)
    return template.substitute(
        title=plan.title,
        title_repr=repr(plan.title),
        subtitle=plan.subtitle,
        subtitle_repr=repr(plan.subtitle),
        scene_name=plan.scene_name,
        scene_file=plan.scene_name.lower() + ".py",
        quality_flag=quality_flag,
        quality_label=quality_label,
        background=background,
        steps_code=_emit_steps(plan),
        scale=f"{options.scale:g}",
        hold=f"{options.hold_duration:g}",
        counter_code=counter_code,
        tail="\n".join(tail_parts),
    )


# --------------------------------------------------------------------------- #
# manim 可用性检测与渲染
# --------------------------------------------------------------------------- #
def latex_status() -> tuple[bool, str]:
    """检测本机 LaTeX 发行版，返回 ``(可用, 说明)``。"""
    for name in ("latex", "pdflatex", "xelatex"):
        path = shutil.which(name)
        if path:
            return True, f"已检测到 LaTeX：{path}"
    return False, "未找到 LaTeX（latex / pdflatex / xelatex）"


def manim_status() -> tuple[bool, str]:
    """检测本机 manim 是否可用，返回 ``(可用, 说明)``。

    会同时报告 LaTeX 的情况：没有 LaTeX 时生成的脚本会自动降级用 Text 拼矩阵。
    """
    version = ""
    try:
        import manim  # noqa: F401

        version = f"manim {getattr(manim, '__version__', '')}".strip()
    except Exception:
        executable = shutil.which("manim")
        if not executable:
            return False, INSTALL_HINT
        version = f"manim 命令：{executable}"

    has_latex, latex_note = latex_status()
    if has_latex:
        return True, f"{version}；{latex_note}（矩阵与公式将使用完整 LaTeX 排版）"
    return True, f"{version}；{latex_note}（将自动降级为 Text 矩阵，仍可渲染）"


def manim_executable() -> list[str]:
    """返回调用 manim 的命令前缀（``['python','-m','manim']`` 或 ``['manim']``）。"""
    try:
        import manim  # noqa: F401

        return [sys.executable, "-m", "manim"]
    except Exception:
        pass
    executable = shutil.which("manim")
    if executable:
        return [executable]
    return [sys.executable, "-m", "manim"]


def render_command(
    script_path: str,
    scene_name: str,
    quality: str = "medium",
    *,
    extra_flags: Sequence[str] = (),
) -> list[str]:
    """构造渲染命令（优先当前解释器的 ``-m manim``，找不到再退回 ``manim`` 命令）。

    ``extra_flags`` 可以传入 ``("-s",)`` 只渲染最后一帧——做冒烟测试时比完整
    渲染快得多，而且同样会完整执行 ``construct()``，能抓到运行期错误。
    """
    flag = QUALITIES.get(quality, QUALITIES["medium"])[3]
    return [*manim_executable(), f"-q{flag}", *extra_flags, script_path, scene_name]


def run_manim(
    script_path: str,
    scene_name: str,
    quality: str = "medium",
    *,
    cwd: str | None = None,
    timeout: int = 900,
    extra_flags: Sequence[str] = (),
) -> subprocess.CompletedProcess:
    """调用 manim 渲染（同步；界面应放到后台线程里跑）。

    返回 :class:`subprocess.CompletedProcess`；stdout/stderr 都是文本。
    """
    command = render_command(script_path, scene_name, quality, extra_flags=extra_flags)
    return subprocess.run(
        command,
        cwd=cwd or os.path.dirname(os.path.abspath(script_path)) or None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
