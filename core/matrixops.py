"""矩阵 / 向量组计算引擎。

本模块把 NumPy 的线性代数能力包装成界面层可以直接消费的 :class:`ResultSheet`：

* :data:`REGISTRY` —— 所有可执行运算的描述表（界面据此自动生成分组按钮）
* :func:`run`      —— 统一的执行入口，**永不抛出异常**

数据契约来自 :mod:`core.spec`（``ResultItem`` / ``ResultSheet``），
数值 → 文本 / LaTeX 的转换全部走 :mod:`core.numberfmt`，本模块不自行格式化数字。

唯一的例外是复数：``numberfmt`` 目前的实现对纯复标量存在缺陷
（``format_number`` 与 ``format_value`` 互相递归；``snap_zero`` 与
``matrix_to_latex`` 会通过 ``float()`` 丢掉虚部），因此复数元素由本模块
调用 ``to_latex`` 直接渲染，其余情形仍一律使用 ``numberfmt`` 的接口。

用法示例::

    from core import matrixops

    sheet = matrixops.run("determinant", A=[[1, 2], [3, 4]])
    print(sheet.plain_text())

    for spec in matrixops.REGISTRY:          # 界面据此生成按钮
        print(spec.group, spec.key, spec.label)

    matrixops.run("scalar_multiply", A=[[1, 2], [3, 4]], scalar=3)
    matrixops.run("identity", scalar=4)
    matrixops.run("angle", A=[[1, 0]], B=[[0, 1]])

约定：

* ``vector_input=True`` 的运算把 ``A`` 的每一**行**当作一个向量；``A`` 为一维时
  整体视为单个向量。两个向量的运算取 ``A`` 的第一行与 ``B`` 的第一行。
* 生成类运算（``arity=0``）不需要 ``A``，结果矩阵放在 ``ResultItem.payload`` 中。
* 所有 ``latex`` 字段都包含等号左边的表达式，且不含中文；中文只出现在
  ``title`` / ``note`` 字段。
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .numberfmt import (
    format_number,
    matrix_to_latex,
    matrix_to_text,
    snap_zero,
    to_latex,
    vector_to_latex,
)

__all__ = [
    "OperationSpec",
    "REGISTRY",
    "MatrixOpError",
    "run",
    "spec_for",
    "grouped_registry",
    "GROUPS",
]

#: 允许的分组名（顺序即界面展示顺序）
GROUPS: tuple[str, ...] = (
    "生成矩阵",
    "基本运算",
    "矩阵乘法",
    "行列式与秩",
    "逆与广义逆",
    "特征值与特征向量",
    "矩阵分解",
    "范数与条件数",
    "线性方程组",
    "向量组",
)

#: 判定零元素的相对容差
_TOL = 1e-9

#: 允许的输入元素量级上限，超过后 det / matmul / power 等会溢出成无穷大
_MAX_SAFE_MAGNITUDE = 1e150


# --------------------------------------------------------------------------- #
# 运算描述
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OperationSpec:
    """一条可执行运算的描述。

    ``key`` 唯一标识运算并用于 :func:`run`；``arity`` 表示需要几个矩阵输入
    （0 表示不需要矩阵，1 表示只需 ``A``，2 表示需要 ``A`` 与 ``B``）；
    ``vector_input=True`` 表示把 ``A`` 的每一行当作一个向量。
    """

    key: str
    label: str
    group: str
    arity: int = 1
    scalar: bool = False
    vector_input: bool = False
    description: str = ""
    latex_hint: str = ""


class MatrixOpError(Exception):
    """运算无法完成时抛出，携带面向用户的中文原因。"""


# --------------------------------------------------------------------------- #
# 输入解析
# --------------------------------------------------------------------------- #
def _is_missing(value: Any) -> bool:
    """判断输入是否为空（``None`` / 空序列 / 全是 ``None`` 的嵌套序列）。"""
    if value is None:
        return True
    if isinstance(value, (str, bytes)):
        return not value.strip()
    arr = np.asarray(value, dtype=object) if not isinstance(value, np.ndarray) else value
    if arr.size == 0:
        return True
    if arr.dtype == object:
        for cell in arr.reshape(-1):
            if cell is None or (isinstance(cell, str) and not cell.strip()):
                continue
            return False
        return True
    return False


def _to_array(value: Any, name: str) -> np.ndarray:
    """把任意输入安全转换为二维数值数组（复数输入保留复数）。"""
    if _is_missing(value):
        raise MatrixOpError(f"没有提供矩阵 {name}。")
    try:
        raw = np.asarray(value)
    except Exception as exc:  # pragma: no cover - numpy 极少失败
        raise MatrixOpError(f"矩阵 {name} 无法解析为数值：{exc}") from exc

    if raw.dtype.kind in "OUSV":
        try:
            raw = raw.astype(np.complex128) if _looks_complex(raw) else raw.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise MatrixOpError(f"矩阵 {name} 含有无法转换为数值的元素（{exc}）。") from exc

    try:
        arr = np.asarray(raw, dtype=np.complex128) if raw.dtype.kind == "c" else np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MatrixOpError(f"矩阵 {name} 含有无法转换为数值的元素（{exc}）。") from exc

    if not np.all(np.isfinite(arr)):
        raise MatrixOpError(f"矩阵 {name} 含有 NaN 或无穷大，请先清理数据。")
    if _has_overflow_risk(arr):
        raise MatrixOpError(
            f"矩阵 {name} 的元素量级过大（|元素| > {_MAX_SAFE_MAGNITUDE:.0e}），"
            "行列式、矩阵乘法等运算会溢出为无穷大，请先缩放数据。"
        )
    if arr.ndim > 2:
        raise MatrixOpError(f"矩阵 {name} 是 {arr.ndim} 维数组，本模块只处理一维向量或二维矩阵。")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise MatrixOpError(f"矩阵 {name} 是空的（形状 {arr.shape}）。")
    return arr


def _looks_complex(arr: np.ndarray) -> bool:
    """对象数组是否含复数字面量。"""
    for cell in arr.reshape(-1):
        if isinstance(cell, (complex, np.complexfloating)):
            return True
    return False


def _has_overflow_risk(arr: np.ndarray) -> bool:
    """元素量级是否大到足以让常见运算（det / matmul / power）溢出为无穷大。"""
    if arr.size == 0:
        return False
    return bool(np.max(np.abs(arr)) > _MAX_SAFE_MAGNITUDE)


def _as_real(arr: np.ndarray, name: str, what: str) -> np.ndarray:
    """要求数组是实矩阵，复矩阵在实数域运算下给出中文提示。"""
    if np.iscomplexobj(arr):
        if float(np.max(np.abs(arr.imag))) > _TOL:
            raise MatrixOpError(f"{what}要求矩阵 {name} 是实矩阵，但它含有非零虚部。")
        return arr.real.astype(np.float64)
    return arr.astype(np.float64)


def _as_square(arr: np.ndarray, name: str) -> np.ndarray:
    """要求方阵。"""
    if arr.shape[0] != arr.shape[1]:
        raise MatrixOpError(f"该运算要求方阵，但矩阵 {name} 的形状是 {arr.shape[0]}×{arr.shape[1]}。")
    return arr


def _as_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    """要求两个矩阵形状一致。"""
    if a.shape != b.shape:
        raise MatrixOpError(
            f"两个矩阵的形状必须相同，但 A 是 {a.shape[0]}×{a.shape[1]}，B 是 {b.shape[0]}×{b.shape[1]}。"
        )


def _as_scalar(scalar: Any, default: float, name: str = "标量 k") -> float:
    """解析标量参数。"""
    if _is_missing(scalar):
        return float(default)
    if isinstance(scalar, (list, tuple, np.ndarray)):
        flat = np.asarray(scalar, dtype=object).reshape(-1)
        if flat.size == 0:
            return float(default)
        scalar = flat[0]
    try:
        value = float(scalar)
    except (TypeError, ValueError) as exc:
        raise MatrixOpError(f"{name} 必须是数值，收到的是 {scalar!r}。") from exc
    if not math.isfinite(value):
        raise MatrixOpError(f"{name} 必须是有限数值。")
    return value


def _as_int(scalar: Any, default: int, name: str, *, minimum: int | None = None) -> int:
    """解析整数参数。"""
    value = _as_scalar(scalar, default, name)
    if abs(value - round(value)) > 1e-9:
        raise MatrixOpError(f"{name} 必须是整数，收到的是 {format_number(value)}。")
    out = int(round(value))
    if minimum is not None and out < minimum:
        raise MatrixOpError(f"{name} 必须不小于 {minimum}，收到的是 {out}。")
    return out


# --------------------------------------------------------------------------- #
# 数值整理
# --------------------------------------------------------------------------- #
def _plain(value: Any) -> str:
    """任意数值 → 纯文本。

    ``numberfmt.format_number`` 无法处理纯复标量（会无限递归），因此复数在这里
    改用 :func:`numberfmt.to_latex`；其余情况仍然走 numberfmt 的正式入口。
    """
    if isinstance(value, (bool, np.bool_)):
        return format_number(value)
    if isinstance(value, (complex, np.complexfloating)) and complex(value).imag != 0.0:
        return _complex_plain(value)
    return format_number(value)


def _clean(value: Any) -> np.ndarray:
    """消除浮点噪声：把 ``1e-17`` 级别的元素吸附为精确 0。"""
    arr = np.asarray(value)
    if arr.dtype.kind == "c":
        scale = float(np.max(np.abs(arr))) if arr.size else 1.0
        real = np.vectorize(lambda v: snap_zero(v.real, scale=scale))(arr)
        imag = np.vectorize(lambda v: snap_zero(v.imag, scale=scale))(arr)
        out = real + 1j * imag
        return out.real.astype(np.float64) if np.all(imag == 0.0) else out
    if arr.dtype.kind in "fc":
        arr = arr.astype(np.float64)
        scale = float(np.max(np.abs(arr))) if arr.size else 1.0
        return np.vectorize(lambda v: snap_zero(v, scale=scale))(arr)
    return arr


def _clean_scalar(value: Any, scale: float = 1.0) -> float:
    """标量版的 :func:`_clean`：实数返回 ``float``，复数保留虚部。"""
    if isinstance(value, (complex, np.complexfloating)):
        real = snap_zero(float(np.real(value)), scale=scale)
        imag = snap_zero(float(np.imag(value)), scale=scale)
        if imag == 0.0:
            return real
        return complex(real, imag)
    return snap_zero(float(np.real(value)), scale=scale)


def _display_snap(arr: np.ndarray) -> np.ndarray:
    """把接近整数的元素吸附为精确整数，仅用于显示。

    数值算法会留下 ``0.9999999999999997`` 这类噪声（例如 ``np.roots`` 求出的
    特征值），显示时吸附成 ``1`` 更可读；``payload`` 仍保留原始精度。
    """
    snapped = np.array(arr, copy=True)
    if snapped.dtype.kind == "c":
        for idx in np.ndindex(snapped.shape):
            value = complex(snapped[idx])
            snapped[idx] = complex(_snap_integer(value.real), _snap_integer(value.imag))
        return snapped
    if snapped.dtype.kind == "f":
        flat = snapped.reshape(-1)
        for i in range(flat.size):
            flat[i] = _snap_integer(float(flat[i]))
    return snapped


def _snap_integer(value: float) -> float:
    """``value`` 距最近整数不超过相对容差 1e-10 时返回该整数。"""
    nearest = round(value)
    if nearest != 0 and abs(value - nearest) <= 1e-10 * abs(nearest):
        return float(nearest)
    return value


def _real_if_close(arr: np.ndarray) -> np.ndarray:
    """虚部可忽略时压缩为实数组（空数组原样返回）。"""
    if arr.size and np.iscomplexobj(arr) and float(np.max(np.abs(arr.imag))) <= 1e-10:
        return arr.real
    return arr


def _is_symmetric(arr: np.ndarray, tol: float = 1e-9) -> bool:
    """判断矩阵是否（数值上）对称。"""
    if arr.shape[0] != arr.shape[1]:
        return False
    return bool(np.max(np.abs(arr - arr.T)) <= tol * max(1.0, float(np.max(np.abs(arr)))))


def _sign_canonical(vec: np.ndarray) -> np.ndarray:
    """符号归一化：让向量中绝对值最大的分量取正号，便于稳定显示。"""
    flat = np.asarray(vec).reshape(-1)
    if flat.size == 0:
        return flat
    idx = int(np.argmax(np.abs(flat)))
    pivot = flat[idx]
    if pivot == 0:
        return flat
    phase = pivot / abs(pivot)
    return flat / phase


def _canonicalize_columns(mat: np.ndarray) -> np.ndarray:
    """对矩阵的每一列分别做符号归一化。"""
    arr = np.array(mat, copy=True)
    for j in range(arr.shape[1]):
        arr[:, j] = _sign_canonical(arr[:, j])
    return arr


def _sort_by_magnitude(values: np.ndarray, vectors: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray | None]:
    """按 |λ| 降序排列（同幅值时按实部、虚部稳定排序）。"""
    flat = np.asarray(values).reshape(-1)
    keys = np.lexsort((np.imag(flat), np.real(flat), -np.abs(flat)))
    out_values = flat[keys]
    if vectors is None:
        return out_values, None
    arr = np.asarray(vectors)
    if arr.ndim == 1:
        return out_values, arr[keys]
    return out_values, arr[:, keys]


def _matrix_rank(arr: np.ndarray, tol: float | None = None) -> int:
    """用奇异值判定数值秩。"""
    return int(np.linalg.matrix_rank(arr, tol=tol))


def _nullspace_basis(arr: np.ndarray, tol: float | None = None) -> np.ndarray:
    """返回零空间的一组标准正交基（列向量组成的矩阵，可能为 0 列）。"""
    if arr.size == 0:
        return np.zeros((arr.shape[1], 0))
    u, s, vh = np.linalg.svd(arr)
    if tol is None:
        tol = float(max(arr.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 1.0))
    rank = int(np.sum(s > tol))
    basis = vh[rank:].conj().T
    return _real_if_close(basis)


def _row_basis(arr: np.ndarray, tol: float | None = None) -> np.ndarray:
    """返回行空间的一组标准正交基（作为矩阵的行）。"""
    if arr.size == 0:
        return np.zeros((0, arr.shape[1]))
    u, s, vh = np.linalg.svd(arr)
    if tol is None:
        tol = float(max(arr.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 1.0))
    rank = int(np.sum(s > tol))
    return _real_if_close(vh[:rank])


def _colspace_basis(arr: np.ndarray, tol: float | None = None) -> np.ndarray:
    """返回列空间的一组标准正交基（作为矩阵的列）。"""
    if arr.size == 0:
        return np.zeros((arr.shape[0], 0))
    u, s, _ = np.linalg.svd(arr)
    if tol is None:
        tol = float(max(arr.shape) * np.finfo(np.float64).eps * (s[0] if s.size else 1.0))
    rank = int(np.sum(s > tol))
    return _real_if_close(u[:, :rank])


def _rref(arr: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """高斯–若尔当消元求行简化阶梯形，返回 ``(RREF, 主元列下标)``。"""
    work = np.array(arr, dtype=np.complex128 if np.iscomplexobj(arr) else np.float64)
    rows, cols = work.shape
    pivots: list[int] = []
    r = 0
    scale = max(1.0, float(np.max(np.abs(work))) if work.size else 1.0)
    tol = 1e-10 * scale
    for c in range(cols):
        if r >= rows:
            break
        candidates = np.abs(work[r:, c])
        k = int(np.argmax(candidates))
        if candidates[k] <= tol:
            work[r:, c] = 0
            continue
        if r + k != r:
            work[[r, r + k]] = work[[r + k, r]]
        work[r] = work[r] / work[r, c]
        for i in range(rows):
            if i != r and abs(work[i, c]) > tol:
                work[i] = work[i] - work[i, c] * work[r]
        work[np.abs(work) <= tol] = 0
        pivots.append(c)
        r += 1
    return _real_if_close(work), pivots


def _poly_str(coeffs: np.ndarray) -> str:
    """把多项式系数（降幂）写成 LaTeX 字符串。"""
    terms: list[str] = []
    degree = len(coeffs) - 1
    for i, raw in enumerate(coeffs):
        power = degree - i
        value = complex(raw)
        if abs(value) <= 1e-12:
            continue
        magnitude = abs(value)
        sign = "-" if value.real < 0 or (value.real == 0 and value.imag < 0) else "+"
        unit = abs(magnitude - 1.0) <= 1e-12
        if power == 0:
            body = to_latex(magnitude)
        else:
            prefix = "" if (unit and power > 0) else to_latex(magnitude)
            body = prefix + (r"\lambda" if power == 1 else rf"\lambda^{{{power}}}")
        if not terms:
            terms.append(("-" if sign == "-" else "") + body)
        else:
            terms.append(f" {sign} {body}")
    return "".join(terms) if terms else "0"


# --------------------------------------------------------------------------- #
# 结果构造
# --------------------------------------------------------------------------- #
def _scalar_item(title: str, value: Any, lhs: str = "", note: str = "",
                 kind: str = "value") -> Any:
    """构造标量结果条目。"""
    from .spec import ResultItem

    number = _clean_scalar(value)
    shown = _display_snap(np.asarray(number)).item()
    latex = f"{lhs} = {to_latex(shown)}" if lhs else to_latex(shown)
    return ResultItem(
        title=title,
        latex=latex,
        note=note,
        kind=kind,
        plain=f"{lhs} = {_plain(shown)}" if lhs else _plain(shown),
        payload=number,
        options={"lhs": lhs},
    )


def _text_item(title: str, latex_body: str, note: str = "", plain: str = "",
               kind: str = "text", payload: Any = None) -> Any:
    """构造纯文本 / LaTeX 片段结果条目。"""
    from .spec import ResultItem

    return ResultItem(
        title=title,
        latex=latex_body,
        note=note,
        kind=kind,
        plain=plain or latex_body,
        payload=payload,
    )


def _needs_complex_text(arr: np.ndarray) -> bool:
    """该数组是否含非零虚部（需要走复数专用的纯文本路径）。"""
    return np.iscomplexobj(arr) and bool(np.any(np.asarray(arr).imag != 0.0))


def _complex_plain(value: Any) -> str:
    """复数的纯文本表示。

    ``numberfmt.format_number`` 对纯复标量会无限递归（``format_number`` 与
    ``format_value`` 互相调用），因此这里改用 :func:`numberfmt.to_latex` 生成
    等价的纯文本形式（``2 + 3i`` / ``-i``），它对复数是安全的。
    """
    number = complex(value)
    return to_latex(complex(_snap_integer(number.real), _snap_integer(number.imag)))


def _complex_latex(value: Any) -> str:
    """复数（或实数）元素的 LaTeX 表示。

    ``numberfmt.matrix_to_latex`` 会先对单元格调用 ``snap_zero``，而它会通过
    ``float()`` 丢掉虚部，所以复数矩阵必须在这里自行渲染。
    """
    if isinstance(value, (complex, np.complexfloating)) and complex(value).imag != 0.0:
        return _complex_plain(value)
    return to_latex(_snap_integer(snap_zero(float(np.real(value)))))


#: 超过这些尺寸的复数矩阵在 LaTeX 中省略中间行列（与 numberfmt 保持一致）
_MAX_LATEX_ROWS = 24
_MAX_LATEX_COLS = 14


def _complex_matrix_latex(arr: np.ndarray, name: str | None = None,
                          env: str = "pmatrix") -> str:
    """复数矩阵的 LaTeX 表示（做法与 ``numberfmt.matrix_to_latex`` 一致）。"""
    matrix = np.asarray(arr)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    n_rows, n_cols = matrix.shape
    row_idx = list(range(n_rows))
    col_idx = list(range(n_cols))
    if n_rows > _MAX_LATEX_ROWS:
        head, tail = _MAX_LATEX_ROWS // 2, _MAX_LATEX_ROWS - _MAX_LATEX_ROWS // 2 - 1
        row_idx = list(range(head)) + [-1] + list(range(n_rows - tail, n_rows))
    if n_cols > _MAX_LATEX_COLS:
        head, tail = _MAX_LATEX_COLS // 2, _MAX_LATEX_COLS - _MAX_LATEX_COLS // 2 - 1
        col_idx = list(range(head)) + [-1] + list(range(n_cols - tail, n_cols))
    rows: list[str] = []
    for i in row_idx:
        cells: list[str] = []
        for j in col_idx:
            if i == -1 and j == -1:
                cells.append(r"\ddots")
            elif i == -1:
                cells.append(r"\vdots")
            elif j == -1:
                cells.append(r"\cdots")
            else:
                cells.append(_complex_latex(matrix[i, j]))
        rows.append(" & ".join(cells))
    env_name = env if env in {"matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix",
                              "Vmatrix", "smallmatrix"} else "pmatrix"
    body = rf"\begin{{{env_name}}}" + r" \\ ".join(rows) + rf"\end{{{env_name}}}"
    return f"{name} = {body}" if name else body


def _complex_matrix_text(arr: np.ndarray, name: str | None = None) -> str:
    """复数矩阵的等宽纯文本表示。

    不能借用 :func:`numberfmt.matrix_to_text`：它内部会执行
    ``arr.astype(complex)``，对字符串单元格会失败，对复数单元格则会丢弃虚部。
    因此这里自己对齐列宽，数字本身仍由 :func:`numberfmt.to_latex` 渲染。
    """
    matrix = np.asarray(arr)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    texts = [[_complex_plain(v) for v in row] for row in matrix]
    widths = [max(len(row[j]) for row in texts) for j in range(len(texts[0]))]
    body = "\n".join(
        "[ " + "  ".join(cell.rjust(w) for cell, w in zip(row, widths)) + " ]" for row in texts
    )
    return f"{name} =\n{body}" if name else body


def _matrix_item(title: str, mat: Any, name: str | None = None, note: str = "",
                 kind: str = "matrix", env: str = "pmatrix",
                 fractions: bool = False) -> Any:
    """构造矩阵结果条目（LaTeX + 等宽纯文本双重表示）。"""
    from .spec import ResultItem

    arr = _clean(_real_if_close(np.asarray(mat)))
    shown = _display_snap(arr)
    if _needs_complex_text(arr):
        plain = _complex_matrix_text(shown, name=name)
        latex = _complex_matrix_latex(shown, name=name, env=env)
    else:
        plain = matrix_to_text(shown, name=name)
        latex = matrix_to_latex(shown, env=env, name=name, fractions=fractions)
    return ResultItem(
        title=title,
        latex=latex,
        note=note,
        kind=kind,
        plain=plain,
        payload=arr,
        options={"env": env, "name": name or ""},
    )


def _vector_item(title: str, vec: Any, name: str | None = None, note: str = "",
                 column: bool = True) -> Any:
    """构造向量结果条目（默认竖排列向量）。"""
    from .spec import ResultItem

    arr = _clean(_real_if_close(np.asarray(vec).reshape(-1)))
    shown = _display_snap(arr)
    if _needs_complex_text(arr):
        body = _complex_matrix_latex(shown.reshape(-1, 1), env="pmatrix")
        plain = _complex_matrix_text(shown.reshape(-1, 1), name=name)
    else:
        body = vector_to_latex(shown)
        plain = matrix_to_text(shown.reshape(-1, 1), name=name)
    latex = f"{name} = {body}" if name else body
    return ResultItem(
        title=title,
        latex=latex,
        note=note,
        kind="vector",
        plain=plain,
        payload=arr,
        options={"name": name or "", "column": column},
    )


def _number_list_item(title: str, values: Sequence[Any], names: Sequence[str] | None = None,
                      note: str = "", lhs: str = "") -> Any:
    """把一组标量写成 ``\\begin{aligned}...\\end{aligned}`` 的 LaTeX。"""
    from .spec import ResultItem

    numbers = [_clean_scalar(v) for v in values]
    shown = [_display_snap(np.asarray(num)).item() for num in numbers]
    labels = list(names) if names else [f"x_{{{i + 1}}}" for i in range(len(numbers))]
    rows = [rf"{lab} &= {to_latex(num)}" for lab, num in zip(labels, shown)]
    if not rows:
        rows = [r"\text{none}"]
    latex = (f"{lhs} = " if lhs else "") + r"\begin{aligned}" + r" \\ ".join(rows) + r"\end{aligned}"
    plain = "\n".join(f"{lab} = {_plain(num)}" for lab, num in zip(labels, shown))
    return ResultItem(
        title=title,
        latex=latex,
        note=note,
        kind="value",
        plain=plain,
        payload=numbers,
    )


def _error_sheet(reason: str, key: str, label: str) -> Any:
    """构造失败结果表：始终返回一条中文说明，绝不抛出异常。"""
    from .spec import ResultItem, ResultSheet

    note = str(reason).strip() or "未知原因。"
    sheet = ResultSheet(
        title=f"矩阵运算：{label}",
        created=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    sheet.add(ResultItem(
        title="无法完成计算",
        latex="",
        note=note,
        level=0,
        kind="text",
        plain=f"无法完成计算：{note}",
        payload=None,
        options={"key": key, "ok": False},
    ))
    return sheet


def _sheet(title: str) -> Any:
    """新建一个结果表。"""
    from .spec import ResultSheet

    return ResultSheet(
        title=title,
        created=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


# --------------------------------------------------------------------------- #
# 生成矩阵
# --------------------------------------------------------------------------- #
def _gen_identity(A: Any, B: Any, scalar: Any) -> list[Any]:
    n = _as_int(scalar, 3, "阶数 n", minimum=1)
    mat = np.eye(n)
    return [_matrix_item("单位矩阵", mat, name="I", note=f"{n} 阶单位矩阵。")]


def _gen_like(scalar_or_none: Any, fill: float, label: str) -> list[Any]:
    n = 3
    if not _is_missing(scalar_or_none):
        n = _as_int(scalar_or_none, 3, "阶数 n", minimum=1)
    mat = np.full((n, n), fill)
    return [_matrix_item(label, mat, name=None, note=f"{n}×{n} 矩阵。")]


def _gen_zeros(A: Any, B: Any, scalar: Any) -> list[Any]:
    mat = _gen_like(scalar, 0.0, "零矩阵")
    if not _is_missing(A):
        arr = _to_array(A, "A")
        mat = [_matrix_item("零矩阵", np.zeros_like(arr, dtype=np.float64),
                            note=f"与 A 同形状 {arr.shape[0]}×{arr.shape[1]} 的零矩阵。")]
    return mat


def _gen_ones(A: Any, B: Any, scalar: Any) -> list[Any]:
    if not _is_missing(A):
        arr = _to_array(A, "A")
        return [_matrix_item("全 1 矩阵", np.ones_like(arr, dtype=np.float64),
                             note=f"与 A 同形状 {arr.shape[0]}×{arr.shape[1]} 的全 1 矩阵。")]
    return _gen_like(scalar, 1.0, "全 1 矩阵")


def _gen_random(A: Any, B: Any, scalar: Any) -> list[Any]:
    n = _as_int(scalar, 3, "阶数 n", minimum=1)
    rng = np.random.default_rng(0)
    mat = rng.random((n, n))
    return [_matrix_item("随机矩阵", mat, name=None,
                         note=f"{n}×{n} 的 [0,1) 均匀随机矩阵，随机种子固定为 0，可复现。")]


def _gen_hilbert(A: Any, B: Any, scalar: Any) -> list[Any]:
    n = _as_int(scalar, 3, "阶数 n", minimum=1)
    idx = np.arange(1, n + 1, dtype=np.float64)
    mat = 1.0 / (idx[:, None] + idx[None, :] - 1.0)
    return [_matrix_item("希尔伯特矩阵", mat, name="H",
                         note=f"{n} 阶希尔伯特矩阵，条件数随阶数迅速增大。")]


def _magic_odd(n: int) -> np.ndarray:
    """构造奇数阶幻方（Siamese 方法）。"""
    mat = np.zeros((n, n), dtype=np.int64)
    i, j = 0, n // 2
    for value in range(1, n * n + 1):
        mat[i, j] = value
        ni, nj = (i - 1) % n, (j + 1) % n
        if mat[ni, nj]:
            i = (i + 1) % n
        else:
            i, j = ni, nj
    return mat


def _gen_magic(A: Any, B: Any, scalar: Any) -> list[Any]:
    n = _as_int(scalar, 3, "阶数 n", minimum=1)
    if n % 2 == 0:
        raise MatrixOpError(f"幻方仅支持奇数阶，收到的是 n = {n}，请改用 3、5、7 这样的奇数阶。")
    mat = _magic_odd(n)
    total = n * (n * n + 1) // 2
    return [_matrix_item("幻方矩阵", mat, name=None,
                         note=f"{n} 阶幻方，每行、每列与两条对角线之和均为 {total}。")]


# --------------------------------------------------------------------------- #
# 基本运算
# --------------------------------------------------------------------------- #
def _op_transpose(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    return [_matrix_item("转置矩阵", arr.T, name=r"A^{T}", note="行列互换。")]


def _op_ctranspose(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    out = arr.conj().T
    return [_matrix_item("共轭转置矩阵", out, name=r"A^{H}", note="转置后再取复共轭。")]


def _op_trace(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    value = float(np.real(np.trace(arr)))
    return [_scalar_item("矩阵的迹", value, lhs=r"\operatorname{tr}(A)",
                         note="主对角线元素之和。")]


def _op_scalar_multiply(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    k = _as_scalar(scalar, 2.0, "标量 k")
    return [_matrix_item("数乘结果", k * arr, name="kA", note=f"k = {_plain(k)}。")]


def _op_add(A: Any, B: Any, scalar: Any) -> list[Any]:
    a = _to_array(A, "A")
    b = _to_array(B, "B")
    _as_same_shape(a, b)
    return [_matrix_item("矩阵之和", a + b, name="A + B")]


def _op_subtract(A: Any, B: Any, scalar: Any) -> list[Any]:
    a = _to_array(A, "A")
    b = _to_array(B, "B")
    _as_same_shape(a, b)
    return [_matrix_item("矩阵之差", a - b, name="A - B")]


def _op_hadamard(A: Any, B: Any, scalar: Any) -> list[Any]:
    a = _to_array(A, "A")
    b = _to_array(B, "B")
    _as_same_shape(a, b)
    return [_matrix_item("哈达玛积", a * b, name=r"A \circ B", note="逐元素相乘。")]


def _op_power(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    k = _as_int(scalar, 2, "幂次 k")
    if k < 0:
        det = complex(np.linalg.det(arr))
        if abs(det) <= _TOL * max(1.0, float(np.max(np.abs(arr))) ** arr.shape[0]):
            raise MatrixOpError("负数幂要求矩阵可逆，但当前矩阵是奇异的（行列式为 0）。")
    if k > 0:
        # 粗略估计增长：任何矩阵范数都不小于谱半径，若 |A|^k 必然溢出则提前提示
        magnitude = float(np.max(np.abs(arr)))
        underflow = magnitude < 1.0 and k * math.log10(magnitude) < -300.0
        overflow = magnitude > 1.0 and k * math.log10(magnitude) > 300.0
        if underflow or overflow:
            raise MatrixOpError(
                f"矩阵幂 A^{k} 的量级超出双精度浮点范围（约 1e{int(k * math.log10(magnitude))}），"
                "请减小幂次或先缩放矩阵。"
            )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            out = np.linalg.matrix_power(arr, k)
    except np.linalg.LinAlgError as exc:
        raise MatrixOpError(f"矩阵幂计算失败：{exc}") from exc
    if not np.all(np.isfinite(out)):
        raise MatrixOpError(f"矩阵幂 A^{k} 溢出为无穷大，请减小幂次或先缩放矩阵。")
    note = f"k = {k}；" + ("逆矩阵的幂。" if k < 0 else "矩阵自乘 k 次。")
    return [_matrix_item("矩阵幂", out, name=f"A^{{{k}}}", note=note)]


def _op_elementwise_power(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    k = _as_scalar(scalar, 2.0, "幂次 k")
    if np.iscomplexobj(arr):
        raise MatrixOpError("逐元素幂暂不支持复数矩阵。")
    if float(k) != round(float(k)) and np.any(arr < 0):
        raise MatrixOpError("存在负元素时，逐元素幂的指数必须是整数。")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = np.power(arr, k)
    if not np.all(np.isfinite(out)):
        raise MatrixOpError("逐元素幂产生了非有限值，请检查底数与指数。")
    return [_matrix_item("逐元素幂", out, name=rf"A^{{\circ {_plain(k)}}}",
                         note=f"每个元素各自取 {_plain(k)} 次幂。")]


def _op_negate(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    return [_matrix_item("取负", -arr, name="-A")]


def _op_abs(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    return [_matrix_item("逐元素绝对值", np.abs(arr), name=r"|A|",
                         note="对复数元素取模。" if np.iscomplexobj(arr) else "")]


def _op_round_matrix(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    k = _as_int(scalar, 2, "小数位数 k", minimum=0)
    if k > 15:
        raise MatrixOpError(f"小数位数 k 必须不大于 15，收到的是 {k}。")
    return [_matrix_item("四舍五入结果", np.round(arr, k), name=rf"\operatorname{{round}}(A,{k})",
                         note=f"保留 {k} 位小数。")]


# --------------------------------------------------------------------------- #
# 矩阵乘法
# --------------------------------------------------------------------------- #
def _op_matmul(A: Any, B: Any, scalar: Any) -> list[Any]:
    a = _to_array(A, "A")
    b = _to_array(B, "B")
    if a.shape[1] != b.shape[0]:
        raise MatrixOpError(
            f"矩阵乘法要求 A 的列数等于 B 的行数，但 A 是 {a.shape[0]}×{a.shape[1]}，B 是 {b.shape[0]}×{b.shape[1]}。"
            f"可尝试先对 B 转置（{a.shape[1]}×{b.shape[1]} 才能相乘）。"
        )
    return [_matrix_item("矩阵乘积", a @ b, name=r"A \cdot B")]


def _op_kron(A: Any, B: Any, scalar: Any) -> list[Any]:
    a = _to_array(A, "A")
    b = _to_array(B, "B")
    return [_matrix_item("克罗内克积", np.kron(a, b), name=r"A \otimes B",
                         note=f"结果形状 {a.shape[0] * b.shape[0]}×{a.shape[1] * b.shape[1]}。")]


# --------------------------------------------------------------------------- #
# 行列式与秩
# --------------------------------------------------------------------------- #
def _op_determinant(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    value = complex(np.linalg.det(arr))
    note = ""
    if abs(value) <= 1e-12 * max(1.0, float(np.max(np.abs(arr))) ** arr.shape[0]):
        note = "行列式为 0，矩阵奇异（不可逆）。"
    return [_scalar_item("行列式", value, lhs=r"\det(A)", note=note)]


def _op_rank(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    r = _matrix_rank(arr)
    return [
        _scalar_item("矩阵的秩", r, lhs=r"\operatorname{rank}(A)",
                     note=f"行数 {arr.shape[0]}，列数 {arr.shape[1]}，"
                          f"{'满秩' if r == min(arr.shape) else '不满秩（行/列向量线性相关）'}。"),
    ]


def _op_rref(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    rref, pivots = _rref(arr)
    rank = len(pivots)
    pivot_text = "、".join(str(p + 1) for p in pivots) if pivots else "无"
    return [
        _matrix_item("行简化阶梯形", rref, name=r"\operatorname{rref}(A)",
                     note=f"主元所在列：{pivot_text}（从 1 开始计数），秩为 {rank}。"),
    ]


def _op_charpoly(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    coeffs = np.asarray(np.poly(arr), dtype=np.complex128)
    poly = _poly_str(coeffs)
    items = [
        _text_item("特征多项式", rf"p(\lambda) = \det(\lambda I - A) = {poly}",
                   note="按降幂排列的系数（最高次项为 1）。",
                   plain=f"det(λI - A) = {poly}",
                   payload=coeffs),
    ]
    # 特征多项式的根即特征值，作为交叉验证
    roots = np.roots(coeffs) if coeffs.size > 1 else np.zeros(0, dtype=complex)
    roots, _ = _sort_by_magnitude(roots)
    if roots.size:
        items.append(_number_list_item(
            "多项式的根（特征值）", roots,
            names=[rf"\lambda_{{{i + 1}}}" for i in range(roots.size)],
            note="与特征值计算互为验证。",
        ))
    return items


def _cofactor_matrix(arr: np.ndarray) -> np.ndarray:
    """按余子式展开计算伴随矩阵（适用于奇异矩阵）。"""
    n = arr.shape[0]
    out = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(arr, i, axis=0), j, axis=1)
            cof = complex(np.linalg.det(minor)) if minor.size else 1.0 + 0.0j
            out[j, i] = ((-1) ** (i + j)) * cof
    return _real_if_close(out)


def _op_adjugate(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    if arr.shape[0] == 0:
        raise MatrixOpError("矩阵为空，无法计算伴随矩阵。")
    adj = _cofactor_matrix(arr)
    scale = max(1.0, float(np.max(np.abs(arr))) ** max(arr.shape[0] - 1, 1))
    singular = abs(complex(np.linalg.det(arr))) <= 1e-12 * scale
    note = "通过余子式展开计算，奇异矩阵同样适用。" if singular else "满足 A·adj(A) = det(A)·I。"
    return [_matrix_item("伴随矩阵", adj, name=r"\operatorname{adj}(A)", note=note)]


def _op_minor(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    k = _as_int(scalar, 1, "阶数 k", minimum=1)
    if k > min(arr.shape):
        raise MatrixOpError(
            f"k 阶子式要求 k 不超过矩阵的行列数，但 k = {k}，而矩阵形状是 {arr.shape[0]}×{arr.shape[1]}。"
        )
    block = arr[:k, :k]
    value = complex(np.linalg.det(block))
    return [
        _matrix_item(f"{k} 阶子式（左上角）", block, name=None,
                     note=f"取自 A 的左上角 {k}×{k} 子块。"),
        _scalar_item("子式的值", value, lhs=r"\det(A_{1:%d,1:%d})" % (k, k)),
    ]


# --------------------------------------------------------------------------- #
# 逆与广义逆
# --------------------------------------------------------------------------- #
def _inverse_core(arr: np.ndarray) -> np.ndarray:
    """求逆，奇异时给出具体中文提示。"""
    n = arr.shape[0]
    scale = max(1.0, float(np.max(np.abs(arr))) ** n)
    det = complex(np.linalg.det(arr))
    if abs(det) <= 1e-12 * scale:
        raise MatrixOpError(
            f"矩阵是奇异的（行列式约为 {_plain(det)}），不存在逆矩阵；"
            "可以改用「摩尔–彭若斯广义逆 A⁺」。"
        )
    try:
        return np.linalg.inv(arr)
    except np.linalg.LinAlgError as exc:
        raise MatrixOpError(f"求逆失败：{exc}。矩阵可能接近奇异。") from exc


def _op_inverse(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    inv = _inverse_core(arr)
    det = float(np.real(np.linalg.det(arr)))
    return [_matrix_item("逆矩阵", inv, name=r"A^{-1}",
                         note=f"det(A) = {_plain(det)}，满足 A·A⁻¹ = I。")]


def _op_pinv(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    pinv = np.linalg.pinv(arr)
    ns = _nullspace_basis(arr).shape[1]
    note = f"满足 A·A⁺·A = A；A 的秩为 {_matrix_rank(arr)}，零空间维数为 {ns}。"
    return [
        _matrix_item("摩尔–彭若斯广义逆", pinv, name=r"A^{+}", note=note),
    ]


def _op_inverse_check(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    inv = _inverse_core(arr)
    prod = arr @ inv
    residual = float(np.max(np.abs(prod - np.eye(arr.shape[0]))))
    ok = residual <= 1e-8
    return [
        _matrix_item("原矩阵", arr, name="A"),
        _matrix_item("逆矩阵", inv, name=r"A^{-1}"),
        _matrix_item("乘积 A·A⁻¹", prod, name=r"A \cdot A^{-1}",
                     note="理论上应为单位矩阵。"),
        _scalar_item("最大偏差", residual, lhs=r"\max_{i,j}|(A A^{-1} - I)_{ij}|",
                     note="验证通过：逆矩阵正确。" if ok else "偏差偏大：矩阵接近奇异，逆矩阵精度有限。"),
    ]


# --------------------------------------------------------------------------- #
# 特征值与特征向量
# --------------------------------------------------------------------------- #
def _is_hermitian(arr: np.ndarray, tol: float = 1e-9) -> bool:
    """判断方阵是否（数值上）埃尔米特（共轭对称，``Aᴴ = A``）。"""
    if arr.shape[0] != arr.shape[1]:
        return False
    scale = max(1.0, float(np.max(np.abs(arr))))
    return bool(np.max(np.abs(arr - arr.conj().T)) <= tol * scale)


def _eigen_core(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """返回 ``(特征值(降序), 特征向量矩阵, 是否使用了 Hermitian 算法)``。

    实对称矩阵与复埃尔米特矩阵都走 ``eigh``（特征值必为实数，特征向量正交）；
    其余矩阵走 ``eig``，特征值与特征向量都可能是复数。
    """
    hermitian = _is_hermitian(arr)
    try:
        if hermitian:
            values, vectors = np.linalg.eigh(arr)
        else:
            values, vectors = np.linalg.eig(arr)
    except np.linalg.LinAlgError as exc:
        raise MatrixOpError(f"特征值分解失败：{exc}。请检查矩阵是否含有非法数值。") from exc
    values, vectors = _sort_by_magnitude(values, vectors)
    vectors = _canonicalize_columns(vectors)
    return _real_if_close(values), _real_if_close(vectors), hermitian


def _op_eigen(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    values, vectors, hermitian = _eigen_core(arr)
    items: list[Any] = [
        _number_list_item("特征值（按 |λ| 降序）", values,
                          names=[rf"\lambda_{{{i + 1}}}" for i in range(values.size)],
                          note="对称/埃尔米特矩阵使用 eigh，一般矩阵使用 eig。"),
        _matrix_item("特征向量矩阵（按列排列）", vectors, name="V",
                     note="第 j 列是特征值 λ_j 对应的特征向量，已做符号归一化。"),
    ]
    residuals = [float(np.max(np.abs(arr @ vectors[:, j] - values[j] * vectors[:, j])))
                 for j in range(values.size)]
    items.append(_number_list_item(
        "残差 ‖A vᵢ − λᵢ vᵢ‖∞ 最大分量", residuals,
        names=[rf"r_{{{i + 1}}}" for i in range(len(residuals))],
        note="用于验证特征对是否准确（接近 0 表示正确）。",
    ))
    items.append(_text_item(
        "使用的算法",
        r"\text{eigh}" if hermitian else r"\text{eig}",
        note="检测到实对称 / 复埃尔米特矩阵，使用 eigh，特征值必为实数。" if hermitian
             else "一般矩阵，使用 eig，特征值可能为复数。",
        plain="eigh" if hermitian else "eig",
    ))
    return items


def _op_eigenvalues(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    values, _, hermitian = _eigen_core(arr)
    return [
        _number_list_item("特征值（按 |λ| 降序）", values,
                          names=[rf"\lambda_{{{i + 1}}}" for i in range(values.size)],
                          note="对称/埃尔米特矩阵使用 eigh。" if hermitian else "一般矩阵使用 eig。"),
    ]


def _op_eigenvectors(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    values, vectors, _ = _eigen_core(arr)
    items: list[Any] = [
        _matrix_item("特征向量矩阵（按列排列）", vectors, name="V",
                     note="每列对应同序号的特征值，已做符号归一化。"),
    ]
    for j in range(values.size):
        items.append(_vector_item(f"λ_{j + 1} 对应的特征向量", vectors[:, j],
                                  name=rf"v_{{{j + 1}}}",
                                  note=f"对应特征值 λ_{j + 1} = {_plain(values[j])}。"))
    return items


def _op_spectral_radius(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    values, _, _ = _eigen_core(arr)
    radius = float(np.max(np.abs(values))) if values.size else 0.0
    converging = radius < 1.0
    return [
        _scalar_item("谱半径", radius, lhs=r"\rho(A) = \max_i |\lambda_i|",
                     note=f"绝对值最大的特征值为 {_plain(values[0])}；"
                          + ("小于 1，幂迭代收敛。" if converging else "不小于 1。")),
    ]


def _op_singular_values(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    sv = np.linalg.svd(arr, compute_uv=False)
    return [
        _number_list_item("奇异值（降序）", sv,
                          names=[rf"\sigma_{{{i + 1}}}" for i in range(sv.size)],
                          note=f"共 {sv.size} 个奇异值 = min(行数, 列数)。"),
        _scalar_item("最大奇异值（2-范数）", float(sv[0]) if sv.size else 0.0, lhs=r"\sigma_1"),
        _scalar_item("最小奇异值", float(sv[-1]) if sv.size else 0.0, lhs=rf"\sigma_{{{sv.size}}}"),
    ]


# --------------------------------------------------------------------------- #
# 矩阵分解
# --------------------------------------------------------------------------- #
def _lu_decompose(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """部分主元 LU 分解，返回 ``(P, L, U)`` 满足 ``P·A = L·U``。"""
    n = arr.shape[0]
    u = np.array(arr, dtype=np.complex128 if np.iscomplexobj(arr) else np.float64)
    l = np.eye(n, dtype=u.dtype)
    perm = np.arange(n)
    for k in range(n - 1):
        column = np.abs(u[k:, k])
        pivot = k + int(np.argmax(column))
        if column[pivot - k] <= 1e-14 * max(1.0, float(np.max(np.abs(arr)))):
            continue  # 该列下方已全为 0，跳过消元
        if pivot != k:
            u[[k, pivot]] = u[[pivot, k]]
            l[[k, pivot], :k] = l[[pivot, k], :k]
            perm[[k, pivot]] = perm[[pivot, k]]
        for i in range(k + 1, n):
            factor = u[i, k] / u[k, k]
            l[i, k] = factor
            u[i, k:] = u[i, k:] - factor * u[k, k:]
    p = np.eye(n, dtype=u.dtype)[np.argsort(perm)]
    return _real_if_close(p), _real_if_close(l), _real_if_close(u)


def _op_lu(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    p, l, u = _lu_decompose(arr)
    residual = float(np.max(np.abs(p @ arr - l @ u)))
    items = [
        _matrix_item("置换矩阵 P", p, name="P", note="记录行交换；满足 P·A = L·U。"),
        _matrix_item("下三角矩阵 L", l, name="L", note="单位下三角，主对角线全为 1。"),
        _matrix_item("上三角矩阵 U", u, name="U"),
        _scalar_item("分解残差", residual, lhs=r"\max|P A - L U|",
                     note="接近 0 表示分解正确。" if residual <= 1e-8 else "残差偏大，矩阵可能接近奇异。"),
    ]
    return items


def _op_qr(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    try:
        q, r = np.linalg.qr(arr)
    except np.linalg.LinAlgError as exc:
        raise MatrixOpError(f"QR 分解失败：{exc}") from exc
    residual = float(np.max(np.abs(q @ r - arr)))
    return [
        _matrix_item("正交矩阵 Q", q, name="Q", note=f"形状 {q.shape[0]}×{q.shape[1]}，列向量标准正交。"),
        _matrix_item("上三角矩阵 R", r, name="R", note=f"形状 {r.shape[0]}×{r.shape[1]}。"),
        _scalar_item("分解残差", residual, lhs=r"\max|Q R - A|",
                     note="接近 0 表示分解正确。" if residual <= 1e-8 else "残差偏大。"),
    ]


def _op_svd(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    try:
        u, s, vh = np.linalg.svd(arr, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise MatrixOpError(f"奇异值分解失败：{exc}") from exc
    sigma = np.diag(s)
    residual = float(np.max(np.abs(u @ sigma @ vh - arr)))
    return [
        _matrix_item("左奇异向量矩阵 U", u, name="U", note="列向量标准正交。"),
        _matrix_item("奇异值对角矩阵 Σ", sigma, name="S",
                     note="对角线元素是降序排列的奇异值（即 Σ）。"),
        _matrix_item("右奇异向量矩阵的转置 Vᵀ", vh, name=r"V^{T}"),
        _number_list_item("奇异值 σᵢ", s, names=[rf"\sigma_{{{i + 1}}}" for i in range(s.size)]),
        _scalar_item("分解残差", residual, lhs=r"\max|U S V^{T} - A|",
                     note="接近 0 表示分解正确。" if residual <= 1e-8 else "残差偏大。"),
    ]


def _op_cholesky(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    if arr.shape[0] != arr.shape[1]:
        raise MatrixOpError("Cholesky 分解要求方阵。")
    arr = _as_real(arr, "A", "Cholesky 分解")
    if not _is_symmetric(arr):
        raise MatrixOpError("Cholesky 分解要求矩阵对称（Aᵀ = A），当前矩阵不是对称矩阵。")
    if float(np.min(np.linalg.eigvalsh(arr))) <= 0.0:
        raise MatrixOpError("矩阵不是正定矩阵（存在非正特征值），无法进行 Cholesky 分解。")
    try:
        l = np.linalg.cholesky(arr)
    except np.linalg.LinAlgError as exc:
        raise MatrixOpError(f"Cholesky 分解失败：{exc}。矩阵可能只是半正定或接近奇异。") from exc
    residual = float(np.max(np.abs(l @ l.T - arr)))
    return [
        _matrix_item("下三角矩阵 L", l, name="L", note="满足 A = L·Lᵀ。"),
        _matrix_item("上三角矩阵 Lᵀ", l.T, name=r"L^{T}"),
        _scalar_item("分解残差", residual, lhs=r"\max|L L^{T} - A|",
                     note="接近 0 表示分解正确。" if residual <= 1e-8 else "残差偏大。"),
    ]


def _op_eigendecomposition(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _as_square(_to_array(A, "A"), "A")
    values, vectors, hermitian = _eigen_core(arr)
    try:
        p_inv = np.linalg.inv(vectors)
    except np.linalg.LinAlgError as exc:
        raise MatrixOpError(
            "特征向量矩阵不可逆，矩阵不能对角化（可能出现重根或缺陷矩阵）。"
        ) from exc
    diag = np.diag(values)
    rebuilt = vectors @ diag @ p_inv
    residual = float(np.max(np.abs(rebuilt - arr)))
    note = ("实对称 / 复埃尔米特矩阵：P 可取为正交（酉）矩阵，P⁻¹ = Pᴴ。" if hermitian
            else "一般矩阵：P 的列是特征向量，需要求逆。")
    return [
        _matrix_item("特征向量矩阵 P", vectors, name="P", note=note),
        _matrix_item("特征值对角矩阵 D", diag, name="D"),
        _matrix_item("逆矩阵 P⁻¹", p_inv, name=r"P^{-1}"),
        _scalar_item("重建残差", residual, lhs=r"\max|P D P^{-1} - A|",
                     note="接近 0 表示 A = PDP⁻¹ 成立。" if residual <= 1e-8 else "残差偏大。"),
    ]


# --------------------------------------------------------------------------- #
# 范数与条件数
# --------------------------------------------------------------------------- #
def _op_norms(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    with np.errstate(over="ignore", invalid="ignore"):
        fro = float(np.linalg.norm(arr, "fro"))
        n1 = float(np.linalg.norm(arr, 1))
        ninf = float(np.linalg.norm(arr, np.inf))
        n2 = float(np.linalg.norm(arr, 2))
        nuclear = float(np.sum(np.linalg.svd(arr, compute_uv=False)))
    if not all(math.isfinite(v) for v in (fro, n1, ninf, n2, nuclear)):
        raise MatrixOpError(
            "矩阵元素过小或过大，计算范数时发生上溢；请先对矩阵做缩放。"
        )
    return [
        _scalar_item("Frobenius 范数", fro, lhs=r"\|A\|_F",
                     note="所有元素平方和开方。"),
        _scalar_item("1-范数（列和最大）", n1, lhs=r"\|A\|_1"),
        _scalar_item("∞-范数（行和最大）", ninf, lhs=r"\|A\|_\infty"),
        _scalar_item("2-范数（最大奇异值）", n2, lhs=r"\|A\|_2"),
        _scalar_item("核范数（奇异值之和）", nuclear, lhs=r"\|A\|_*"),
    ]


def _op_condition(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        cond2 = float(np.linalg.cond(arr, 2))
        cond1 = float(np.linalg.cond(arr, 1))
        condinf = float(np.linalg.cond(arr, np.inf))
    if not math.isfinite(cond2):
        note = "条件数为无穷大：矩阵奇异（不可逆）。"
    elif cond2 > 1e12:
        note = "条件数极大，矩阵接近奇异，数值求解结果可能不可靠。"
    else:
        note = "条件数越小，矩阵求逆与解方程越稳定（单位矩阵的条件数为 1）。"
    return [
        _scalar_item("2-范数条件数", cond2, lhs=r"\operatorname{cond}_2(A)", note=note),
        _scalar_item("1-范数条件数", cond1, lhs=r"\operatorname{cond}_1(A)"),
        _scalar_item("∞-范数条件数", condinf, lhs=r"\operatorname{cond}_\infty(A)"),
    ]


# --------------------------------------------------------------------------- #
# 线性方程组
# --------------------------------------------------------------------------- #
def _solve_rhs(arr: np.ndarray, B: Any, name: str) -> tuple[np.ndarray, str]:
    """从 B 得到右端向量 b；B 为空时取全 1 向量。"""
    if _is_missing(B):
        b = np.ones(arr.shape[0], dtype=np.float64)
        return b, "未提供 B，已使用全 1 向量作为右端项 b。"
    raw = _to_array(B, "B")
    if raw.shape[1] != 1:
        if raw.shape[0] == 1:
            raw = raw.reshape(-1, 1)
        else:
            return raw[:, 0], f"B 有多列，{name}只使用第一列作为右端项 b。"
    if raw.shape[0] != arr.shape[0]:
        raise MatrixOpError(
            f"右端项 b 的长度必须等于 A 的行数，但 b 有 {raw.shape[0]} 个元素，A 有 {arr.shape[0]} 行。"
        )
    return raw.reshape(-1), f"使用 B 的第一列作为右端项 b（{raw.shape[0]} 维）。"


def _op_solve(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    b, note = _solve_rhs(arr, B, "解方程组")
    square = arr.shape[0] == arr.shape[1]
    singular = abs(complex(np.linalg.det(arr))) <= 1e-12 * max(
        1.0, float(np.max(np.abs(arr))) ** arr.shape[0]) if square else False
    if square and not singular:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x = np.linalg.solve(arr, b)
        method = "唯一解：系数矩阵可逆，使用直接法求解。"
    else:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            x, *_ = np.linalg.lstsq(arr, b, rcond=None)
        method = ("系数矩阵不是方阵，已改用最小二乘求解。" if not square
                  else "系数矩阵奇异，解不唯一，已返回最小范数最小二乘解。")
    residual = arr @ x - b
    items = [
        _matrix_item("系数矩阵 A", arr, name="A", note=note),
        _vector_item("右端向量 b", b, name="b"),
        _vector_item("解向量 x", x, name="x", note=f"{method}"),
        _scalar_item("残差范数", float(np.linalg.norm(residual)), lhs=r"\|A x - b\|_2",
                     note="接近 0 表示 x 确实是解。" if np.linalg.norm(residual) <= 1e-8 * max(1.0, float(np.linalg.norm(b)))
                          else "残差偏大，方程组可能无解或不一致。"),
    ]
    return items


def _op_solve_lstsq(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    b, note = _solve_rhs(arr, B, "最小二乘")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, residuals, rank, sv = np.linalg.lstsq(arr, b, rcond=None)
        residual = arr @ x - b
        norm_res = float(np.linalg.norm(residual))
        norm_x = float(np.linalg.norm(x))
    items = [
        _matrix_item("系数矩阵 A", arr, name="A",
                     note=f"{note} 矩阵形状 {arr.shape[0]}×{arr.shape[1]}，数值秩 {int(rank)}。"),
        _vector_item("右端向量 b", b, name="b"),
        _vector_item("最小二乘解 x", x, name=r"\hat{x}",
                     note="使 ‖Ax − b‖₂ 达到最小的解。"),
        _scalar_item("残差范数", norm_res, lhs=r"\|A \hat{x} - b\|_2"),
        _scalar_item("解的 2-范数", norm_x, lhs=r"\|\hat{x}\|_2"),
    ]
    if np.size(residuals):
        items.append(_scalar_item("残差平方和", float(np.sum(residuals)), lhs=r"\|A \hat{x} - b\|_2^2"))
    return items


def _op_nullspace(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    basis = _nullspace_basis(arr)
    rank = _matrix_rank(arr)
    dim = basis.shape[1]
    items = [
        _matrix_item("零空间的标准正交基（按列）", basis, name="N",
                     note="每列是一个基向量。") if dim else
        _text_item("零空间", r"\dim N(A) = 0",
                   note="零空间只含零向量，矩阵列向量线性无关。",
                   plain="零空间只含零向量（维数为 0）。"),
        _scalar_item("零空间维数", dim, lhs=r"\dim N(A)",
                     note=f"也称零化度 = 列数 {arr.shape[1]} − 秩 {rank}。"),
        _scalar_item("矩阵的秩", rank, lhs=r"\operatorname{rank}(A)"),
    ]
    if dim:
        check = float(np.max(np.abs(arr @ basis))) if basis.size else 0.0
        items.append(_scalar_item("验证 ‖A·N‖∞", check, lhs=r"\max|A N|",
                                  note="接近 0 表示基向量确实在零空间中。"))
    return items


def _op_colspace(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    basis = _colspace_basis(arr)
    rank = _matrix_rank(arr)
    return [
        _matrix_item("列空间的标准正交基（按列）", basis, name="C",
                     note=f"维数等于秩 {rank}，每列是一个基向量。"),
        _scalar_item("列空间的维数", rank, lhs=r"\dim C(A)"),
    ]


def _op_rowspace(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _to_array(A, "A")
    basis = _row_basis(arr)
    rank = _matrix_rank(arr)
    return [
        _matrix_item("行空间的标准正交基（按行）", basis, name="R",
                     note=f"维数等于秩 {rank}，每行是一个基向量。"),
        _scalar_item("行空间的维数", rank, lhs=r"\dim R(A)"),
    ]


# --------------------------------------------------------------------------- #
# 向量组
# --------------------------------------------------------------------------- #
def _rows_as_vectors(A: Any, name: str = "A") -> np.ndarray:
    """把输入按“每行一个向量”解析；一维时视为单个向量。"""
    if _is_missing(A):
        raise MatrixOpError(f"没有提供向量组 {name}。")
    raw = np.asarray(A)
    if raw.dtype.kind in "OUSV":
        raw = np.asarray(raw, dtype=np.complex128) if _looks_complex(raw) else np.asarray(raw, dtype=np.float64)
    if raw.ndim == 0:
        raise MatrixOpError(f"向量组 {name} 至少需要一个向量。")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    elif raw.ndim > 2:
        raise MatrixOpError(f"向量组 {name} 是 {raw.ndim} 维数组，无法按行解释为向量。")
    if raw.shape[1] == 0:
        raise MatrixOpError(f"向量组 {name} 的向量是零维的。")
    try:
        arr = np.asarray(raw, dtype=np.complex128) if raw.dtype.kind == "c" else np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MatrixOpError(f"向量组 {name} 含有无法转换为数值的元素（{exc}）。") from exc
    if not np.all(np.isfinite(arr)):
        raise MatrixOpError(f"向量组 {name} 含有 NaN 或无穷大。")
    if _has_overflow_risk(arr):
        raise MatrixOpError(
            f"向量组 {name} 的分量过大（绝对值 > {_MAX_SAFE_MAGNITUDE:.0e}），"
            "范数与内积会溢出，请先缩放数据。"
        )
    return arr


def _first_vector(A: Any, B: Any) -> tuple[np.ndarray, np.ndarray]:
    """取两个向量：A 的第一行与 B 的第一行。"""
    u_mat = _rows_as_vectors(A, "A")
    v_mat = _rows_as_vectors(B, "B")
    u = u_mat[0]
    v = v_mat[0]
    if u.size != v.size:
        raise MatrixOpError(
            f"两个向量的维数必须相同，但 u 是 {u.size} 维，v 是 {v.size} 维。"
        )
    return u, v


def _op_vector_rank(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _rows_as_vectors(A, "A")
    r = _matrix_rank(arr)
    count = arr.shape[0]
    return [
        _scalar_item("向量组的秩", r, lhs=r"\operatorname{rank}\{\alpha_1,\cdots,\alpha_m\}",
                     note=f"共 {count} 个 {arr.shape[1]} 维向量，"
                          f"{'线性无关' if r == count else '线性相关'}。"),
        _matrix_item("向量组（按行排列）", arr, name="A"),
    ]


def _op_linear_independence(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _rows_as_vectors(A, "A")
    r = _matrix_rank(arr)
    count = arr.shape[0]
    independent = r == count
    conclusion = "线性无关" if independent else "线性相关"
    note = (f"向量组的秩 {r} 等于向量个数 {count}，因此{conclusion}。"
            if independent else
            f"向量组的秩 {r} 小于向量个数 {count}，因此{conclusion}，"
            f"其中 {count - r} 个向量可由其余向量线性表示。")
    text = r"\text{" + ("linearly independent" if independent else "linearly dependent") + "}"
    return [
        _text_item("线性相关性判定", text, note=note, plain=conclusion,
                   payload={"independent": independent, "rank": r, "count": count}),
        _scalar_item("向量组的秩", r, lhs=r"\operatorname{rank}(A)", note=f"向量个数为 {count}。"),
    ]


def _gram_schmidt(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """经典施密特正交化，返回 ``(正交基行向量, 标准正交基行向量, 被跳过的向量下标)``。"""
    tol = 1e-10 * max(1.0, float(np.max(np.abs(arr))) if arr.size else 1.0)
    orthogonal: list[np.ndarray] = []
    skipped: list[int] = []
    for i in range(arr.shape[0]):
        v = np.array(arr[i], dtype=np.complex128 if np.iscomplexobj(arr) else np.float64)
        for u in orthogonal:
            v = v - (np.dot(np.conj(u), arr[i]) / np.dot(np.conj(u), u)) * u
        if float(np.linalg.norm(v)) <= tol:
            skipped.append(i)
            continue
        orthogonal.append(v)
    if not orthogonal:
        width = arr.shape[1]
        return np.zeros((0, width)), np.zeros((0, width)), skipped
    orth = np.vstack(orthogonal)
    normed = np.vstack([u / np.linalg.norm(u) for u in orthogonal])
    return _real_if_close(orth), _real_if_close(normed), skipped


def _op_gram_schmidt(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _rows_as_vectors(A, "A")
    orth, normed, skipped = _gram_schmidt(arr)
    note = f"共 {arr.shape[0]} 个输入向量，得到 {orth.shape[0]} 个正交基向量。"
    if skipped:
        note += "原向量组线性相关，下标 " + "、".join(str(i + 1) for i in skipped) + " 的向量被跳过。"
    items: list[Any] = [
        _text_item("说明", r"\text{Gram-Schmidt}", note=note,
                   plain=note, payload={"skipped": skipped}),
    ]
    if orth.shape[0]:
        items.append(_matrix_item("正交基（按行）", orth, name="U",
                                  note="两两正交，先做正交化。"))
        items.append(_matrix_item("标准正交基（按行）", normed, name="Q",
                                  note="在正交基基础上单位化，每行 2-范数为 1。"))
    else:
        items.append(_text_item("正交基", r"\text{input vectors are all zero}",
                                note="所有输入向量都是零向量，无法构造正交基。",
                                plain="所有输入向量都是零向量。"))
    return items


def _op_dot(A: Any, B: Any, scalar: Any) -> list[Any]:
    u, v = _first_vector(A, B)
    value = complex(np.vdot(u, v))
    angle_note = ""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu > 0 and nv > 0:
        cos = float(np.real(value)) / (nu * nv)
        angle_note = f"两向量夹角的余弦约为 {_plain(cos)}。"
    return [
        _scalar_item("内积", value, lhs=r"u \cdot v",
                     note=("实向量为对应分量乘积之和；复向量取共轭。" + angle_note)),
        _vector_item("向量 u", u, name="u"),
        _vector_item("向量 v", v, name="v"),
    ]


def _op_cross(A: Any, B: Any, scalar: Any) -> list[Any]:
    u, v = _first_vector(A, B)
    if u.size != 3:
        raise MatrixOpError(f"叉积只对三维向量有定义，但当前向量是 {u.size} 维。")
    if np.iscomplexobj(u) or np.iscomplexobj(v):
        raise MatrixOpError("叉积暂不支持复数向量。")
    w = np.cross(u.real, v.real)
    return [
        _vector_item("叉积", w, name=r"u \times v",
                     note="结果与 u、v 都垂直，方向遵循右手定则。"),
        _vector_item("向量 u", u, name="u"),
        _vector_item("向量 v", v, name="v"),
    ]


def _op_vector_norm(A: Any, B: Any, scalar: Any) -> list[Any]:
    arr = _rows_as_vectors(A, "A")
    with np.errstate(over="ignore", invalid="ignore"):
        if arr.shape[0] == 1:
            v = arr[0]
            n1, n2, ninf = (float(np.linalg.norm(v, 1)), float(np.linalg.norm(v, 2)),
                            float(np.linalg.norm(v, np.inf)))
        else:
            n1 = [float(np.linalg.norm(row, 1)) for row in arr]
            n2 = [float(np.linalg.norm(row, 2)) for row in arr]
            ninf = [float(np.linalg.norm(row, np.inf)) for row in arr]
    if not all(math.isfinite(x) for x in ([n1, n2, ninf] if arr.shape[0] == 1 else [*n1, *n2, *ninf])):
        raise MatrixOpError("向量分量过大或过小，计算范数时发生上溢，请先缩放数据。")
    if arr.shape[0] == 1:
        return [
            _scalar_item("1-范数", n1, lhs=r"\|u\|_1", note="分量绝对值之和。"),
            _scalar_item("2-范数", n2, lhs=r"\|u\|_2", note="欧几里得长度。"),
            _scalar_item("∞-范数", ninf, lhs=r"\|u\|_\infty", note="分量绝对值的最大值。"),
            _vector_item("向量 u", v, name="u"),
        ]
    items: list[Any] = [_matrix_item("向量组（按行）", arr, name="A",
                                     note=f"共 {arr.shape[0]} 个向量，下面给出每个向量的范数。")]
    names = [rf"\|u_{{{i + 1}}}\|" for i in range(arr.shape[0])]
    items.append(_number_list_item("1-范数", n1, names=names))
    items.append(_number_list_item("2-范数", n2, names=names))
    items.append(_number_list_item("∞-范数", ninf, names=names))
    return items


def _op_angle(A: Any, B: Any, scalar: Any) -> list[Any]:
    u, v = _first_vector(A, B)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        raise MatrixOpError("存在零向量，零向量与任何向量的夹角没有定义（内积为 0）。")
    cos = float(np.real(np.vdot(u, v))) / (nu * nv)
    cos = min(1.0, max(-1.0, cos))
    radians = math.acos(cos)
    degrees = math.degrees(radians)
    note = "两向量正交。" if abs(cos) <= 1e-12 else ("两向量同向。" if abs(cos - 1.0) <= 1e-12
                                                     else ("两向量反向。" if abs(cos + 1.0) <= 1e-12 else ""))
    return [
        _scalar_item("夹角余弦", cos, lhs=r"\operatorname{cos}\theta = \frac{u \cdot v}{\|u\|_2 \|v\|_2}", note=note),
        _scalar_item("夹角（弧度）", radians, lhs=r"\theta"),
        _scalar_item("夹角（角度制）", degrees, lhs=r"\theta^{\circ}"),
        _vector_item("向量 u", u, name="u"),
        _vector_item("向量 v", v, name="v"),
    ]


def _op_projection(A: Any, B: Any, scalar: Any) -> list[Any]:
    u, v = _first_vector(A, B)
    uu = complex(np.vdot(u, u))
    if abs(uu) <= 1e-14:
        raise MatrixOpError("投影方向 u 是零向量，无法计算投影。")
    coeff = complex(np.vdot(u, v)) / uu
    proj = coeff * u
    orth = v - proj
    return [
        _vector_item("v 在 u 上的投影", proj, name=r"\operatorname{proj}_u(v)",
                     note="即 (u·v / u·u)·u，与 u 平行。"),
        _scalar_item("投影系数", coeff, lhs=r"\frac{u \cdot v}{u \cdot u}"),
        _vector_item("垂直分量", orth, name=r"v - \operatorname{proj}_u(v)",
                     note="与 u 正交（残差部分）。"),
        _vector_item("向量 u", u, name="u"),
        _vector_item("向量 v", v, name="v"),
    ]


def _op_span_contains(A: Any, B: Any, scalar: Any) -> list[Any]:
    basis_mat = _rows_as_vectors(A, "A")
    if _is_missing(B):
        raise MatrixOpError("请把待判定的向量放在 B 中（取 B 的第一行）。")
    target_mat = _rows_as_vectors(B, "B")
    target = target_mat[0]
    if target.size != basis_mat.shape[1]:
        raise MatrixOpError(
            f"待判定向量是 {target.size} 维，而张成空间中的向量是 {basis_mat.shape[1]} 维，维数不一致。"
        )
    rank_a = _matrix_rank(basis_mat)
    stacked = np.vstack([basis_mat, target.reshape(1, -1)])
    rank_b = _matrix_rank(stacked)
    contained = rank_b == rank_a
    text = r"\text{" + ("inside the span" if contained else "outside the span") + "}"
    answer = "在张成空间中" if contained else "不在张成空间中"
    note = (f"秩(A) = {rank_a}，秩([A; b]) = {rank_b}，两者相等，因此 b 可以由 A 的行向量线性表示。"
            if contained else
            f"秩(A) = {rank_a}，秩([A; b]) = {rank_b}，秩增大，因此 b 不能由 A 的行向量线性表示。")
    items: list[Any] = [
        _text_item("判定结论", text, note=note, plain=answer,
                   payload={"contained": contained, "rank_A": rank_a, "rank_augmented": rank_b}),
    ]
    # 若在张成空间内，顺便给出线性表示系数（最小二乘精确解）
    if contained:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            coeffs, *_ = np.linalg.lstsq(basis_mat.T, target, rcond=None)
        items.append(_number_list_item(
            "线性表示系数 cᵢ（b = Σ cᵢ·αᵢ）", coeffs,
            names=[rf"c_{{{i + 1}}}" for i in range(coeffs.size)],
            note="由最小二乘精确求解（此时残差为 0）。",
        ))
    items.append(_matrix_item("生成向量组（按行）", basis_mat, name="A"))
    items.append(_vector_item("待判定向量 b", target, name="b"))
    return items


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def _spec(key: str, label: str, group: str, description: str, latex_hint: str = "",
          *, arity: int = 1, scalar: bool = False, vector_input: bool = False) -> OperationSpec:
    """构造 :class:`OperationSpec` 的简写。"""
    return OperationSpec(
        key=key, label=label, group=group, arity=arity, scalar=scalar,
        vector_input=vector_input, description=description, latex_hint=latex_hint,
    )


#: 所有可执行运算的注册表（顺序即界面按钮顺序）
REGISTRY: list[OperationSpec] = [
    # -- 生成矩阵 ---------------------------------------------------------- #
    _spec("identity", "单位矩阵 Iₙ", "生成矩阵", "生成 n 阶单位矩阵，n 由标量给出（默认 3）。",
          r"I_n", arity=0, scalar=True),
    _spec("zeros", "零矩阵", "生成矩阵", "生成全 0 矩阵：有 A 时与 A 同形状，否则 3×3。",
          r"0", arity=0, scalar=True),
    _spec("ones", "全 1 矩阵", "生成矩阵", "生成全 1 矩阵：有 A 时与 A 同形状，否则 3×3。",
          r"1", arity=0, scalar=True),
    _spec("random", "随机矩阵", "生成矩阵", "生成 n×n 的 [0,1) 均匀随机矩阵，种子固定为 0，结果可复现。",
          r"R", arity=0, scalar=True),
    _spec("hilbert", "希尔伯特矩阵", "生成矩阵", "生成 n 阶希尔伯特矩阵 H(i,j) = 1/(i+j−1)。",
          r"H", arity=0, scalar=True),
    _spec("magic", "幻方矩阵", "生成矩阵", "生成 n 阶幻方（仅支持奇数阶），各行各列与对角线之和相等。",
          r"M", arity=0, scalar=True),

    # -- 基本运算 ---------------------------------------------------------- #
    _spec("transpose", "转置 Aᵀ", "基本运算", "行列互换，得到转置矩阵。", r"A^{T}"),
    _spec("ctranspose", "共轭转置 Aᴴ", "基本运算", "转置后再取复共轭。", r"A^{H}"),
    _spec("trace", "迹 tr(A)", "基本运算", "主对角线元素之和（要求方阵）。", r"\operatorname{tr}(A)"),
    _spec("scalar_multiply", "数乘 kA", "基本运算", "矩阵每个元素乘以标量 k（默认 2）。",
          r"kA", scalar=True),
    _spec("add", "加法 A+B", "基本运算", "同形状矩阵逐元素相加。", r"A + B", arity=2),
    _spec("subtract", "减法 A−B", "基本运算", "同形状矩阵逐元素相减。", r"A - B", arity=2),
    _spec("hadamard", "哈达玛积 A∘B", "基本运算", "同形状矩阵逐元素相乘。",
          r"A \circ B", arity=2),
    _spec("power", "矩阵幂 Aᵏ", "基本运算", "方阵的整数次幂，k 为负时先求逆（默认 k = 2）。",
          r"A^{k}", scalar=True),
    _spec("elementwise_power", "逐元素幂 A.^k", "基本运算", "每个元素分别取 k 次幂（默认 k = 2）。",
          r"A^{\circ k}", scalar=True),
    _spec("negate", "取负 −A", "基本运算", "所有元素取相反数。", r"-A"),
    _spec("abs", "逐元素绝对值 |A|", "基本运算", "每个元素取绝对值（复数取模）。", r"|A|"),
    _spec("round_matrix", "四舍五入到 k 位小数", "基本运算", "把每个元素保留 k 位小数（默认 k = 2）。",
          r"\operatorname{round}(A,k)", scalar=True),

    # -- 矩阵乘法 ---------------------------------------------------------- #
    _spec("matmul", "矩阵乘法 A·B", "矩阵乘法", "矩阵乘积，要求 A 的列数等于 B 的行数。",
          r"A \cdot B", arity=2),
    _spec("kron", "克罗内克积 A⊗B", "矩阵乘法", "张量积式的分块乘法，结果尺寸为两矩阵尺寸之积。",
          r"A \otimes B", arity=2),

    # -- 行列式与秩 -------------------------------------------------------- #
    _spec("determinant", "行列式 det(A)", "行列式与秩", "求方阵的行列式，并提示矩阵是否奇异。",
          r"\det(A)"),
    _spec("rank", "秩 rank(A)", "行列式与秩", "矩阵的数值秩（基于奇异值判定）。",
          r"\operatorname{rank}(A)"),
    _spec("rref", "行简化阶梯形 RREF", "行列式与秩", "高斯–若尔当消元得到行简化阶梯形与主元列。",
          r"\operatorname{rref}(A)"),
    _spec("charpoly", "特征多项式", "行列式与秩", "计算 det(λI − A)，并给出其根作为特征值验证。",
          r"\det(\lambda I - A)"),
    _spec("adjugate", "伴随矩阵 adj(A)", "行列式与秩", "用余子式展开计算伴随矩阵，奇异矩阵同样适用。",
          r"\operatorname{adj}(A)"),
    _spec("minor", "k 阶子式", "行列式与秩", "取左上角 k×k 子块并求其行列式（默认 k = 1）。",
          r"\det(A_{k \times k})", scalar=True),

    # -- 逆与广义逆 -------------------------------------------------------- #
    _spec("inverse", "逆矩阵 A⁻¹", "逆与广义逆", "求方阵的逆矩阵，奇异时给出中文提示。",
          r"A^{-1}"),
    _spec("pinv", "摩尔–彭若斯广义逆 A⁺", "逆与广义逆", "对任意矩阵求伪逆，奇异与长方形矩阵均可用。",
          r"A^{+}"),
    _spec("inverse_check", "验证 A·A⁻¹", "逆与广义逆", "计算 A·A⁻¹ 并检查它与单位矩阵的最大偏差。",
          r"A \cdot A^{-1}"),

    # -- 特征值与特征向量 -------------------------------------------------- #
    _spec("eigen", "特征值与特征向量", "特征值与特征向量",
          "对称矩阵用 eigh、一般矩阵用 eig，结果按 |λ| 降序并给出残差验证。",
          r"A v = \lambda v"),
    _spec("eigenvalues", "特征值 λᵢ", "特征值与特征向量", "只输出降序排列的特征值。", r"\lambda_i"),
    _spec("eigenvectors", "特征向量 vᵢ", "特征值与特征向量",
          "输出特征向量矩阵及每个特征值对应的特征向量（已做符号归一化）。", r"v_i"),
    _spec("spectral_radius", "谱半径 ρ(A)", "特征值与特征向量",
          "特征值绝对值的最大值，用于判断幂迭代是否收敛。", r"\rho(A)"),
    _spec("singular_values", "奇异值 σᵢ", "特征值与特征向量", "按降序输出全部奇异值。", r"\sigma_i"),

    # -- 矩阵分解 ---------------------------------------------------------- #
    _spec("lu", "LU 分解", "矩阵分解", "部分主元高斯消元，返回满足 P·A = L·U 的 P、L、U。",
          r"P A = L U"),
    _spec("qr", "QR 分解", "矩阵分解", "正交–上三角分解 A = Q·R（调用 numpy）。", r"A = Q R"),
    _spec("svd", "奇异值分解 SVD", "矩阵分解", "返回 U、Σ、Vᵀ，并验证 U·Σ·Vᵀ = A。",
          r"A = U S V^{T}"),
    _spec("cholesky", "Cholesky 分解", "矩阵分解",
          "对称正定矩阵的 A = L·Lᵀ 分解，非正定时给出中文提示。", r"A = L L^{T}"),
    _spec("eigendecomposition", "特征分解 A = PDP⁻¹", "矩阵分解",
          "特征分解，输出 P、D 与 P⁻¹ 并验证重建结果。", r"A = P D P^{-1}"),

    # -- 范数与条件数 ------------------------------------------------------ #
    _spec("norms", "各种范数", "范数与条件数",
          "Frobenius、1-范数、∞-范数、2-范数与核范数。", r"\|A\|_F"),
    _spec("condition", "条件数 cond(A)", "范数与条件数",
          "2-范数、1-范数与∞-范数条件数，并提示数值稳定性。", r"\operatorname{cond}(A)"),

    # -- 线性方程组 -------------------------------------------------------- #
    _spec("solve", "解 Ax = b", "线性方程组",
          "解线性方程组，b 取 B 的第一列；B 为空时使用全 1 向量。", r"A x = b"),
    _spec("solve_lstsq", "最小二乘解 Ax ≈ b", "线性方程组",
          "使 ‖Ax − b‖₂ 最小的解，适合超定或不满秩方程组。", r"A x \approx b"),
    _spec("nullspace", "零空间（基础解系）", "线性方程组",
          "输出零空间的标准正交基与零空间维数。", r"A x = 0"),
    _spec("colspace", "列空间的一组基", "线性方程组", "输出列空间的标准正交基。", r"C(A)"),
    _spec("rowspace", "行空间的一组基", "线性方程组", "输出行空间的标准正交基。", r"R(A)"),

    # -- 向量组 ------------------------------------------------------------ #
    _spec("vector_rank", "向量组的秩", "向量组", "把 A 的每一行当作向量，求向量组的秩。",
          r"\operatorname{rank}\{\alpha_i\}", vector_input=True),
    _spec("linear_independence", "线性相关性判定", "向量组",
          "判断向量组线性相关还是线性无关，并给出理由。", r"\sum c_i \alpha_i = 0",
          vector_input=True),
    _spec("gram_schmidt", "施密特正交化", "向量组",
          "施密特正交化，同时给出正交基与标准正交基。", r"Q^{T} Q = I", vector_input=True),
    _spec("dot", "内积 u·v", "向量组",
          "A 的第一行与 B 的第一行的内积。", r"u \cdot v", arity=2, vector_input=True),
    _spec("cross", "叉积 u×v", "向量组",
          "三维向量的叉积（A、B 的第一行）。", r"u \times v", arity=2, vector_input=True),
    _spec("vector_norm", "向量范数", "向量组",
          "向量（或向量组中每个向量）的 1、2、∞ 范数。", r"\|u\|_2", vector_input=True),
    _spec("angle", "夹角余弦与夹角", "向量组",
          "计算两向量夹角的余弦与弧度/角度值。",
          r"\cos\theta", arity=2, vector_input=True),
    _spec("projection", "投影 proj_u(v)", "向量组",
          "把 B 的第一行投影到 A 的第一行上，并给出垂直分量。",
          r"\operatorname{proj}_u(v)", arity=2, vector_input=True),
    _spec("span_contains", "判断向量是否在张成空间中", "向量组",
          "判断 B 的第一行能否由 A 的各行线性表示。",
          r"b \in \operatorname{span}\{\alpha_i\}", arity=2, vector_input=True),
]

_HANDLERS: dict[str, Callable[[Any, Any, Any], list[Any]]] = {
    # 生成矩阵
    "identity": _gen_identity,
    "zeros": _gen_zeros,
    "ones": _gen_ones,
    "random": _gen_random,
    "hilbert": _gen_hilbert,
    "magic": _gen_magic,
    # 基本运算
    "transpose": _op_transpose,
    "ctranspose": _op_ctranspose,
    "trace": _op_trace,
    "scalar_multiply": _op_scalar_multiply,
    "add": _op_add,
    "subtract": _op_subtract,
    "hadamard": _op_hadamard,
    "power": _op_power,
    "elementwise_power": _op_elementwise_power,
    "negate": _op_negate,
    "abs": _op_abs,
    "round_matrix": _op_round_matrix,
    # 矩阵乘法
    "matmul": _op_matmul,
    "kron": _op_kron,
    # 行列式与秩
    "determinant": _op_determinant,
    "rank": _op_rank,
    "rref": _op_rref,
    "charpoly": _op_charpoly,
    "adjugate": _op_adjugate,
    "minor": _op_minor,
    # 逆与广义逆
    "inverse": _op_inverse,
    "pinv": _op_pinv,
    "inverse_check": _op_inverse_check,
    # 特征值与特征向量
    "eigen": _op_eigen,
    "eigenvalues": _op_eigenvalues,
    "eigenvectors": _op_eigenvectors,
    "spectral_radius": _op_spectral_radius,
    "singular_values": _op_singular_values,
    # 矩阵分解
    "lu": _op_lu,
    "qr": _op_qr,
    "svd": _op_svd,
    "cholesky": _op_cholesky,
    "eigendecomposition": _op_eigendecomposition,
    # 范数与条件数
    "norms": _op_norms,
    "condition": _op_condition,
    # 线性方程组
    "solve": _op_solve,
    "solve_lstsq": _op_solve_lstsq,
    "nullspace": _op_nullspace,
    "colspace": _op_colspace,
    "rowspace": _op_rowspace,
    # 向量组
    "vector_rank": _op_vector_rank,
    "linear_independence": _op_linear_independence,
    "gram_schmidt": _op_gram_schmidt,
    "dot": _op_dot,
    "cross": _op_cross,
    "vector_norm": _op_vector_norm,
    "angle": _op_angle,
    "projection": _op_projection,
    "span_contains": _op_span_contains,
}

for _item in REGISTRY:
    if _item.key not in _HANDLERS:
        raise RuntimeError(f"注册表条目 {_item.key!r} 没有对应的处理函数。")

_SPEC_BY_KEY: dict[str, OperationSpec] = {spec.key: spec for spec in REGISTRY}


# --------------------------------------------------------------------------- #
# 公开接口
# --------------------------------------------------------------------------- #
def spec_for(key: str) -> OperationSpec | None:
    """按 key 查找运算描述，找不到返回 ``None``。"""
    return _SPEC_BY_KEY.get(str(key))


def grouped_registry() -> dict[str, list[OperationSpec]]:
    """按分组名返回注册表（保持 :data:`GROUPS` 的顺序）。"""
    out: dict[str, list[OperationSpec]] = {name: [] for name in GROUPS}
    for spec in REGISTRY:
        out.setdefault(spec.group, []).append(spec)
    return {name: items for name, items in out.items() if items}


def run(key: str, A: Any = None, B: Any = None, scalar: Any = None) -> Any:
    """执行一次矩阵运算，返回 :class:`core.spec.ResultSheet`。

    参数
    ----
    key
        :data:`REGISTRY` 中的运算标识。
    A, B
        矩阵 / 向量输入，接受嵌套列表、``np.ndarray`` 或标量；``arity=0`` 的
        生成类运算可以省略。
    scalar
        标量参数（阶数、幂次、小数位数等）。

    本函数**永不抛出异常**：任何失败都会返回只含一条
    ``ResultItem(title="无法完成计算", note=...)`` 的结果表。
    """
    spec = _SPEC_BY_KEY.get(str(key))
    label = spec.label if spec is not None else str(key)
    try:
        if spec is None:
            known = "、".join(s.key for s in REGISTRY)
            raise MatrixOpError(f"未知的运算标识 {key!r}。可用的运算有：{known}。")
        handler = _HANDLERS[spec.key]
        items = handler(A, B, scalar)
        if not items:
            raise MatrixOpError("运算没有产生任何结果。")
        sheet = _sheet(f"矩阵运算：{label}")
        sheet.extend(items)
        return sheet
    except MatrixOpError as exc:
        return _error_sheet(str(exc), str(key), label)
    except np.linalg.LinAlgError as exc:
        return _error_sheet(f"线性代数计算失败：{exc}", str(key), label)
    except MemoryError:
        return _error_sheet("内存不足，矩阵规模过大，请减小尺寸后重试。", str(key), label)
    except Exception as exc:  # noqa: BLE001 - 兜底保证界面永远不会收到异常
        return _error_sheet(f"发生未预期的错误（{type(exc).__name__}）：{exc}", str(key), label)
