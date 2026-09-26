"""数据层：数据集模型、CSV 智能读取、手动录入与表达式派生。

本模块**不依赖 Qt**，可以在脚本或测试中直接使用::

    store = DataSetStore()
    ds, warnings = load_csv("samples/sensor.csv")
    store.add(ds)
    y = evaluate_expression("sin(x) * exp(-x/5)", ds.frame)
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DataSet",
    "DataSetStore",
    "CsvReadOptions",
    "CsvPreview",
    "ENCODING_CANDIDATES",
    "DELIMITER_CHOICES",
    "sniff_text",
    "read_csv_frame",
    "load_csv",
    "frame_from_text",
    "frame_from_columns",
    "evaluate_expression",
    "expression_namespace",
    "EXPRESSION_HELP",
    "format_dataframe_preview",
    "column_kind",
    "numeric_columns",
    "records_from_frame",
]

#: 自动探测时依次尝试的编码
ENCODING_CANDIDATES: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "gb18030",
    "big5",
    "cp936",
    "latin-1",
)

#: 分隔符选项：显示名 -> 实际字符（``auto`` 表示自动探测）
DELIMITER_CHOICES: dict[str, str] = {
    "自动探测": "auto",
    "逗号 ,": ",",
    "分号 ;": ";",
    "制表符 Tab": "\t",
    "竖线 |": "|",
    "空格": " ",
}

_HEADER_CHOICES: dict[str, str] = {"自动判断": "auto", "有表头": "yes", "无表头": "no"}

_NUMERIC_KINDS = "biufc"


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
@dataclass
class DataSet:
    """一个具名数据集（内部是 pandas ``DataFrame``）。"""

    name: str
    frame: pd.DataFrame
    source: str = "manual"          # csv | manual | expression | sample
    path: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- 基本属性 ---------------------------------------------------------- #
    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    @property
    def n_cols(self) -> int:
        return int(self.frame.shape[1])

    @property
    def columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]

    @property
    def numeric_columns(self) -> list[str]:
        return numeric_columns(self.frame)

    @property
    def shape_text(self) -> str:
        return f"{self.n_rows} 行 × {self.n_cols} 列"

    def column(self, name: str) -> pd.Series:
        """按列名取值；``name`` 为空时返回行索引序列。"""
        if not name:
            return pd.Series(np.arange(len(self.frame)), name="index")
        if name not in self.frame.columns:
            raise KeyError(name)
        return self.frame[name]

    def arrays(self, columns: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """导出为 ``{列名: ndarray}``，便于生成源码或计算。"""
        names = list(columns) if columns is not None else self.columns
        out: dict[str, np.ndarray] = {}
        for name in names:
            if name in self.frame.columns:
                out[name] = np.asarray(self.frame[name])
        return out

    def records(self) -> list[dict[str, Any]]:
        return records_from_frame(self.frame)

    def summary(self) -> str:
        return f"{self.name}（{self.shape_text}）"


def column_kind(series: pd.Series) -> str:
    """返回列的语义类型：``numeric`` / ``datetime`` / ``text``。"""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    coerced = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() > 0 and coerced.notna().sum() >= series.notna().sum() * 0.9:
        return "numeric"
    return "text"


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """返回所有数值列名。"""
    return [str(c) for c in frame.columns if column_kind(frame[c]) == "numeric"]


def records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """把 DataFrame 转成 Python 原生对象列表（便于 JSON/源码导出）。"""
    out: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key, value in record.items():
            row[str(key)] = _to_python_native(value)
        out.append(row)
    return out


def _to_python_native(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


# --------------------------------------------------------------------------- #
# CSV 读取
# --------------------------------------------------------------------------- #
@dataclass
class CsvReadOptions:
    """CSV 读取选项。"""

    delimiter: str = "auto"
    encoding: str = "auto"
    header: str = "auto"            # auto | yes | no
    decimal: str = "."
    thousands: str = ""
    skip_rows: int = 0
    skip_footer: int = 0
    comment: str = "#"
    na_values: str = ""
    use_first_column_as_index: bool = False
    max_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "header": self.header,
            "decimal": self.decimal,
            "thousands": self.thousands,
            "skip_rows": self.skip_rows,
            "skip_footer": self.skip_footer,
            "comment": self.comment,
            "na_values": self.na_values,
            "use_first_column_as_index": self.use_first_column_as_index,
            "max_rows": self.max_rows,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "CsvReadOptions":
        if not data:
            return cls()
        names = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class CsvPreview:
    """一次 CSV 解析的结果。"""

    frame: pd.DataFrame
    encoding: str = "utf-8"
    delimiter: str = ","
    has_header: bool = True
    warnings: list[str] = field(default_factory=list)
    skipped_lines: int = 0

    @property
    def ok(self) -> bool:
        return self.frame is not None and self.frame.shape[1] > 0


def _decode(raw: bytes, encoding: str) -> str | None:
    for candidate in ([encoding] if encoding != "auto" else ENCODING_CANDIDATES):
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    try:
        return raw.decode("latin-1")
    except Exception:
        return None


def _count_delimiters(line: str, candidates: str = ",;\t|") -> str:
    """在候选分隔符里挑一个「出现次数最多且各列数一致」的。"""
    best, best_score = ",", -1.0
    for delim in candidates:
        counts = [part.count(delim) for part in line.split("\n") if part.strip()]
        if not counts:
            continue
        mean = sum(counts) / len(counts)
        if mean <= 0:
            continue
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        score = mean - variance * 2.0
        if score > best_score:
            best, best_score = delim, score
    return best


def sniff_text(raw: bytes, options: CsvReadOptions | None = None) -> tuple[str, str, bool]:
    """探测 ``(encoding, delimiter, has_header)``。"""
    options = options or CsvReadOptions()
    text = _decode(raw, options.encoding) or ""
    sample = "\n".join(text.splitlines()[:60])

    delimiter = options.delimiter
    if delimiter == "auto":
        delimiter = _count_delimiters(sample)
        if delimiter == "," and sample.count(";") > sample.count(","):
            delimiter = ";"
        if delimiter == "," and sample.count("\t") > sample.count(","):
            delimiter = "\t"

    has_header = options.header != "no"
    if options.header == "auto":
        try:
            rows = list(csv.reader(io.StringIO(sample), delimiter=delimiter))
        except csv.Error:
            rows = []
        rows = [r for r in rows if any(str(c).strip() for c in r)]
        has_header = _guess_header(rows)
    return options.encoding if options.encoding != "auto" else _detect_encoding(raw), delimiter, has_header


def _detect_encoding(raw: bytes) -> str:
    for candidate in ENCODING_CANDIDATES:
        try:
            raw.decode(candidate)
            return candidate
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def _looks_numeric(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    try:
        float(value.replace(",", ""))
        return True
    except ValueError:
        return False


def _guess_header(rows: list[list[str]]) -> bool:
    """首行全为非数值、且其后存在数值行 -> 认为是表头。"""
    if not rows:
        return False
    if len(rows) == 1:
        return not all(_looks_numeric(c) for c in rows[0] if c.strip())
    first = [c for c in rows[0] if str(c).strip()]
    rest = [c for row in rows[1:12] for c in row if str(c).strip()]
    if not first:
        return False
    first_numeric = sum(1 for c in first if _looks_numeric(c))
    rest_numeric = sum(1 for c in rest if _looks_numeric(c))
    if first_numeric == 0 and rest_numeric > 0:
        return True
    if first_numeric == len(first) and rest_numeric == len(rest):
        return False
    return first_numeric < max(1, len(first) // 2)


def read_csv_frame(path: str, options: CsvReadOptions | None = None) -> CsvPreview:
    """读取 CSV，返回 :class:`CsvPreview`（含自动探测出的编码与分隔符）。"""
    options = options or CsvReadOptions()
    warnings: list[str] = []
    with open(path, "rb") as handle:
        raw = handle.read()

    encoding, delimiter, has_header = sniff_text(raw, options)
    if options.header == "yes":
        has_header = True
    elif options.header == "no":
        has_header = False

    na_values = [v for v in re.split(r"[,\s]+", options.na_values or "") if v] or None
    read_kwargs: dict[str, Any] = {
        "sep": delimiter,
        "encoding": encoding,
        "header": 0 if has_header else None,
        "skiprows": max(int(options.skip_rows), 0),
        "skipfooter": max(int(options.skip_footer), 0),
        "decimal": options.decimal or ".",
        "na_values": na_values,
        "keep_default_na": True,
        "engine": "python",
        "skip_blank_lines": True,
    }
    if options.comment:
        read_kwargs["comment"] = options.comment
    if options.thousands:
        read_kwargs["thousands"] = options.thousands
    if options.max_rows:
        read_kwargs["nrows"] = int(options.max_rows)

    try:
        frame = pd.read_csv(path, **read_kwargs)
    except Exception as exc:  # 宽松降级：换引擎再试一次
        warnings.append(f"按推断参数读取失败（{exc}），已改用宽松模式重试。")
        frame = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding=encoding,
            header=0 if has_header else None,
        )

    if not has_header:
        frame.columns = [f"列{i + 1}" for i in range(frame.shape[1])]

    frame = _tidy_frame(frame, warnings)
    if options.use_first_column_as_index and frame.shape[1] > 1:
        frame = frame.set_index(frame.columns[0])
        frame.index.name = str(frame.index.name or "index")

    if frame.empty:
        warnings.append("表格中没有解析出任何数据行。")

    return CsvPreview(
        frame=frame,
        encoding=encoding,
        delimiter=delimiter,
        has_header=has_header,
        warnings=warnings,
    )


def _tidy_frame(frame: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    """统一列名为字符串、去掉全空列与重复列名。"""
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    drop = [c for c in frame.columns if frame[c].isna().all()]
    if drop:
        frame = frame.drop(columns=drop)
        warnings.append(f"已忽略全为空的列：{', '.join(drop[:6])}")
    seen: dict[str, int] = {}
    new_columns: list[str] = []
    for name in frame.columns:
        if name in seen:
            seen[name] += 1
            new_columns.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_columns.append(name)
    frame.columns = new_columns
    # 尝试把看起来是数字的文本列转成数值
    for name in frame.columns:
        if frame[name].dtype == object:
            coerced = pd.to_numeric(frame[name], errors="coerce")
            non_null = frame[name].notna().sum()
            if non_null and coerced.notna().sum() >= non_null * 0.95:
                frame[name] = coerced
    if frame.columns.empty:
        warnings.append("没有可用的列名。")
    return frame


def load_csv(path: str, options: CsvReadOptions | None = None, *, name: str | None = None):
    """读取 CSV 并包装为 :class:`DataSet`，返回 ``(dataset, warnings)``。"""
    preview = read_csv_frame(path, options)
    dataset = DataSet(
        name=name or os.path.splitext(os.path.basename(path))[0],
        frame=preview.frame,
        source="csv",
        path=os.path.abspath(path),
        meta={
            "encoding": preview.encoding,
            "delimiter": preview.delimiter,
            "has_header": preview.has_header,
        },
    )
    return dataset, preview.warnings


def frame_from_columns(
    columns: Mapping[str, Sequence[Any]] | Sequence[Sequence[Any]],
    names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """从「列字典」或「行列表」构造 DataFrame，长度不足处补 NaN。"""
    if isinstance(columns, Mapping):
        data = {str(k): list(v) for k, v in columns.items()}
        return pd.DataFrame(data)
    rows = [list(row) for row in columns]
    width = max((len(r) for r in rows), default=0)
    labels = list(names) if names else [f"列{i + 1}" for i in range(width)]
    padded = [r + [np.nan] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded, columns=labels[:width])


def frame_from_text(
    text: str,
    *,
    delimiter: str = "auto",
    has_header: bool | None = None,
    decimal: str = ".",
) -> pd.DataFrame:
    """把粘贴的文本（TSV/CSV）解析为 DataFrame。"""
    text = (text or "").strip("\n")
    if not text.strip():
        return pd.DataFrame()
    options = CsvReadOptions(
        delimiter=delimiter,
        header="auto" if has_header is None else ("yes" if has_header else "no"),
        decimal=decimal,
    )
    encoding, delim, header = sniff_text(text.encode("utf-8"), options)
    if has_header is not None:
        header = has_header
    frame = pd.read_csv(
        io.StringIO(text),
        sep=delim,
        header=0 if header else None,
        engine="python",
        decimal=decimal or ".",
        skip_blank_lines=True,
    )
    if not header:
        frame.columns = [f"列{i + 1}" for i in range(frame.shape[1])]
    warnings: list[str] = []
    return _tidy_frame(frame, warnings)


def format_dataframe_preview(frame: pd.DataFrame, max_rows: int = 8) -> str:
    """生成用于日志/工具提示的文本预览。"""
    if frame is None or frame.empty:
        return "（空表）"
    with pd.option_context("display.max_columns", 12, "display.width", 160):
        return frame.head(max_rows).to_string()


# --------------------------------------------------------------------------- #
# 表达式求值
# --------------------------------------------------------------------------- #
#: 允许在表达式中使用的 NumPy 函数与常量
_SAFE_NAMES: dict[str, Any] = {
    name: getattr(np, name)
    for name in (
        "abs", "absolute", "arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctan2",
        "arctanh", "ceil", "clip", "conj", "cos", "cosh", "cumsum", "cumprod", "deg2rad",
        "degrees", "diff", "e", "exp", "expm1", "floor", "fmax", "fmin", "gradient",
        "hypot", "inf", "isfinite", "isinf", "isnan", "linspace", "log", "log10", "log1p",
        "log2", "maximum", "mean", "median", "minimum", "nan", "nan_to_num", "pi", "power",
        "rad2deg", "radians", "real", "imag", "angle", "rint", "round", "sign", "sin",
        "sinc", "sinh", "sqrt", "square", "std", "sum", "tan", "tanh", "trapz", "trunc",
        "var", "where", "zeros", "ones", "full", "arange", "concatenate", "sort", "argsort",
        "mod", "remainder", "prod", "max", "min", "ptp", "median", "percentile", "logspace",
    )
    if hasattr(np, name)
}
_SAFE_NAMES.update({"np": np, "numpy": np, "True": True, "False": False})

#: 表达式帮助文本（界面会展示）
EXPRESSION_HELP = (
    "支持 NumPy 表达式。可用变量：\n"
    "  x   —— 当前数据集第一列（或所选 X 列）\n"
    "  t   —— 行序号（0,1,2,...）\n"
    "  i   —— 同 t\n"
    "  n   —— 行数\n"
    "  任意列名（如 value、temp_1）\n"
    "可用函数：sin cos tan exp log log10 sqrt abs power sign clip where\n"
    "          linspace arange cumsum diff gradient mean std sum min max\n"
    "          nan_to_num 以及 numpy 的常量 pi e\n"
    "示例：sin(x) * exp(-x/5)      |      linspace(0, 2*pi, 200)\n"
    "      x**2 - 3*x + 1          |      np.where(x > 0, x, 0)"
)

_FORBIDDEN = re.compile(r"(__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\blambda\b|\bclass\b|\bglobal\b|\bos\b|\bsys\b)")


def expression_namespace(frame: pd.DataFrame, x_column: str | None = None) -> dict[str, Any]:
    """构造表达式求值用的命名空间。"""
    namespace: dict[str, Any] = dict(_SAFE_NAMES)
    n = len(frame)
    namespace["n"] = n
    namespace["t"] = np.arange(n, dtype=float)
    namespace["i"] = np.arange(n, dtype=float)
    namespace["index"] = np.arange(n, dtype=float)
    if frame is not None and frame.shape[1] > 0:
        first = frame.columns[0]
        try:
            namespace["x"] = np.asarray(pd.to_numeric(frame[first], errors="coerce"), dtype=float)
        except Exception:
            namespace["x"] = np.arange(n, dtype=float)
        if x_column and x_column in frame.columns:
            try:
                namespace["x"] = np.asarray(pd.to_numeric(frame[x_column], errors="coerce"), dtype=float)
            except Exception:
                pass
        for name in frame.columns:
            key = str(name)
            if not key.isidentifier():
                continue
            series = frame[name]
            if column_kind(series) == "numeric":
                namespace[key] = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
    return namespace


def evaluate_expression(
    expression: str,
    frame: pd.DataFrame,
    *,
    x_column: str | None = None,
    extra: Mapping[str, Any] | None = None,
    target_length: int | None = None,
) -> np.ndarray:
    """求值一个 NumPy 表达式，返回与数据行数对齐的 ``ndarray``。

    抛出 :class:`ValueError` 并附带中文原因，界面直接展示即可。
    """
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("表达式为空。")
    if _FORBIDDEN.search(expression):
        raise ValueError("表达式包含不允许使用的名称。")

    namespace = expression_namespace(frame, x_column)
    if extra:
        namespace.update({k: v for k, v in extra.items() if not k.startswith("_")})

    try:
        value = eval(compile(expression, "<表达式>", "eval"), {"__builtins__": {}}, namespace)  # noqa: S307
    except NameError as exc:
        raise ValueError(f"表达式中的名称未定义：{exc}") from exc
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误：{exc.msg}") from exc
    except Exception as exc:
        raise ValueError(f"表达式求值失败：{exc}") from exc

    length = target_length if target_length is not None else len(frame)
    if isinstance(value, (int, float, bool, np.number)):
        return np.full(length, float(value), dtype=float)
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(length, float(array), dtype=float)
    array = array.reshape(-1)
    if length and array.size != length:
        if array.size == 1:
            return np.full(length, float(array[0]), dtype=float)
        array = np.resize(array, length)
    return array


# --------------------------------------------------------------------------- #
# 数据集仓库
# --------------------------------------------------------------------------- #
class DataSetStore:
    """数据集集合，带变更通知（不依赖 Qt）。"""

    def __init__(self) -> None:
        self._items: dict[str, DataSet] = {}
        self._listeners: list[Callable[[], None]] = []

    # -- 通知 -------------------------------------------------------------- #
    def subscribe(self, callback: Callable[[], None]) -> Callable[[], None]:
        """注册变更回调，返回取消注册的函数。"""
        self._listeners.append(callback)

        def unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return unsubscribe

    def _notify(self) -> None:
        for callback in list(self._listeners):
            try:
                callback()
            except Exception:
                pass

    # -- 增删改查 ---------------------------------------------------------- #
    def unique_name(self, base: str) -> str:
        """生成不与现有数据集冲突的名称。"""
        base = (base or "数据集").strip() or "数据集"
        if base not in self._items:
            return base
        index = 2
        while f"{base} ({index})" in self._items:
            index += 1
        return f"{base} ({index})"

    def add(self, dataset: DataSet, *, replace: bool = False, notify: bool = True) -> DataSet:
        name = dataset.name if replace else self.unique_name(dataset.name)
        dataset.name = name
        self._items[name] = dataset
        if notify:
            self._notify()
        return dataset

    def remove(self, name: str, *, notify: bool = True) -> bool:
        if name in self._items:
            del self._items[name]
            if notify:
                self._notify()
            return True
        return False

    def rename(self, old: str, new: str, *, notify: bool = True) -> bool:
        if old not in self._items or not new.strip():
            return False
        dataset = self._items.pop(old)
        dataset.name = self.unique_name(new.strip())
        self._items[dataset.name] = dataset
        if notify:
            self._notify()
        return True

    def update_frame(self, name: str, frame: pd.DataFrame, *, notify: bool = True) -> bool:
        dataset = self._items.get(name)
        if dataset is None:
            return False
        dataset.frame = frame
        if notify:
            self._notify()
        return True

    def get(self, name: str) -> DataSet | None:
        return self._items.get(name)

    def require(self, name: str) -> DataSet:
        dataset = self._items.get(name)
        if dataset is None:
            raise KeyError(f"数据集不存在：{name}")
        return dataset

    def names(self) -> list[str]:
        return list(self._items.keys())

    def items(self) -> list[DataSet]:
        return list(self._items.values())

    def frames(self) -> dict[str, pd.DataFrame]:
        return {name: ds.frame for name, ds in self._items.items()}

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)

    def clear(self, *, notify: bool = True) -> None:
        self._items.clear()
        if notify:
            self._notify()
