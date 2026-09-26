"""可复用控件：matplotlib 画布、LaTeX 结果视图、数据表、矩阵编辑器。

其中 :class:`LatexButton` / :class:`LatexLabel` 会把带 ``$...$`` 的文本交给
:mod:`core.latex` 渲染成矢量位图再显示，因此界面上的数学符号
（``Aᵀ``、``A⁻¹``、``λᵢ``、``‖A‖`` 等）是**真正的 LaTeX 排版**，
不依赖系统字体是否含有这些 Unicode 字符。
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
from typing import Any, Callable, Sequence

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
import pandas as pd
from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QFont,
    QGuiApplication,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core.latex import (
    LatexDocument,
    TextBlock,
    figure_from_layout,
    figure_to_png_bytes,
    figure_to_svg,
    layout_document,
    render_document,
    render_strip,
)
from core.latex.results import ResultStyle, document_for_sheet
from core.numberfmt import matrix_to_latex

__all__ = [
    "MplCanvas",
    "LatexView",
    "LatexButton",
    "LatexLabel",
    "latex_pixmaps",
    "render_latex_pixmap",
    "render_latex_pixmaps",
    "latex_cache_dir",
    "latex_to_plain",
    "DataTableWidget",
    "MatrixEditor",
    "MatrixEditorDialog",
    "clipboard_to_grid",
    "parse_number",
    "grid_to_matrix",
    "FILL_POLICIES",
]


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def clipboard_to_grid(text: str) -> list[list[str]]:
    """把剪贴板文本解析为二维字符串表格（自动识别 Tab / 逗号 / 分号）。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not text.strip():
        return []
    sample = "\n".join(text.splitlines()[:20])
    delimiter = "\t"
    if sample.count("\t") == 0:
        counts = {d: sample.count(d) for d in (",", ";", "|")}
        best = max(counts, key=lambda k: counts[k])
        delimiter = best if counts[best] > 0 else "\t"
    try:
        return [row for row in csv.reader(io.StringIO(text), delimiter=delimiter)]
    except csv.Error:
        return [[cell for cell in line.split(delimiter)] for line in text.splitlines()]


def parse_number(text: str) -> float | complex:
    """解析一个数值（支持 ``1+2j`` 这类复数与千分位逗号）。"""
    text = str(text).strip().replace("，", ",").replace(" ", "")
    if text == "":
        return 0.0
    text = text.replace(",", "")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return complex(text.replace("i", "j"))
    except ValueError as exc:
        raise ValueError(f"无法解析数值：{text!r}") from exc


def _set_accent(button: QPushButton, accent: bool = True) -> QPushButton:
    button.setProperty("accent", "true" if accent else "false")
    return button


# --------------------------------------------------------------------------- #
# LaTeX 渲染成位图（界面上的数学符号）
# --------------------------------------------------------------------------- #
_MATH_PLAIN_RULES: tuple[tuple[str, str], ...] = (
    (r"\circ", "\u2218"), (r"\otimes", "\u2297"), (r"\oplus", "\u2295"),
    (r"\times", "\u00d7"), (r"\cdot", "\u00b7"), (r"\pm", "\u00b1"),
    (r"\infty", "\u221e"), (r"\lambda", "\u03bb"), (r"\sigma", "\u03c3"),
    (r"\rho", "\u03c1"), (r"\theta", "\u03b8"), (r"\mu", "\u03bc"),
    (r"\alpha", "\u03b1"), (r"\beta", "\u03b2"), (r"\pi", "\u03c0"),
    (r"\leq", "\u2264"), (r"\geq", "\u2265"), (r"\neq", "\u2260"),
    (r"\approx", "\u2248"), (r"\cdots", "\u22ef"), (r"\dots", "\u2026"),
)


def latex_to_plain(text: str) -> str:
    """把含 ``$...$`` 的标签降级成可读纯文本（渲染未完成时的占位显示）。"""
    text = str(text or "")

    def convert_math(match: re.Match[str]) -> str:
        body = match.group(1)
        for command, char in _MATH_PLAIN_RULES:
            body = body.replace(command, char)
        body = re.sub(r"\\(?:mathrm|mathbf|mathit|text|operatorname)\{([^{}]*)\}", r"\1", body)
        body = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", body)
        body = re.sub(r"\\(?:left|right|,|;|!|quad|qquad)", "", body)
        body = re.sub(r"\\[a-zA-Z]+", "", body)
        body = body.replace("{", "").replace("}", "")
        return body

    return re.sub(r"\$([^$]*)\$", convert_math, text)


def _latex_document(text: str, *, fontsize: float, color: str, bold: bool, padding: float = 1.5) -> LatexDocument:
    """构造单行标签的文档。"""
    document = LatexDocument(fontsize=fontsize, color=color, padding=padding, background=None)
    document.add(
        TextBlock(
            text=text,
            fontsize=fontsize,
            color=color,
            bold=bold,
            space_before=0.0,
            space_after=0.0,
            line_spacing=1.1,
        )
    )
    return document


def _buffer_to_pixmap(buffer: Any, rect: tuple[int, int, int, int], dpi: int) -> QPixmap:
    """从整图缓冲区里裁一块并转成 QPixmap。"""
    x, y, width, height = rect
    band = np.ascontiguousarray(buffer[y : y + height, x : x + width])
    if band.size == 0:
        return QPixmap()
    image = QImage(band.tobytes(), band.shape[1], band.shape[0], 4 * band.shape[1], QImage.Format_RGBA8888).copy()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(dpi / 96.0)
    return pixmap


def render_latex_pixmap(
    text: str,
    *,
    fontsize: float = 10.5,
    color: str = "#000000",
    bold: bool = False,
    dpi: int = 192,
    padding: float = 1.5,
) -> QPixmap:
    """渲染单个标签（内部走批量接口，便于统一缓存）。"""
    pixmaps = render_latex_pixmaps(
        [(text, fontsize, color, bold)], dpi=dpi, padding=padding
    )
    return pixmaps[0] if pixmaps else QPixmap()


def render_latex_pixmaps(
    requests: Sequence[tuple[str, float, str, bool]],
    *,
    dpi: int = 192,
    padding: float = 1.5,
) -> list[QPixmap]:
    """**一次画完**一批标签，返回与 ``requests`` 等长的 QPixmap 列表。

    ``requests`` 的元素是 ``(文本, 字号, 颜色, 是否加粗)``。

    逐张渲染时每张都要新建 Figure + 画布并单独走一遍 Agg 绘制，实测单张
    50 ms；而排版只要 0.5 ms。这里把几十个标签竖着拼成一条「胶片」只画一次，
    55 个矩阵运算按钮的总耗时从约 9 s 降到几百毫秒，做到按钮上的符号
    **实时**用 LaTeX 渲染。
    """
    if not requests:
        return []
    documents = [
        _latex_document(text, fontsize=fontsize, color=color, bold=bold, padding=padding)
        for text, fontsize, color, bold in requests
    ]
    buffer, slices = render_strip(documents, width=8000.0, dpi=dpi, background=None)
    return [_buffer_to_pixmap(buffer, (s.x, s.y, s.width, s.height), dpi) for s in slices]


# --------------------------------------------------------------------------- #
# 磁盘缓存
# --------------------------------------------------------------------------- #
#: 缓存格式版本：改了渲染方式或排版引擎行为时递增即可让旧缓存失效
_LATEX_CACHE_VERSION = "3"


def latex_cache_dir() -> str:
    """LaTeX 位图磁盘缓存目录（无法创建时返回空串表示禁用缓存）。"""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, "TableDrawer", "latex-cache")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return ""
    return path


def _disk_cache_path(text: str, fontsize: float, color: str, bold: bool, dpi: int) -> str:
    """按「文本 + 字号 + 颜色 + 字体配置 + 引擎版本」算出缓存文件名。"""
    from core.mplsetup import font_config

    directory = latex_cache_dir()
    if not directory:
        return ""
    try:
        config = font_config()
    except Exception:
        config = {}
    payload = "\x1f".join(
        [
            _LATEX_CACHE_VERSION,
            str(text),
            f"{float(fontsize):.2f}",
            str(color),
            "1" if bold else "0",
            str(int(dpi)),
            str(config.get("preset", "")),
            ",".join(config.get("latin", []) or []),
            ",".join(config.get("cjk", []) or []),
            ",".join(config.get("mono", []) or []),
            str(config.get("mathtext", "")),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return os.path.join(directory, f"{digest}.png")


def _screen_latex_dpi() -> int:
    """按屏幕的 devicePixelRatio 选渲染分辨率。

    位图栅格化的开销大致与 dpi² 成正比：固定用 192 dpi 时一批 55 个标签的
    Agg 绘制要约 800 ms，而按实际屏幕比例（常见 1.25 → 120 dpi）只要约
    300 ms，在 HiDPI 上依然清晰。
    """
    try:
        screen = QGuiApplication.primaryScreen()
        ratio = float(screen.devicePixelRatio()) if screen is not None else 1.0
    except Exception:
        ratio = 1.0
    ratio = max(1.0, min(ratio, 3.0))
    return int(round(96.0 * ratio))


class LatexPixmapStore:
    """LaTeX 位图缓存：内存 → 磁盘 → 批量渲染。

    三级策略：

    1. **内存缓存**命中 → 立刻返回（0 ms）；
    2. **磁盘缓存**命中 → 读 PNG（几十微秒），跨进程重启也有效；
    3. 都没有 → 排进队列，用 0 间隔定时器**分块批量渲染**（一次画一块），
       避免逐个新建 Figure 的开销，也避免长时间阻塞界面。

    渲染完成前控件先用纯文本占位，因此界面从不会卡住。配合磁盘缓存，
    第二次启动时 55 个运算按钮的符号是**瞬间**出现的。
    """

    #: 每个定时器周期最多渲染多少个标签（太小会多次重建 Figure，太大又会卡）
    chunk_size = 16

    def __init__(self, dpi: int | None = None) -> None:
        self._dpi = int(dpi) if dpi else _screen_latex_dpi()
        self._cache: dict[tuple, QPixmap] = {}
        self._queue: list[tuple] = []
        self._timer = QTimer()
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self._process_batch)
        self.render_count = 0        # 真正走渲染的批次数（测试用）

    @property
    def dpi(self) -> int:
        return self._dpi

    @staticmethod
    def key(text: str, fontsize: float, color: str, bold: bool) -> tuple:
        return (str(text), round(float(fontsize), 2), str(color), bool(bold))

    def cached(self, text: str, fontsize: float, color: str, bold: bool = False) -> QPixmap | None:
        return self._cache.get(self.key(text, fontsize, color, bold))

    def _load_from_disk(self, text: str, fontsize: float, color: str, bold: bool) -> QPixmap | None:
        path = _disk_cache_path(text, fontsize, color, bold, self._dpi)
        if not path or not os.path.exists(path):
            return None
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return None
        pixmap.setDevicePixelRatio(self._dpi / 96.0)
        return pixmap

    def _save_to_disk(self, text: str, fontsize: float, color: str, bold: bool, pixmap: QPixmap) -> None:
        path = _disk_cache_path(text, fontsize, color, bold, self._dpi)
        if not path:
            return
        try:
            pixmap.save(path, "PNG")
        except Exception:
            pass

    def request(
        self,
        text: str,
        *,
        fontsize: float,
        color: str,
        bold: bool = False,
        callback: Callable[[QPixmap], None],
    ) -> None:
        """请求一个标签的位图；命中缓存立刻回调，否则排队等批量渲染。"""
        key = self.key(text, fontsize, color, bold)
        pixmap = self._cache.get(key)
        if pixmap is None:
            pixmap = self._load_from_disk(text, fontsize, color, bold)
            if pixmap is not None:
                self._cache[key] = pixmap
        if pixmap is not None:
            callback(pixmap)
            return
        self._queue.append((key, text, fontsize, color, bold, callback))
        if not self._timer.isActive():
            self._timer.start()

    def prefetch(self, requests: Sequence[tuple[str, float, str, bool]]) -> int:
        """预热一批标签（无控件回调），返回其中已缓存的个数。"""
        hits = 0
        for text, fontsize, color, bold in requests:
            key = self.key(text, fontsize, color, bold)
            if key in self._cache:
                hits += 1
                continue
            pixmap = self._load_from_disk(text, fontsize, color, bold)
            if pixmap is not None:
                self._cache[key] = pixmap
                hits += 1
        return hits

    def pending(self) -> int:
        return len(self._queue)

    def _process_batch(self) -> None:
        """把队列里的一批标签**一次画完**；还有剩余就再排一个周期。

        分块是为了既摊销 Figure 的创建开销，又不让单次阻塞太久。
        """
        if not self._queue:
            return
        take = self._queue[: self.chunk_size]
        self._queue = self._queue[self.chunk_size :]

        # 同一块里可能有重复请求（多个控件用同一个标签），去重后再渲染
        unique: dict[tuple, tuple[str, float, str, bool]] = {}
        for key, text, fontsize, color, bold, _cb in take:
            unique.setdefault(key, (text, fontsize, color, bold))

        requests = list(unique.values())
        rendered: dict[tuple, QPixmap] = {}
        try:
            pixmaps = render_latex_pixmaps(requests, dpi=self._dpi)
        except Exception:
            pixmaps = []
        for (text, fontsize, color, bold), pixmap in zip(requests, pixmaps):
            if not pixmap.isNull():
                key = self.key(text, fontsize, color, bold)
                self._cache[key] = pixmap
                rendered[key] = pixmap
                self._save_to_disk(text, fontsize, color, bold, pixmap)
        self.render_count += 1

        for key, text, fontsize, color, bold, callback in take:
            pixmap = rendered.get(key)
            if pixmap is None:
                continue
            try:
                callback(pixmap)
            except RuntimeError:
                pass          # 控件已被销毁

        if self._queue:
            self._timer.start()

    def clear(self, *, disk: bool = False) -> None:
        """清空内存缓存（``disk=True`` 时连同磁盘缓存一起删掉）。"""
        self._cache.clear()
        self._queue.clear()
        if disk:
            directory = latex_cache_dir()
            if directory:
                for name in os.listdir(directory):
                    if name.endswith(".png"):
                        try:
                            os.remove(os.path.join(directory, name))
                        except OSError:
                            pass


_STORE: LatexPixmapStore | None = None


def latex_pixmaps() -> LatexPixmapStore:
    """取得全局 LaTeX 位图缓存（延迟创建，确保 QApplication 已存在）。"""
    global _STORE
    if _STORE is None:
        _STORE = LatexPixmapStore()
    return _STORE


class LatexLabel(QLabel):
    """用 LaTeX 渲染 ``$...$`` 的标签（透明背景，随主题换色）。"""

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        fontsize: float = 11.0,
        color: str = "#000000",
        bold: bool = False,
    ) -> None:
        super().__init__(parent)
        self._latex = text
        self._fontsize = float(fontsize)
        self._color = color
        self._bold = bold
        self.setText(latex_to_plain(text))
        self.refresh()

    def set_latex(self, text: str) -> None:
        self._latex = text
        self.setText(latex_to_plain(text))
        self.refresh()

    def set_latex_color(self, color: str) -> None:
        if color and color != self._color:
            self._color = color
            self.refresh()

    def refresh(self) -> None:
        latex_pixmaps().request(
            self._latex,
            fontsize=self._fontsize,
            color=self._color,
            bold=self._bold,
            callback=self._apply,
        )

    def _apply(self, pixmap: QPixmap) -> None:
        self.setText("")
        self.setPixmap(pixmap)
        self.setFixedSize(
            QSize(max(int(pixmap.width() / pixmap.devicePixelRatio()), 1),
                  max(int(pixmap.height() / pixmap.devicePixelRatio()), 1))
        )


class LatexButton(QPushButton):
    """按钮文案用 LaTeX 排版（图标形式），点击行为与普通按钮一致。

    * ``plain`` 是渲染完成前的占位文本，也是无障碍名称与工具提示；
    * 主题切换后调用 :meth:`refresh` 会用新颜色重新渲染。
    """

    def __init__(
        self,
        latex_text: str,
        plain_text: str = "",
        parent: QWidget | None = None,
        *,
        fontsize: float = 10.5,
        color: str = "#000000",
        bold: bool = False,
        tooltip: str = "",
    ) -> None:
        plain = plain_text or latex_to_plain(latex_text)
        super().__init__(plain, parent)
        self._latex = latex_text
        self._plain = plain
        self._fontsize = float(fontsize)
        self._color = color
        self._bold = bold
        self.setAccessibleName(plain)
        self.setToolTip(tooltip or plain)
        self.setMinimumHeight(28)
        self.refresh()

    def latex_text(self) -> str:
        return self._latex

    def plain_text(self) -> str:
        return self._plain

    def set_latex_color(self, color: str) -> None:
        if color and color != self._color:
            self._color = color
            self.refresh()

    def refresh(self) -> None:
        """重新请求渲染（主题/字体变化后调用）。"""
        self.setIcon(QIcon())
        self.setText(self._plain)
        latex_pixmaps().request(
            self._latex,
            fontsize=self._fontsize,
            color=self._color,
            bold=self._bold,
            callback=self._apply,
        )

    def _apply(self, pixmap: QPixmap) -> None:
        ratio = pixmap.devicePixelRatio() or 1.0
        self.setIcon(QIcon(pixmap))
        self.setIconSize(
            QSize(max(int(pixmap.width() / ratio), 1), max(int(pixmap.height() / ratio), 1))
        )
        self.setText("")


# --------------------------------------------------------------------------- #
# matplotlib 画布
# --------------------------------------------------------------------------- #
class MplCanvas(QWidget):
    """封装 ``FigureCanvasQTAgg`` 与导航工具栏。"""

    def __init__(self, parent: QWidget | None = None, *, toolbar: bool = True, figsize=(8, 5), dpi=110):
        super().__init__(parent)
        self._toolbar_enabled = bool(toolbar)
        self._figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasQTAgg(self._figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(240, 160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toolbar: NavigationToolbar2QT | None = None
        if self._toolbar_enabled:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(QSize(16, 16))
            layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    @property
    def figure(self) -> Figure:
        return self._figure

    def set_figure(self, figure: Figure) -> None:
        """替换底层 figure。

        画布与导航工具栏必须**一起重建**：工具栏在构造时通过
        ``mpl_connect`` 绑定了旧画布的事件，只换画布会让平移/缩放失效。
        """
        layout = self.layout()
        for widget in (self.toolbar, self.canvas):
            if widget is not None:
                layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()

        self._figure = figure
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(240, 160)

        if self._toolbar_enabled:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.toolbar.setIconSize(QSize(16, 16))
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None
        layout.addWidget(self.canvas, 1)
        self.canvas.draw_idle()

    def refresh(self) -> None:
        """按画布控件的实际大小对齐 Figure，然后重绘。

        复用同一个 Figure 时不会再触发 ``FigureCanvasQT.resizeEvent``，
        matplotlib 就失去了「figure 跟着控件走」的自动同步；不同步的话，
        图会缩在画布角落、字号比例也不对。

        这里严格照抄 matplotlib 自己的算法
        （``w = 控件宽 × device_pixel_ratio``，再 ``set_size_inches(w / dpi)``），
        所以 HiDPI 屏上也能精确对齐。注意不能用
        ``FigureCanvasBase.get_width_height()`` —— 它返回的是 **Figure** 的尺寸。
        """
        try:
            ratio = float(getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0)
            dpi = float(self._figure.get_dpi()) or 110.0
            width = self.canvas.width() * ratio
            height = self.canvas.height() * ratio
            if width > 20 and height > 20:
                current = self._figure.get_size_inches() * dpi
                if abs(width - current[0]) > 2 or abs(height - current[1]) > 2:
                    self._figure.set_size_inches(
                        width / dpi, height / dpi, forward=False
                    )
        except Exception:
            pass
        self.canvas.draw_idle()

    def save(self, path: str, **kwargs: Any) -> str:
        """按扩展名保存当前图，返回真实路径。"""
        self._figure.savefig(path, **kwargs)
        return path

    def copy_to_clipboard(self) -> bool:
        """把当前图以位图形式复制到剪贴板。"""
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            FigureCanvasAgg(self._figure)
            buffer = io.BytesIO()
            self._figure.savefig(buffer, format="png", dpi=150)
            image = QImage.fromData(buffer.getvalue(), "PNG")
            if image.isNull():
                return False
            QGuiApplication.clipboard().setImage(image)
            return True
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# LaTeX 结果视图
# --------------------------------------------------------------------------- #
class LatexView(QScrollArea):
    """可滚动、可缩放的 LaTeX 文档视图。

    内容按视口宽度重新排版（自动折行 / 过宽公式自动缩小），
    缩放通过提高渲染分辨率实现，因此放大后依然锐利。
    """

    zoomChanged = Signal(float)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        base_dpi: int = 96,
        background: str = "#ffffff",
    ):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # 垂直滚动条**常驻**：否则「内容变高 → 出现滚动条 → 视口变窄 → 重新折行
        # → 高度又变 → 滚动条消失」会形成来回抖动的闭环。
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._base_dpi = int(base_dpi)
        self._background = background
        self._zoom = 1.0
        self._document: LatexDocument | None = None
        self._last_width = -1
        self._style: ResultStyle | None = None

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._canvas = FigureCanvasQTAgg(Figure(figsize=(4, 2), dpi=self._base_dpi))
        self._canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._canvas.setMinimumSize(20, 20)
        self._layout.addWidget(self._canvas, 0, Qt.AlignTop | Qt.AlignLeft)
        self.setWidget(container)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._render)

    # -- 内容 -------------------------------------------------------------- #
    def set_document(self, document: LatexDocument | None) -> None:
        """设置要渲染的文档。"""
        self._document = document
        self._last_width = -1
        self._timer.start()

    def set_sheet(self, sheet: Any, style: ResultStyle | None = None) -> None:
        """直接渲染一个 :class:`core.spec.ResultSheet`。"""
        if style is not None:
            self._style = style
        effective = self._style or ResultStyle()
        effective.background = self._background
        self.set_document(document_for_sheet(sheet, effective))

    def set_blocks(self, blocks: Sequence[Any], fontsize: float = 12.0) -> None:
        """直接渲染一组文档块。"""
        document = LatexDocument(fontsize=fontsize, background=self._background)
        document.extend(blocks)
        self.set_document(document)

    def set_message(self, text: str, *, muted: bool = True) -> None:
        """显示一条提示文本（用于空状态或错误）。"""
        color = (self._style.note_color if self._style else "#6b7280") if muted else None
        document = LatexDocument(fontsize=12.0, background=self._background)
        document.add(TextBlock(text=text, color=color, italic=muted, space_before=0.0))
        self.set_document(document)

    def set_background(self, color: str) -> None:
        self._background = color
        self._last_width = -1
        self._timer.start()

    def document(self) -> LatexDocument | None:
        return self._document

    # -- 缩放 -------------------------------------------------------------- #
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, value: float) -> None:
        value = max(0.4, min(float(value), 4.0))
        if abs(value - self._zoom) < 1e-3:
            return
        self._zoom = value
        self._last_width = -1
        self._timer.start()
        self.zoomChanged.emit(self._zoom)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.15)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.15)

    def reset_zoom(self) -> None:
        self.set_zoom(1.0)

    # -- 渲染 -------------------------------------------------------------- #
    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._timer.start()

    def _replace_canvas(self, figure: Figure) -> None:  # 兼容旧调用：现在改为复用同一画布
        self._render()

    def _render(self) -> None:
        """按当前视口宽度重新排版并绘制。

        复用同一个 ``Figure`` 与 ``FigureCanvasQTAgg``（只 ``clear()`` 重画），
        这样重排时不会重建 Qt 控件，界面不会跳。
        """
        if self._document is None:
            return
        viewport_width = max(self.viewport().width(), 160)
        # 布局宽度以「点」为单位，与缩放无关；缩放只改变渲染分辨率
        width_pt = viewport_width * 72.0 / self._base_dpi
        if abs(width_pt - self._last_width) < 1.0:
            return
        self._last_width = width_pt

        dpi = int(round(self._base_dpi * self._zoom))
        try:
            render_document(
                self._document,
                width_pt,
                dpi=dpi,
                background=self._background,
                figure=self._canvas.figure,
            )
        except Exception as exc:  # 渲染失败时给出可见反馈而不是崩溃
            fallback = LatexDocument(fontsize=12.0, background=self._background)
            fallback.add(TextBlock(text=f"渲染失败：{exc}", color="#cf222e", space_before=0.0))
            render_document(
                fallback,
                width_pt,
                dpi=dpi,
                background=self._background,
                figure=self._canvas.figure,
            )

        figure = self._canvas.figure
        pixel_w = max(int(round(figure.get_figwidth() * figure.dpi)), 20)
        pixel_h = max(int(round(figure.get_figheight() * figure.dpi)), 20)
        self._canvas.setFixedSize(pixel_w, pixel_h)
        self._canvas.draw_idle()

    # -- 导出 / 复制 ------------------------------------------------------- #
    def _current_figure(self) -> Figure | None:
        return self._canvas.figure if self._canvas is not None else None

    def to_svg(self) -> str:
        figure = self._current_figure()
        return figure_to_svg(figure) if figure is not None else ""

    def export_svg(self, path: str) -> bool:
        figure = self._current_figure()
        if figure is None:
            return False
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(figure_to_svg(figure))
        return True

    def export_png(self, path: str, dpi: int = 200) -> bool:
        figure = self._current_figure()
        if figure is None:
            return False
        figure.savefig(path, format="png", dpi=dpi, facecolor=self._background)
        return True

    def copy_image(self) -> bool:
        """把当前结果复制为图片。"""
        figure = self._current_figure()
        if figure is None:
            return False
        try:
            data = figure_to_png_bytes(figure, dpi=200, facecolor=self._background)
            image = QImage.fromData(data, "PNG")
            if image.isNull():
                return False
            QGuiApplication.clipboard().setImage(image)
            return True
        except Exception:
            return False

    def copy_svg_text(self) -> bool:
        svg = self.to_svg()
        if not svg:
            return False
        QGuiApplication.clipboard().setText(svg)
        return True


# --------------------------------------------------------------------------- #
# 数据表
# --------------------------------------------------------------------------- #
class _PasteTableWidget(QTableWidget):
    """支持从剪贴板粘贴的表格基类（Tab / 逗号 / 分号分隔均可）。

    另外提供 :attr:`edgeHandler`：当光标停在最后一行按 ↓、或最后一列按 → 时，
    先把机会交给外部处理（用于「自动扩展行列」）；返回 ``True`` 表示事件已被消费。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: ``(row, col, key) -> bool``；返回 True 表示已处理并吞掉该按键
        self.edgeHandler: Callable[[int, int, int], bool] | None = None

    def keyPressEvent(self, event):  # noqa: N802
        if event.matches(QKeySequence.Paste) and self._handle_paste():
            event.accept()
            return
        if self.edgeHandler is not None and event.modifiers() in (Qt.NoModifier, Qt.KeypadModifier):
            if event.key() in (Qt.Key_Down, Qt.Key_Right):
                if self.edgeHandler(self.currentRow(), self.currentColumn(), event.key()):
                    event.accept()
                    return
        super().keyPressEvent(event)

    def _handle_paste(self) -> bool:
        text = QGuiApplication.clipboard().text()
        grid = clipboard_to_grid(text)
        if not grid:
            return False
        row = max(self.currentRow(), 0)
        col = max(self.currentColumn(), 0)
        return self.paste_grid(grid, row, col)

    def paste_grid(self, grid: list[list[str]], row: int, col: int) -> bool:
        """把二维字符串表格写入指定位置，必要时扩展行列。"""
        needed_rows = row + len(grid)
        needed_cols = col + max((len(r) for r in grid), default=0)
        if needed_rows > self.rowCount():
            self.setRowCount(needed_rows)
        if needed_cols > self.columnCount():
            self.setColumnCount(needed_cols)
        for i, line in enumerate(grid):
            for j, cell in enumerate(line):
                item = self.item(row + i, col + j)
                if item is None:
                    item = QTableWidgetItem()
                    self.setItem(row + i, col + j, item)
                item.setText(str(cell).strip())
        return True


class DataTableWidget(_PasteTableWidget):
    """DataFrame 表格：可编辑、可粘贴，双向同步回 DataFrame。"""

    frameEdited = Signal()

    def __init__(self, parent: QWidget | None = None, *, editable: bool = True, max_rows: int = 5000):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._loading = False
        self._editable = editable
        self._max_rows = max_rows
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.AnyKeyPressed
            if editable
            else QAbstractItemView.NoEditTriggers
        )
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.horizontalHeader().setStretchLastSection(False)
        self.verticalHeader().setDefaultSectionSize(24)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.itemChanged.connect(self._on_item_changed)

    # -- 数据 -------------------------------------------------------------- #
    def set_frame(self, frame: pd.DataFrame | None) -> None:
        self._loading = True
        try:
            self.clear()
            self._frame = pd.DataFrame() if frame is None else frame
            if self._frame.empty and self._frame.shape[1] == 0:
                self.setRowCount(0)
                self.setColumnCount(0)
                return
            rows = min(len(self._frame), self._max_rows)
            columns = list(self._frame.columns)
            self.setRowCount(rows)
            self.setColumnCount(len(columns))
            self.setHorizontalHeaderLabels([str(c) for c in columns])
            self.setVerticalHeaderLabels([str(i + 1) for i in range(rows)])
            for i in range(rows):
                for j, name in enumerate(columns):
                    value = self._frame.iat[i, j]
                    self.setItem(i, j, QTableWidgetItem(_display_value(value)))
            self.resizeColumnsToContents()
            for j in range(self.columnCount()):
                self.setColumnWidth(j, min(max(self.columnWidth(j) + 12, 64), 260))
        finally:
            self._loading = False

    def frame(self) -> pd.DataFrame:
        return self._frame

    def display_note(self) -> str:
        """当数据被截断显示时返回说明文本。"""
        if len(self._frame) > self._max_rows:
            return f"表格仅显示前 {self._max_rows} 行（共 {len(self._frame)} 行）"
        return ""

    # -- 编辑 -------------------------------------------------------------- #
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or not self._editable:
            return
        row, col = item.row(), item.column()
        if row >= len(self._frame) or col >= self._frame.shape[1]:
            return
        name = self._frame.columns[col]
        text = item.text().strip()
        try:
            value = parse_number(text)
        except ValueError:
            if text == "":
                value = np.nan
            else:
                value = text
        if isinstance(value, complex) and value.imag == 0:
            value = value.real
        try:
            self._frame.iat[row, col] = value
        except (TypeError, ValueError):
            self._frame[name] = self._frame[name].astype(object)
            self._frame.iat[row, col] = value
        self.frameEdited.emit()

    def paste_grid(self, grid: list[list[str]], row: int, col: int) -> bool:
        if not self._editable:
            return False
        ok = super().paste_grid(grid, row, col)
        if ok:
            self._sync_frame_from_cells()
        return ok

    def _sync_frame_from_cells(self) -> None:
        """粘贴后把整张表同步回 DataFrame（列可能被扩展）。"""
        rows = min(self.rowCount(), len(self._frame)) if len(self._frame) else self.rowCount()
        columns = [self.horizontalHeaderItem(j).text() if self.horizontalHeaderItem(j) else f"列{j + 1}"
                   for j in range(self.columnCount())]
        data: dict[str, list[Any]] = {name: [] for name in columns}
        for i in range(self.rowCount()):
            for j, name in enumerate(columns):
                item = self.item(i, j)
                text = item.text().strip() if item is not None else ""
                try:
                    value = parse_number(text) if text else np.nan
                except ValueError:
                    value = text
                if isinstance(value, complex) and value.imag == 0:
                    value = value.real
                data[name].append(value)
        self._loading = True
        try:
            self._frame = pd.DataFrame(data)
            self.setVerticalHeaderLabels([str(i + 1) for i in range(self.rowCount())])
        finally:
            self._loading = False
        self.frameEdited.emit()


def _display_value(value: Any) -> str:
    """单元格显示文本：整数不带小数点，浮点保留合理位数。"""
    if value is None:
        return ""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.6g}"
    if isinstance(value, (np.floating,)):
        return _display_value(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, (np.bool_,)):
        return "True" if value else "False"
    return str(value)


# --------------------------------------------------------------------------- #
# 矩阵编辑器
# --------------------------------------------------------------------------- #
#: 空白单元格填充策略：``键 -> 中文说明``
FILL_POLICIES: dict[str, str] = {
    "zero": "填 0",
    "one": "填 1",
    "nan": "填 NaN（缺失值）",
    "previous": "向左复制（沿用左侧最近的值）",
    "interpolate": "按行线性插值",
    "strict": "不填充 —— 有空单元格就报错",
}


def _parse_cell(text: str) -> Any:
    """单元格文本 -> 数值；空返回 ``None``。"""
    text = str(text).strip()
    if not text:
        return None
    return parse_number(text)


def grid_to_matrix(
    cells: Sequence[Sequence[str]],
    *,
    fill: str = "zero",
    trim: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把字符串网格转成 NumPy 矩阵，并按策略填充空白。

    ``trim=True`` 时先「识别有效区域」：找出最后一个非空单元格所在的行与列，
    把右下角的空白整块裁掉——这样用户在一个大表格里填了 3×4 就得到 3×4 的矩阵，
    而不需要手工调尺寸。

    返回 ``(矩阵, 说明信息)``；``fill="strict"`` 且存在空白时抛 :class:`ValueError`。
    """
    rows = [list(row) for row in (cells or [])]
    if not rows:
        raise ValueError("矩阵是空的。")
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        raise ValueError("矩阵是空的。")
    for row in rows:
        row.extend([""] * (width - len(row)))

    total_rows, total_cols = len(rows), width
    if trim:
        last_row = last_col = -1
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if str(cell).strip():
                    last_row = max(last_row, i)
                    last_col = max(last_col, j)
        if last_row < 0:
            raise ValueError("矩阵中还没有填写任何数值。")
        rows = [row[: last_col + 1] for row in rows[: last_row + 1]]

    n_rows, n_cols = len(rows), len(rows[0])
    values: list[list[Any]] = [[None] * n_cols for _ in range(n_rows)]
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            try:
                values[i][j] = _parse_cell(cell)
            except ValueError as exc:
                raise ValueError(f"第 {i + 1} 行第 {j + 1} 列：{exc}") from exc
    blanks = [(i, j) for i in range(n_rows) for j in range(n_cols) if values[i][j] is None]

    if blanks and fill == "strict":
        preview = "、".join(f"({i + 1},{j + 1})" for i, j in blanks[:8])
        more = f" 等 {len(blanks)} 处" if len(blanks) > 8 else ""
        raise ValueError(f"以下单元格为空，请填写或改用其它填充方式：{preview}{more}")

    if blanks:
        if fill == "one":
            for i, j in blanks:
                values[i][j] = 1
        elif fill == "nan":
            for i, j in blanks:
                values[i][j] = float("nan")
        elif fill == "previous":
            for i in range(n_rows):
                carried: Any = 0
                for j in range(n_cols):
                    if values[i][j] is None:
                        values[i][j] = carried
                    else:
                        carried = values[i][j]
        elif fill == "interpolate":
            for i in range(n_rows):
                known = [j for j in range(n_cols) if values[i][j] is not None]
                if not known:
                    for j in range(n_cols):
                        values[i][j] = 0
                    continue
                xs = np.array(known, dtype=float)
                ys = np.array([float(np.real(values[i][j])) for j in known], dtype=float)
                for j in range(n_cols):
                    if values[i][j] is None:
                        values[i][j] = float(np.interp(j, xs, ys))
        else:                     # "zero" 及未知策略
            for i, j in blanks:
                values[i][j] = 0

    matrix = np.array(values, dtype=complex)
    if np.allclose(matrix.imag, 0.0):
        matrix = matrix.real
    info = {
        "rows": n_rows,
        "cols": n_cols,
        "blanks": len(blanks),
        "source_rows": total_rows,
        "source_cols": total_cols,
        "trimmed": bool(trim and (n_rows != total_rows or n_cols != total_cols)),
        "fill": fill,
    }
    return matrix, info


def describe_grid(info: dict[str, Any]) -> str:
    """把 :func:`grid_to_matrix` 的说明信息转成中文描述。"""
    parts = [f"有效区域 {info['rows']} × {info['cols']}"]
    if info.get("trimmed"):
        parts.append(f"（表格 {info['source_rows']} × {info['source_cols']}，已裁掉右下空白）")
    if info["blanks"]:
        parts.append(f"空白 {info['blanks']} 处 → {FILL_POLICIES.get(info['fill'], info['fill'])}")
    else:
        parts.append("没有空白单元格")
    return "；".join(parts)


def result_style_for(background: str) -> ResultStyle:
    """按背景色亮度挑一套 LaTeX 结果配色（深色底用浅色字）。"""
    from core.latex.results import DARK_RESULT_STYLE, LIGHT_RESULT_STYLE

    dark = False
    try:
        text = str(background).lstrip("#")
        red, green, blue = (int(text[i : i + 2], 16) for i in (0, 2, 4))
        dark = (0.299 * red + 0.587 * green + 0.114 * blue) < 128
    except Exception:
        dark = False
    style = DARK_RESULT_STYLE if dark else LIGHT_RESULT_STYLE
    style.background = background
    return style


class MatrixEditor(QWidget):
    """矩阵输入网格：支持粘贴、尺寸调整与常用矩阵生成。"""

    matrixChanged = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        rows: int = 3,
        cols: int = 3,
        label: str = "A",
        auto_grow: bool = True,
        fill_policy: str = "zero",
        trim: bool = True,
        max_size: int = 600,
    ):
        super().__init__(parent)
        self._label = label
        self._loading = False
        self._suppress_grow = False
        self.auto_grow = bool(auto_grow)
        self.fill_policy = fill_policy if fill_policy in FILL_POLICIES else "zero"
        self.trim = bool(trim)
        self.max_size = int(max_size)
        self._palette_color = "#000000"
        self._palette_bg = "#ffffff"
        self.last_info: dict[str, Any] = {}

        self.table = _PasteTableWidget(rows, cols)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setMinimumHeight(120)
        self.table.itemChanged.connect(self._on_changed)
        # 自动扩展只由「方向键走到边界」触发，填写数据不会让表格变大
        self.table.edgeHandler = self._on_edge_key
        self._refresh_headers()

        # 尺寸控制
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, self.max_size)
        self.rows_spin.setValue(rows)
        self.rows_spin.setPrefix("行 ")
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, self.max_size)
        self.cols_spin.setValue(cols)
        self.cols_spin.setPrefix("列 ")
        apply_button = QPushButton("应用尺寸")
        apply_button.clicked.connect(lambda: self.resize_matrix(self.rows_spin.value(), self.cols_spin.value()))

        self.grow_check = QCheckBox("自动扩展")
        self.grow_check.setChecked(self.auto_grow)
        self.grow_check.setToolTip(
            "勾选后，当光标停在最后一行按 ↓、或最后一列按 → 时自动新增一行/一列，\n"
            "并把光标移到新出现的空白格——可以像无限大的表格一样一直填下去。\n"
            "注意：单纯填写数据或点击单元格**不会**让表格变大。"
        )
        self.grow_check.toggled.connect(self._on_grow_toggled)

        expand_button = QPushButton("放大…")
        expand_button.setToolTip("在独立的大窗口里编辑矩阵（可全屏），并实时预览识别结果")
        expand_button.clicked.connect(self.open_fullscreen)

        size_row = QHBoxLayout()
        size_row.setSpacing(6)
        size_row.addWidget(self.rows_spin)
        size_row.addWidget(self.cols_spin)
        size_row.addWidget(apply_button)
        size_row.addStretch(1)
        size_row.addWidget(expand_button)

        grow_row = QHBoxLayout()
        grow_row.setSpacing(6)
        grow_row.addWidget(self.grow_check)
        grow_row.addStretch(1)

        # 快捷生成
        quick = QHBoxLayout()
        quick.setSpacing(6)
        for text, slot in (
            ("单位阵", self.set_identity),
            ("零阵", self.set_zeros),
            ("全 1", self.set_ones),
            ("随机", self.set_random),
            ("转置", self.transpose),
            ("清空", self.clear_values),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            quick.addWidget(button)
        quick.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(size_row)
        layout.addLayout(grow_row)
        layout.addLayout(quick)
        layout.addWidget(self.table, 1)

    # -- 内部 -------------------------------------------------------------- #
    def _refresh_headers(self) -> None:
        self.table.setHorizontalHeaderLabels([str(j + 1) for j in range(self.table.columnCount())])
        self.table.setVerticalHeaderLabels([str(i + 1) for i in range(self.table.rowCount())])

    def _on_changed(self, _item: QTableWidgetItem) -> None:
        """单元格内容变化。

        注意：**写入数据不会触发表格扩展**——否则用户每填最后一格都会多出一行空行。
        扩展只由「方向键移动到空白位置」触发，见 :meth:`_on_edge_key`。
        """
        if self._loading:
            return
        self.matrixChanged.emit()

    def _on_grow_toggled(self, checked: bool) -> None:
        self.auto_grow = bool(checked)

    def _cell_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item is not None else ""

    def _grow_to(self, row: int, col: int) -> bool:
        """在下边界之外补一行/一列（受 ``max_size`` 限制）。"""
        rows, cols = self.table.rowCount(), self.table.columnCount()
        grew = False
        self._loading = True
        try:
            if row >= rows - 1 and rows < self.max_size:
                self.table.setRowCount(rows + 1)
                grew = True
            if col >= cols - 1 and cols < self.max_size:
                self.table.setColumnCount(cols + 1)
                grew = True
        finally:
            self._loading = False
        if grew:
            self._refresh_headers()
            self.rows_spin.setValue(self.table.rowCount())
            self.cols_spin.setValue(self.table.columnCount())
        return grew

    def _on_edge_key(self, row: int, col: int, key: int) -> bool:
        """方向键走到边界时按需扩展。

        只有「↓ 停在最后一行」或「→ 停在最后一列」才扩展，并把光标移到新出现的
        空白单元格；单纯填写数据或点击单元格都不会让表格变大。
        """
        if not self.auto_grow or row < 0 or col < 0:
            return False
        rows, cols = self.table.rowCount(), self.table.columnCount()
        at_bottom = key == Qt.Key_Down and row >= rows - 1
        at_right = key == Qt.Key_Right and col >= cols - 1
        if not (at_bottom or at_right):
            return False
        if not self._grow_to(row, col):
            return False
        self._suppress_grow = True
        try:
            if at_bottom:
                self.table.setCurrentCell(min(row + 1, self.table.rowCount() - 1), col)
            else:
                self.table.setCurrentCell(row, min(col + 1, self.table.columnCount() - 1))
        finally:
            self._suppress_grow = False
        return True

    def _grow_if_needed(self, row: int, col: int, has_content: bool) -> None:
        """兼容旧调用：现在改为「方向键触发的扩展」，这里只在边界补足。"""
        if has_content and self.auto_grow:
            self._grow_to(row, col)

    def _fill(self, func) -> None:
        self._loading = True
        try:
            for i in range(self.table.rowCount()):
                for j in range(self.table.columnCount()):
                    value = func(i, j)
                    item = self.table.item(i, j)
                    if item is None:
                        item = QTableWidgetItem()
                        self.table.setItem(i, j, item)
                    item.setText(_display_value(value))
        finally:
            self._loading = False
        self.matrixChanged.emit()

    # -- 公开 API ---------------------------------------------------------- #
    def set_label(self, label: str) -> None:
        self._label = label

    def label(self) -> str:
        return self._label

    def resize_matrix(self, rows: int, cols: int) -> None:
        rows, cols = max(1, int(rows)), max(1, int(cols))
        self._loading = True
        try:
            old = self.matrix(allow_empty=True)
            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            self._refresh_headers()
            for i in range(rows):
                for j in range(cols):
                    value = ""
                    if old is not None and i < old.shape[0] and j < old.shape[1]:
                        value = _display_value(old[i, j])
                    item = self.table.item(i, j)
                    if item is None:
                        item = QTableWidgetItem()
                        self.table.setItem(i, j, item)
                    item.setText(value)
            self.rows_spin.setValue(rows)
            self.cols_spin.setValue(cols)
        finally:
            self._loading = False
        self.matrixChanged.emit()

    def matrix(self, *, allow_empty: bool = True, fill: str | None = None, trim: bool | None = None) -> np.ndarray | None:
        """按当前策略读取矩阵。

        * ``fill``：空白单元格的填充方式，``None`` 表示用编辑器当前设置；
        * ``trim``：是否自动裁剪到「有效区域」，``None`` 表示用编辑器当前设置；
        * 整个表格为空且 ``allow_empty`` 时返回 ``None``。

        解析失败或 ``fill="strict"`` 遇到空白时抛 :class:`ValueError`（含中文原因）。
        """
        cells = self.cells()
        if not any(cell for row in cells for cell in row):
            if allow_empty:
                return None
            raise ValueError("矩阵中还没有填写任何数值。")
        resolved_fill = fill if fill is not None else self.fill_policy
        resolved_trim = self.trim if trim is None else bool(trim)
        matrix, info = grid_to_matrix(cells, fill=resolved_fill, trim=resolved_trim)
        self.last_info = info
        return matrix

    # -- 网格读写 ---------------------------------------------------------- #
    def cells(self) -> list[list[str]]:
        """当前网格的原始文本（含空白单元格）。"""
        grid: list[list[str]] = []
        for i in range(self.table.rowCount()):
            row: list[str] = []
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                row.append(item.text().strip() if item is not None else "")
            grid.append(row)
        return grid

    def set_cells(self, cells: Sequence[Sequence[str]], *, reset_size: bool = True) -> None:
        """整块写入网格（``reset_size=True`` 时按内容调整行列数）。"""
        grid = [list(row) for row in (cells or [])]
        if not grid:
            return
        columns = max(len(row) for row in grid)
        self._loading = True
        try:
            if reset_size:
                self.table.setRowCount(max(len(grid), 1))
                self.table.setColumnCount(max(columns, 1))
            for i in range(self.table.rowCount()):
                for j in range(self.table.columnCount()):
                    text = str(grid[i][j]) if i < len(grid) and j < len(grid[i]) else ""
                    item = self.table.item(i, j)
                    if item is None:
                        self.table.setItem(i, j, QTableWidgetItem(text))
                    else:
                        item.setText(text)
            self._refresh_headers()
            self.rows_spin.setValue(self.table.rowCount())
            self.cols_spin.setValue(self.table.columnCount())
        finally:
            self._loading = False
        self.matrixChanged.emit()

    def used_shape(self) -> tuple[int, int]:
        """「有效区域」：最后一个非空单元格所在的行数 / 列数。"""
        grid = self.cells()
        last_row = last_col = 0
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell:
                    last_row = max(last_row, i + 1)
                    last_col = max(last_col, j + 1)
        return last_row, last_col

    def trim_to_used(self) -> bool:
        """把表格尺寸收缩到有效区域，返回是否发生了变化。"""
        rows, cols = self.used_shape()
        if rows == 0 or cols == 0:
            return False
        if rows == self.table.rowCount() and cols == self.table.columnCount():
            return False
        self.resize_matrix(rows, cols)
        return True

    # -- 读取策略 ---------------------------------------------------------- #
    def set_fill_policy(self, policy: str) -> None:
        if policy in FILL_POLICIES:
            self.fill_policy = policy

    def set_trim(self, trim: bool) -> None:
        self.trim = bool(trim)

    def set_auto_grow(self, enabled: bool) -> None:
        self.auto_grow = bool(enabled)
        self.grow_check.setChecked(bool(enabled))

    def set_latex_color(self, color: str) -> None:
        """记录主题文字色（供放大编辑窗口的预览使用）。"""
        self._palette_color = color or self._palette_color

    def set_latex_background(self, color: str) -> None:
        """记录主题背景色（供放大编辑窗口的 LaTeX 预览使用）。"""
        self._palette_bg = color or self._palette_bg

    def set_latex_theme(self, color: str, background: str) -> None:
        self.set_latex_color(color)
        self.set_latex_background(background)

    def latex_color(self) -> str:
        return self._palette_color

    def latex_background(self) -> str:
        return self._palette_bg

    def open_fullscreen(self, parent: QWidget | None = None) -> bool:
        """打开放大编辑窗口；用户确认时把结果写回本编辑器。"""
        dialog = MatrixEditorDialog(parent or self.window(), editor=self, label=self._label)
        if dialog.exec() != QDialog.Accepted:
            return False
        self.set_cells(dialog.cells())
        self.set_fill_policy(dialog.fill_policy())
        self.set_trim(dialog.trim())
        self.set_auto_grow(dialog.auto_grow())
        return True

    def set_matrix(self, matrix: Any, *, precision: int = 6) -> None:
        """写入矩阵（自动调整表格尺寸）。"""
        array = np.atleast_2d(np.asarray(matrix))
        self._loading = True
        try:
            self.table.setRowCount(array.shape[0])
            self.table.setColumnCount(array.shape[1])
            self._refresh_headers()
            self.rows_spin.setValue(array.shape[0])
            self.cols_spin.setValue(array.shape[1])
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    item = QTableWidgetItem(_format_entry(array[i, j], precision))
                    self.table.setItem(i, j, item)
        finally:
            self._loading = False
        self.matrixChanged.emit()

    def text(self) -> str:
        """导出为可粘贴的纯文本（Tab 分隔）。"""
        lines = []
        for i in range(self.table.rowCount()):
            cells = []
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                cells.append(item.text().strip() if item is not None else "0")
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def set_text(self, text: str) -> None:
        """从纯文本导入。"""
        grid = clipboard_to_grid(text)
        if not grid:
            return
        cols = max(len(r) for r in grid)
        self._loading = True
        try:
            self.table.setRowCount(len(grid))
            self.table.setColumnCount(cols)
            self._refresh_headers()
            for i, line in enumerate(grid):
                for j in range(cols):
                    value = line[j].strip() if j < len(line) else ""
                    self.table.setItem(i, j, QTableWidgetItem(value))
            self.rows_spin.setValue(len(grid))
            self.cols_spin.setValue(cols)
        finally:
            self._loading = False
        self.matrixChanged.emit()

    # -- 快捷生成 ---------------------------------------------------------- #
    def set_identity(self) -> None:
        n, m = self.table.rowCount(), self.table.columnCount()
        self._fill(lambda i, j: 1 if i == j else 0)

    def set_zeros(self) -> None:
        self._fill(lambda i, j: 0)

    def set_ones(self) -> None:
        self._fill(lambda i, j: 1)

    def set_random(self) -> None:
        rows, cols = self.table.rowCount(), self.table.columnCount()
        values = np.random.default_rng().normal(size=(rows, cols)).round(3)
        self._fill(lambda i, j: values[i, j])

    def transpose(self) -> None:
        matrix = self.matrix()
        if matrix is None:
            return
        self.set_matrix(np.asarray(matrix).T)

    def clear_values(self) -> None:
        self._loading = True
        try:
            for i in range(self.table.rowCount()):
                for j in range(self.table.columnCount()):
                    item = self.table.item(i, j)
                    if item is None:
                        self.table.setItem(i, j, QTableWidgetItem(""))
                    else:
                        item.setText("")
        finally:
            self._loading = False
        self.matrixChanged.emit()


def _format_entry(value: Any, precision: int = 6) -> str:
    from core.numberfmt import format_number

    if isinstance(value, (complex, np.complexfloating)):
        if abs(complex(value).imag) < 1e-12:
            return format_number(float(np.real(value)), precision)
        return format_number(complex(value), precision)
    return format_number(value, precision)


# --------------------------------------------------------------------------- #
# 放大编辑窗口
# --------------------------------------------------------------------------- #
class MatrixEditorDialog(QDialog):
    """矩阵放大编辑窗口：大表格 + 自动扩展 + 空白填充策略 + **LaTeX 实时预览**。

    设计目标（来自使用反馈）：

    1. 可以把窗口最大化/全屏，舒舒服服地填一个很大的矩阵；
    2. 表格随填写**自动变大**，不用事先想好行列数；
    3. 结束时自动识别「有效区域」，右下角的空白整块裁掉；
    4. 中间没填的格子由用户选择填充方式（0 / 1 / NaN / 向左复制 / 行内插值 / 报错）。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        editor: MatrixEditor,
        label: str = "A",
        initial_size: int = 15,
    ):
        super().__init__(parent)
        self._editor = editor
        self._label = label
        self._loading = False
        #: 放大编辑时至少给这么大的网格，方便直接往里填（而不是只有 3×3）
        self._initial_size = max(int(initial_size), 1)

        self.setWindowTitle(f"编辑矩阵 {label} —— 放大 / 全屏编辑")
        self.resize(1180, 780)
        self.setSizeGripEnabled(True)

        self.grid = _PasteTableWidget(editor.table.rowCount(), editor.table.columnCount())
        self.grid.setAlternatingRowColors(True)
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.grid.verticalHeader().setDefaultSectionSize(26)
        self.grid.setMinimumSize(360, 260)
        self.grid.itemChanged.connect(self._on_item_changed)
        # 与主界面一致：只有方向键走到边界才扩展
        self.grid.edgeHandler = self._on_edge_key

        # ---- 工具条 ------------------------------------------------------ #
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, editor.max_size)
        self.rows_spin.setValue(self.grid.rowCount())
        self.rows_spin.setPrefix("行 ")
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, editor.max_size)
        self.cols_spin.setValue(self.grid.columnCount())
        self.cols_spin.setPrefix("列 ")
        size_button = QPushButton("应用尺寸")
        size_button.clicked.connect(lambda: self._resize(self.rows_spin.value(), self.cols_spin.value()))

        quick_buttons = [
            ("单位阵", self._set_identity),
            ("零阵", lambda: self._fill(lambda i, j: 0)),
            ("全 1", lambda: self._fill(lambda i, j: 1)),
            ("随机", self._set_random),
            ("转置", self._transpose),
            ("清空", self._clear),
            ("识别有效区域", self._trim),
        ]
        quick_row = QHBoxLayout()
        quick_row.setSpacing(4)
        for text, slot in quick_buttons:
            button = QPushButton(text)
            button.clicked.connect(slot)
            quick_row.addWidget(button)
        quick_row.addStretch(1)

        paste_button = QPushButton("从剪贴板覆盖")
        paste_button.clicked.connect(self._paste_from_clipboard)
        copy_button = QPushButton("复制表格")
        copy_button.clicked.connect(self._copy_to_clipboard)
        io_row = QHBoxLayout()
        io_row.addWidget(paste_button)
        io_row.addWidget(copy_button)
        io_row.addStretch(1)

        top = QHBoxLayout()
        top.addWidget(self.rows_spin)
        top.addWidget(self.cols_spin)
        top.addWidget(size_button)
        top.addSpacing(10)
        top.addWidget(paste_button)
        top.addWidget(copy_button)
        top.addStretch(1)

        # ---- 右侧：策略 + 预览 ------------------------------------------- #
        self.grow_check = QCheckBox("自动扩展（在最后一行/列填写时新增行列）")
        self.grow_check.setChecked(editor.auto_grow)
        self.grow_check.toggled.connect(lambda value: self._refresh_preview())

        self.fill_combo = QComboBox()
        for key, text in FILL_POLICIES.items():
            self.fill_combo.addItem(text, key)
        index = self.fill_combo.findData(editor.fill_policy)
        self.fill_combo.setCurrentIndex(max(index, 0))
        self.fill_combo.currentIndexChanged.connect(lambda *_: self._refresh_preview())

        self.trim_check = QCheckBox("自动识别有效区域（裁掉右下空白）")
        self.trim_check.setChecked(editor.trim)
        self.trim_check.toggled.connect(lambda value: self._refresh_preview())

        self.strict_hint = QLabel("")
        self.strict_hint.setWordWrap(True)

        policy_box = QGroupBox("读取方式")
        policy_layout = QVBoxLayout(policy_box)
        policy_layout.addWidget(self.grow_check)
        policy_form = QFormLayout()
        policy_form.addRow("空白单元格", self.fill_combo)
        policy_layout.addLayout(policy_form)
        policy_layout.addWidget(self.trim_check)
        policy_layout.addWidget(self.strict_hint)

        self.preview_view = LatexView(self)
        self.preview_view.set_background(editor.latex_background())
        self.preview_view.setMinimumHeight(220)

        preview_box = QGroupBox("识别结果预览（LaTeX 渲染）")
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.addWidget(self.preview_view, 1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setProperty("muted", "true")

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(policy_box)
        side_layout.addWidget(preview_box, 1)
        side_layout.addWidget(self.summary)
        side.setMinimumWidth(320)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(top)
        left_layout.addLayout(quick_row)
        left_layout.addWidget(self.grid, 1)
        splitter.addWidget(left)
        splitter.addWidget(side)
        splitter.setChildrenCollapsible(False)
        for index in range(splitter.count()):
            splitter.setCollapsible(index, False)
        left.setMinimumWidth(360)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([780, 380])
        self.splitter = splitter

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定并写回")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._buttons = buttons

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        hint = QLabel(
            "提示：双击单元格编辑；可直接从 Excel 按 Ctrl+V 粘贴；"
            "填到最后一行/列时会自动增加行列（可一直往下填）；"
            "按 Esc 取消，Ctrl+Enter 直接确定。"
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", "true")
        layout.addWidget(hint)
        layout.addWidget(buttons)

        # 所有子控件就绪后再灌入数据（_copy_cells_from 会更新行列 spinbox）
        self._copy_cells_from(editor.cells())
        self._ensure_minimum_size()
        self._refresh_preview()

    # ------------------------------------------------------------------ #
    # 基础操作
    # ------------------------------------------------------------------ #
    def _ensure_minimum_size(self) -> None:
        """把网格补到至少 ``initial_size`` 见方，让用户一进来就有足够的填写空间。"""
        rows = max(self.grid.rowCount(), self._initial_size)
        cols = max(self.grid.columnCount(), self._initial_size)
        limit = max(self._editor.max_size, self._initial_size)
        rows, cols = min(rows, limit), min(cols, limit)
        if rows == self.grid.rowCount() and cols == self.grid.columnCount():
            return
        self._loading = True
        try:
            self.grid.setRowCount(rows)
            self.grid.setColumnCount(cols)
            self._refresh_headers()
            self.rows_spin.setValue(rows)
            self.cols_spin.setValue(cols)
        finally:
            self._loading = False

    def _on_edge_key(self, row: int, col: int, key: int) -> bool:
        """方向键走到边界时扩展（与主界面同一套逻辑）。"""
        if not self.grow_check.isChecked() or row < 0 or col < 0:
            return False
        rows, cols = self.grid.rowCount(), self.grid.columnCount()
        at_bottom = key == Qt.Key_Down and row >= rows - 1
        at_right = key == Qt.Key_Right and col >= cols - 1
        if not (at_bottom or at_right):
            return False
        limit = max(self._editor.max_size, 1)
        grew = False
        self._loading = True
        try:
            if at_bottom and rows < limit:
                self.grid.setRowCount(rows + 1)
                grew = True
            if at_right and cols < limit:
                self.grid.setColumnCount(cols + 1)
                grew = True
        finally:
            self._loading = False
        if not grew:
            return False
        self._refresh_headers()
        self.rows_spin.setValue(self.grid.rowCount())
        self.cols_spin.setValue(self.grid.columnCount())
        self._loading = True
        try:
            if at_bottom:
                self.grid.setCurrentCell(min(row + 1, self.grid.rowCount() - 1), col)
            else:
                self.grid.setCurrentCell(row, min(col + 1, self.grid.columnCount() - 1))
        finally:
            self._loading = False
        self._refresh_preview()
        return True

    def _copy_cells_from(self, cells: Sequence[Sequence[str]]) -> None:
        self._loading = True
        try:
            rows = max(len(cells), 1)
            cols = max((len(row) for row in cells), default=1)
            self.grid.setRowCount(rows)
            self.grid.setColumnCount(cols)
            self.grid.setHorizontalHeaderLabels([str(j + 1) for j in range(cols)])
            self.grid.setVerticalHeaderLabels([str(i + 1) for i in range(rows)])
            for i in range(rows):
                for j in range(cols):
                    text = str(cells[i][j]) if i < len(cells) and j < len(cells[i]) else ""
                    self.grid.setItem(i, j, QTableWidgetItem(text))
            self.rows_spin.setValue(rows)
            self.cols_spin.setValue(cols)
        finally:
            self._loading = False

    def cells(self) -> list[list[str]]:
        grid: list[list[str]] = []
        for i in range(self.grid.rowCount()):
            row: list[str] = []
            for j in range(self.grid.columnCount()):
                item = self.grid.item(i, j)
                row.append(item.text().strip() if item is not None else "")
            grid.append(row)
        return grid

    def fill_policy(self) -> str:
        return self.fill_combo.currentData() or "zero"

    def trim(self) -> bool:
        return self.trim_check.isChecked()

    def auto_grow(self) -> bool:
        return self.grow_check.isChecked()

    def _refresh_headers(self) -> None:
        self.grid.setHorizontalHeaderLabels([str(j + 1) for j in range(self.grid.columnCount())])
        self.grid.setVerticalHeaderLabels([str(i + 1) for i in range(self.grid.rowCount())])

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """内容变化只更新预览；扩展交给 :meth:`_on_edge_key`（方向键触发）。"""
        if self._loading:
            return
        self._refresh_preview()

    def _resize(self, rows: int, cols: int) -> None:
        self._loading = True
        try:
            self.grid.setRowCount(max(1, rows))
            self.grid.setColumnCount(max(1, cols))
            self._refresh_headers()
        finally:
            self._loading = False
        self._refresh_preview()

    def _fill(self, func) -> None:
        self._loading = True
        try:
            for i in range(self.grid.rowCount()):
                for j in range(self.grid.columnCount()):
                    item = self.grid.item(i, j)
                    if item is None:
                        item = QTableWidgetItem()
                        self.grid.setItem(i, j, item)
                    item.setText(_display_value(func(i, j)))
        finally:
            self._loading = False
        self._refresh_preview()

    def _set_identity(self) -> None:
        self._fill(lambda i, j: 1 if i == j else 0)

    def _set_random(self) -> None:
        rows, cols = self.grid.rowCount(), self.grid.columnCount()
        values = np.random.default_rng().normal(size=(rows, cols)).round(3)
        self._fill(lambda i, j: values[i, j])

    def _clear(self) -> None:
        self._loading = True
        try:
            for i in range(self.grid.rowCount()):
                for j in range(self.grid.columnCount()):
                    item = self.grid.item(i, j)
                    if item is None:
                        self.grid.setItem(i, j, QTableWidgetItem(""))
                    else:
                        item.setText("")
        finally:
            self._loading = False
        self._refresh_preview()

    def _transpose(self) -> None:
        cells = self.cells()
        transposed = [[cells[i][j] for i in range(len(cells))] for j in range(len(cells[0]))] if cells else []
        self._copy_cells_from(transposed)
        self._refresh_preview()

    def _trim(self) -> None:
        rows = cols = 0
        cells = self.cells()
        for i, row in enumerate(cells):
            for j, cell in enumerate(row):
                if cell:
                    rows, cols = max(rows, i + 1), max(cols, j + 1)
        if rows and cols:
            self._resize(rows, cols)

    def _paste_from_clipboard(self) -> None:
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            QMessageBox.information(self, "剪贴板为空", "请先复制一块数据。")
            return
        self._copy_cells_from(clipboard_to_grid(text))
        self._refresh_preview()

    def _copy_to_clipboard(self) -> None:
        lines = ["\t".join(row) for row in self.cells()]
        QGuiApplication.clipboard().setText("\n".join(lines))

    # ------------------------------------------------------------------ #
    # 预览
    # ------------------------------------------------------------------ #
    def _refresh_preview(self) -> None:
        from core.spec import ResultItem, ResultSheet

        fill = self.fill_policy()
        trim = self.trim()
        try:
            matrix, info = grid_to_matrix(self.cells(), fill=fill, trim=trim)
            latex = matrix_to_latex(matrix, precision=6, name=self._label)
            note = describe_grid(info)
            sheet = ResultSheet(items=[ResultItem(title="", latex=latex, note=note, kind="matrix")])
            self.preview_view.set_sheet(sheet, result_style_for(self._editor.latex_background()))
            self.summary.setText(note)
            self.summary.setStyleSheet("")
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(True)
            self.strict_hint.setText("")
        except ValueError as exc:
            self.preview_view.set_message(f"暂时无法识别：{exc}", muted=False)
            self.summary.setText(str(exc))
            self.summary.setStyleSheet("color: #cf222e;")
            self.strict_hint.setText(str(exc))
            # strict 模式下的空单元格是「可修正的问题」，其余是致命的
            self._buttons.button(QDialogButtonBox.Ok).setEnabled(fill != "strict" or "空" not in str(exc))

    # ------------------------------------------------------------------ #
    # 键盘
    # ------------------------------------------------------------------ #
    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and event.modifiers() & Qt.ControlModifier:
            self.accept()
            return
        super().keyPressEvent(event)
