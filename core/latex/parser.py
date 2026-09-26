"""把 LaTeX 子集解析为 :mod:`core.latex.nodes` 中的 AST。

实现范围
--------
* **结构命令**：``\\frac`` / ``\\dfrac`` / ``\\tfrac``、``\\sqrt[n]{}``、
  ``\\left<delim> ... \\right<delim>``、上标 ``^``、下标 ``_``、``{}`` 分组
* **环境**：``matrix`` / ``pmatrix`` / ``bmatrix`` / ``Bmatrix`` / ``vmatrix`` /
  ``Vmatrix`` / ``smallmatrix`` / ``cases`` / ``array{ccc}`` / ``aligned`` /
  ``align`` / ``split`` / ``gathered``
* **其余命令**（``\\det``、``\\lambda``、``\\sum``、``\\int``、``\\text``、
  ``\\operatorname`` …）原样保留在 :class:`~core.latex.nodes.Run` 中，交给
  matplotlib 的 ``mathtext`` 渲染。

之所以要自己解析环境，是因为 ``mathtext`` 不支持 ``\\begin{...}``（实测会
抛 ``Unknown symbol: \\begin``）。
"""

from __future__ import annotations

from .nodes import (
    DELIM_CHARS,
    Fence,
    Frac,
    Group,
    Matrix,
    Node,
    Plain,
    Run,
    Script,
    Space,
    Sqrt,
    as_group,
)

__all__ = [
    "LatexSyntaxError",
    "parse_math",
    "parse_latex",
    "atomize",
    "contains_env",
    "strip_math_delimiters",
]


class LatexSyntaxError(ValueError):
    """括号/环境不配对等语法错误。"""


_WHITESPACE = " \t\r\n"
_LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
#: 结构命令中「参数长度固定」的集合
_FRAC_COMMANDS = {"frac", "dfrac", "tfrac", "cfrac"}
#: 带一个「文本参数」的命令：内容含非 ASCII 时转为正体文本节点
_TEXT_COMMANDS = {
    "text", "textrm", "textnormal", "textup", "cn",
    "mathrm", "mathbf", "mathit", "mathsf", "mathtt",
    "textbf", "textit", "operatorname", "bm",
}
_SKIP_ROW_SPACING = "[]"


def _needs_plain(text: str) -> bool:
    """``\\text{...}`` 的内容是否需要走正体文本路径。

    两种情况必须转 :class:`Plain`：

    * 含非 ASCII 字符（中文等）—— mathtext 没有对应字形；
    * 含 ``_ ^ \\ & $ % # { } ~`` 等 LaTeX 特殊字符 —— 交给 mathtext 会被
      当成上下标/命令而错排，例如列名 ``temp_1``。
    """
    text = text or ""
    if any(ord(ch) > 0x00FF for ch in text):
        return True
    return any(ch in "_^\\&$%#{}~" for ch in text)



# --------------------------------------------------------------------------- #
# 基础扫描工具
# --------------------------------------------------------------------------- #
def _read_command_name(src: str, i: int) -> tuple[str, int]:
    """``src[i] == '\\\\'``，返回 ``(命令名, 下一个位置)``。

    字母命令返回 ``"frac"`` 这样的名字；符号命令（``\\\\``、``\\{``、``\\,``）
    返回其后的单个字符。
    """
    j = i + 1
    if j >= len(src):
        return "", j
    if src[j] in _LETTERS:
        k = j
        while k < len(src) and src[k] in _LETTERS:
            k += 1
        return src[j:k], k
    return src[j], j + 1


def _find_matching_brace(src: str, i: int) -> int:
    """``src[i] == '{'``，返回配对 ``'}'`` 的下标；不配对返回 ``-1``。"""
    depth = 0
    j = i
    n = len(src)
    while j < n:
        ch = src[j]
        if ch == "\\":
            _name, j = _read_command_name(src, j)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def _read_group(src: str, i: int) -> tuple[str, int]:
    """读取 ``{...}`` 分组内容。"""
    if i >= len(src) or src[i] != "{":
        raise LatexSyntaxError(f"期望 '{{'，实际得到 {src[i:i + 1]!r}")
    end = _find_matching_brace(src, i)
    if end < 0:
        raise LatexSyntaxError("花括号不配对")
    return src[i + 1 : end], end + 1


def _read_arguments(src: str, i: int, count: int) -> tuple[list[str], int] | None:
    """连续读取 ``count`` 个 ``{...}`` 参数；失败返回 ``None``。"""
    args: list[str] = []
    j = i
    for _ in range(count):
        while j < len(src) and src[j] in _WHITESPACE:
            j += 1
        if j >= len(src) or src[j] != "{":
            return None
        try:
            content, j = _read_group(src, j)
        except LatexSyntaxError:
            return None
        args.append(content)
    return args, j


def _read_optional_bracket(src: str, i: int) -> tuple[str, int]:
    """读取 ``[...]`` 可选参数，返回内容与结束位置。"""
    j = i
    while j < len(src) and src[j] in _WHITESPACE:
        j += 1
    if j < len(src) and src[j] == "[":
        depth = 0
        k = j
        while k < len(src):
            if src[k] == "[":
                depth += 1
            elif src[k] == "]":
                depth -= 1
                if depth == 0:
                    return src[j + 1 : k], k + 1
            k += 1
    return "", i


def _read_delimiter(src: str, i: int) -> tuple[str, int]:
    """读取一个定界符记号，返回 ``(记号, 下一个位置)``。"""
    j = i
    while j < len(src) and src[j] in _WHITESPACE:
        j += 1
    if j >= len(src):
        return ".", j
    if src[j] == "\\":
        name, k = _read_command_name(src, j)
        if name and not name[0].isalpha():
            token = "\\" + name
            if token in DELIM_CHARS:
                return token, k
        token = "\\" + name
        return (token if token in DELIM_CHARS else "."), k
    ch = src[j]
    if ch in DELIM_CHARS:
        return ch, j + 1
    return ".", j


def contains_env(src: str) -> bool:
    """判断字符串中是否出现 ``\\begin{...}``（需要走自研排版路径）。"""
    return "\\begin{" in (src or "")


def strip_math_delimiters(text: str) -> str:
    """去掉整串外层的 ``$`` / ``$$``。"""
    text = (text or "").strip()
    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        return text[2:-2]
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        return text[1:-1]
    return text


# --------------------------------------------------------------------------- #
# 环境与 \left..\right 扫描
# --------------------------------------------------------------------------- #
def _find_env_end(src: str, i: int, env: str) -> tuple[str, int] | None:
    """从 ``i`` 开始找配对的 ``\\end{env}``，返回 ``(环境体, 结束后的位置)``。"""
    stack = 1
    n = len(src)
    j = i
    while j < n:
        if src[j] == "\\":
            name, k = _read_command_name(src, j)
            if name == "begin":
                try:
                    _inner, k2 = _read_group(src, k)
                except LatexSyntaxError:
                    j = k
                    continue
                stack += 1
                j = k2
                continue
            if name == "end":
                try:
                    _inner, k2 = _read_group(src, k)
                except LatexSyntaxError:
                    j = k
                    continue
                stack -= 1
                if stack == 0:
                    return src[i:k], k2
                j = k2
                continue
            j = k
            continue
        j += 1
    return None


def _find_right(src: str, i: int) -> tuple[str, str, int] | None:
    """从 ``i`` 开始找配对的 ``\\right``，返回 ``(内容, 右定界符, 结束位置)``。"""
    left_depth = 0
    n = len(src)
    j = i
    while j < n:
        if src[j] == "\\":
            name, k = _read_command_name(src, j)
            if name == "left":
                left_depth += 1
                j = k
                continue
            if name == "right":
                if left_depth == 0:
                    delim, after = _read_delimiter(src, k)
                    return src[i:j], delim, after
                left_depth -= 1
                j = k
                continue
            j = k
            continue
        j += 1
    return None


def _split_env_body(body: str) -> list[list[str]]:
    """按 ``\\\\`` 拆行、``&`` 拆列；嵌套环境与花括号内的分隔符不受影响。"""
    rows: list[list[str]] = [[]]
    buf: list[str] = []
    depth = 0
    env_depth = 0
    i = 0
    n = len(body)

    while i < n:
        ch = body[i]
        if ch == "\\":
            name, j = _read_command_name(body, i)
            if name == "begin":
                env_depth += 1
                buf.append(body[i:j])
                i = j
                continue
            if name == "end":
                env_depth = max(0, env_depth - 1)
                buf.append(body[i:j])
                i = j
                continue
            if name == "\\" and depth == 0 and env_depth == 0:
                rows[-1].append("".join(buf))
                buf = []
                rows.append([])
                i = j
                # 跳过 \\ 后的可选间距 [2pt]
                _opt, i = _read_optional_bracket(body, i)
                continue
            if name in ("hline", "hdashline") and depth == 0 and env_depth == 0:
                i = j
                continue
            if name in ("cr",) and depth == 0 and env_depth == 0:
                rows[-1].append("".join(buf))
                buf = []
                rows.append([])
                i = j
                continue
            buf.append(body[i:j])
            i = j
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "&" and depth == 0 and env_depth == 0:
            rows[-1].append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1

    rows[-1].append("".join(buf))
    cleaned = [row for row in rows if any(cell.strip() for cell in row)]
    return cleaned or [[".", ""]]


def _build_matrix(env: str, body: str, col_spec: str = "") -> Matrix:
    """把环境体解析成 :class:`Matrix` 节点。"""
    raw_rows = _split_env_body(body)
    rows: list[list[Group]] = []
    for raw_row in raw_rows:
        rows.append([as_group(parse_math(cell)) for cell in raw_row])
    return Matrix(env=env, rows=rows, col_spec=col_spec)


# --------------------------------------------------------------------------- #
# 上标 / 下标
# --------------------------------------------------------------------------- #
def _read_script_arg(src: str, i: int) -> tuple[str | None, int]:
    """读取 ``^`` / ``_`` 之后的一个参数。"""
    j = i
    n = len(src)
    while j < n and src[j] in _WHITESPACE:
        j += 1
    if j >= n:
        return None, i
    if src[j] == "{":
        try:
            content, k = _read_group(src, j)
        except LatexSyntaxError:
            return None, i
        return content, k
    if src[j] == "\\":
        _name, k = _read_command_name(src, j)
        return src[j:k], k
    return src[j], j + 1


def _attach_scripts(node: Node, src: str, i: int) -> tuple[Node, int]:
    """把紧跟其后的 ``^`` / ``_`` 绑定到 ``node`` 上。"""
    sup: str | None = None
    sub: str | None = None
    n = len(src)
    while True:
        j = i
        while j < n and src[j] in _WHITESPACE:
            j += 1
        if j >= n or src[j] not in "^_":
            break
        arg, k = _read_script_arg(src, j + 1)
        if arg is None:
            break
        if src[j] == "^":
            sup = arg
        else:
            sub = arg
        i = k
    if sup is None and sub is None:
        return node, i
    return (
        Script(
            base=node,
            sup=as_group(parse_math(sup)) if sup is not None else None,
            sub=as_group(parse_math(sub)) if sub is not None else None,
        ),
        i,
    )


# --------------------------------------------------------------------------- #
# 主解析器
# --------------------------------------------------------------------------- #
def parse_math(src: str) -> list[Node]:
    """解析一段数学体（不含外层 ``$``），返回水平排列的节点列表。"""
    nodes: list[Node] = []
    buf: list[str] = []
    src = strip_math_delimiters(src) if src.strip().startswith("$") else src
    n = len(src)

    def flush() -> None:
        """把缓冲区作为一个 mathtext 原子输出。

        mathtext 会忽略 ``$...$`` 内部的首尾空白，因此在 Run 与矩阵/根式等
        结构化节点交界处，空白必须显式转成 :class:`Space`，否则会出现
        ``\\det(A) =`` 紧贴矩阵定界符的问题。
        """
        text = "".join(buf)
        buf.clear()
        if not text:
            return
        core = text.strip()
        if not core:
            nodes.append(Space(em=0.25))
            return
        if text[0].isspace():
            nodes.append(Space(em=0.25))
        nodes.append(Run(core))
        if text[-1].isspace():
            nodes.append(Space(em=0.25))

    i = 0
    while i < n:
        ch = src[i]

        if ch == "\\":
            name, j = _read_command_name(src, i)

            if name == "begin":
                try:
                    env, k = _read_group(src, j)
                except LatexSyntaxError:
                    buf.append(src[i:j])
                    i = j
                    continue
                env = env.strip()
                col_spec = ""
                if env.startswith("array") or env.startswith("tabular"):
                    spec, k2 = _read_optional_bracket(src, k)
                    if spec:
                        col_spec = spec
                    if k2 < len(src) and src[k2] == "{":
                        try:
                            col_spec, k = _read_group(src, k2)
                        except LatexSyntaxError:
                            pass
                found = _find_env_end(src, k, env)
                if found is None:
                    buf.append(src[i:j])
                    i = j
                    continue
                body, after = found
                flush()
                node: Node = _build_matrix(env, body, col_spec)
                node, i = _attach_scripts(node, src, after)
                nodes.append(node)
                continue

            if name in _FRAC_COMMANDS:
                args = _read_arguments(src, j, 2)
                if args is not None:
                    (num_src, den_src), jj = args
                    if contains_env(num_src) or contains_env(den_src):
                        flush()
                        node = Frac(
                            num=as_group(parse_math(num_src)),
                            den=as_group(parse_math(den_src)),
                        )
                        node, i = _attach_scripts(node, src, jj)
                        nodes.append(node)
                        continue
                    buf.append(src[i:jj])
                    i = jj
                    continue
                buf.append(src[i:j])
                i = j
                continue

            if name == "sqrt":
                index_src, k = _read_optional_bracket(src, j)
                args = _read_arguments(src, k, 1)
                if args is not None:
                    (body_src,), jj = args
                    if contains_env(body_src) or (index_src and contains_env(index_src)):
                        flush()
                        node = Sqrt(
                            body=as_group(parse_math(body_src)),
                            index=as_group(parse_math(index_src)) if index_src else None,
                        )
                        node, i = _attach_scripts(node, src, jj)
                        nodes.append(node)
                        continue
                    buf.append(src[i:jj])
                    i = jj
                    continue
                buf.append(src[i:j])
                i = j
                continue

            if name in _TEXT_COMMANDS:
                # \text{中文} 之类：mathtext 无法渲染中文字形，转成 Plain 节点
                # 交由中文字体绘制；含 LaTeX 特殊字符的内容同理。
                args = _read_arguments(src, j, 1)
                if args is not None:
                    (content,), jj = args
                    if _needs_plain(content):
                        flush()
                        nodes.append(
                            Plain(
                                text=content,
                                bold=name in ("mathbf", "textbf", "bm"),
                                italic=name in ("mathit", "textit"),
                            )
                        )
                        i = jj
                        continue
                    buf.append(src[i:jj])
                    i = jj
                    continue
                buf.append(src[i:j])
                i = j
                continue

            if name == "left":
                delim, k = _read_delimiter(src, j)
                found = _find_right(src, k)
                if found is None:
                    buf.append(src[i:j])
                    i = j
                    continue
                content, right_delim, after = found
                if contains_env(content):
                    flush()
                    node = Fence(
                        left=delim,
                        body=as_group(parse_math(content)),
                        right=right_delim,
                    )
                    node, i = _attach_scripts(node, src, after)
                    nodes.append(node)
                    continue
                buf.append(src[i:after])
                i = after
                continue

            buf.append(src[i:j])
            i = j
            continue

        if ch == "{":
            end = _find_matching_brace(src, i)
            if end < 0:
                buf.append(ch)
                i += 1
                continue
            content = src[i + 1 : end]
            if contains_env(content):
                flush()
                node = as_group(parse_math(content))
                node, i = _attach_scripts(node, src, end + 1)
                nodes.append(node)
                continue
            buf.append(src[i : end + 1])
            i = end + 1
            continue

        buf.append(ch)
        i += 1

    flush()
    return nodes


def parse_latex(src: str) -> Group:
    """解析为单个 :class:`Group`（便于直接排版）。"""
    return as_group(parse_math(src))


# --------------------------------------------------------------------------- #
# 降级：把解析失败的 Run 拆成更小的原子
# --------------------------------------------------------------------------- #
def atomize(src: str) -> list[Node]:
    """把一段 mathtext 无法整体解析的公式拆成尽量小的原子。

    仅在 :class:`~core.latex.nodes.Run` 度量失败时作为降级路径调用，
    因此允许丢失部分间距与分组语义，但必须保证「不会死循环」。
    """
    nodes: list[Node] = []
    buf: list[str] = []
    i = 0
    n = len(src)

    def flush() -> None:
        text = "".join(buf)
        buf.clear()
        if text:
            nodes.append(Run(text) if text.strip() else Space(em=0.28))

    while i < n:
        ch = src[i]

        if ch == "\\":
            name, j = _read_command_name(src, i)
            if name in ("begin", "end"):
                # 环境无法拆分：整体作为一个 Run
                flush()
                nodes.append(Run(src[i:j]))
                i = j
                continue
            if name in _FRAC_COMMANDS:
                args = _read_arguments(src, j, 2)
                if args is not None:
                    (num_src, den_src), jj = args
                    flush()
                    nodes.append(
                        Frac(
                            num=as_group(atomize(num_src)),
                            den=as_group(atomize(den_src)),
                        )
                    )
                    i = jj
                    continue
            if name == "sqrt":
                index_src, k = _read_optional_bracket(src, j)
                args = _read_arguments(src, k, 1)
                if args is not None:
                    (body_src,), jj = args
                    flush()
                    nodes.append(
                        Sqrt(
                            body=as_group(atomize(body_src)),
                            index=as_group(atomize(index_src)) if index_src else None,
                        )
                    )
                    i = jj
                    continue
            flush()
            nodes.append(Run(src[i:j]))
            i = j
            continue

        if ch in "^_":
            arg, j = _read_script_arg(src, i + 1)
            if arg is None:
                flush()
                buf.append(ch)
                i += 1
                continue
            flush()
            group = as_group(atomize(arg))
            if nodes and not isinstance(nodes[-1], Space):
                last = nodes.pop()
                if isinstance(last, Script):
                    if ch == "^":
                        last.sup = group
                    else:
                        last.sub = group
                    nodes.append(last)
                else:
                    nodes.append(
                        Script(
                            base=last,
                            sup=group if ch == "^" else None,
                            sub=group if ch == "_" else None,
                        )
                    )
            else:
                nodes.append(Run(src[i : i + 2]))
            i = j
            continue

        if ch == "{":
            end = _find_matching_brace(src, i)
            if end < 0:
                flush()
                i += 1
                continue
            inner = atomize(src[i + 1 : end])
            flush()
            nodes.extend(inner)
            i = end + 1
            continue

        if ch == "}":
            i += 1
            continue

        if ch in _WHITESPACE:
            if buf:
                flush()
            if nodes and not isinstance(nodes[-1], Space):
                nodes.append(Space(em=0.28))
            i += 1
            continue

        buf.append(ch)
        i += 1

    flush()
    return nodes


def latex_to_unicode(src: str) -> str:
    """把 LaTeX 源码粗转为 Unicode 文本（最终兜底显示）。"""
    from .nodes import UNICODE_FALLBACK
    import re

    out = src or ""
    for key in sorted(UNICODE_FALLBACK, key=len, reverse=True):
        out = out.replace(key, UNICODE_FALLBACK[key])
    out = re.sub(r"\\(?:text|operatorname|mathrm|mathbf|mathit)\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\[a-zA-Z]+", "", out)
    out = out.replace("\\\\", " ")
    out = out.replace("{", "").replace("}", "").replace("^", "").replace("_", "")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def literal_node(src: str) -> Plain:
    """构造降级用的正体文本节点。"""
    return Plain(text=latex_to_unicode(src))
