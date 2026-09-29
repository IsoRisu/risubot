# latexrenderer.py
"""Render LaTeX expressions found in a text string to PNG images."""

import io
import re

import matplotlib

matplotlib.use("Agg")  # headless; required for systemd
from matplotlib.figure import Figure  # no pyplot: it keeps global state and isn't thread-safe

# Matches, in priority order:
#   ``` ... ``` / ` ... `   (code; passed through untouched so `$HOME` etc. isn't mangled)
#   $$ ... $$   (display, multiline ok)
#   \[ ... \]   (display)
#   \( ... \)   (inline)
#   $ ... $     (inline; not adjacent to digits/word chars, to avoid "costs $5 and $10")
_LATEX_RE = re.compile(
    r"(?P<code>```.*?```|`[^`\n]+`)"
    r"|\\begin\{(?P<env>equation|align|gather|multline|eqnarray)\*?\}(?P<env_body>.+?)\\end\{(?P=env)\*?\}"
    r"|(\$\$(?P<display_dollar>.+?)\$\$)"
    r"|(\\\[(?P<display_bracket>.+?)\\\])"
    r"|(\\\((?P<inline_paren>.+?)\\\))"
    r"|(?<![\w$])\$(?P<inline_dollar>[^\s$][^$]*?)(?<![\s\\])\$(?![\w$])",
    re.DOTALL,
)

# matplotlib's mathtext is a LaTeX subset. Map common commands it lacks onto
# ones it has, so more expressions render instead of falling back to source.
_SUBSTITUTIONS = [
    (re.compile(r"\\[dt]frac\b"), r"\\frac"),
    (re.compile(r"\\(?:text|textbf|textit|operatorname|mbox)\s*\{"), r"\\mathrm{"),
    (re.compile(r"\\boldsymbol\s*\{"), r"\\mathbf{"),
    (re.compile(r"\\(?:displaystyle|textstyle|scriptstyle|limits|nolimits)\b"), ""),
    (re.compile(r"\\(?:left|right)\s*\."), ""),
    (re.compile(r"\\[;:!]"), r"\\,"),
    (re.compile(r"\\quad\b|\\qquad\b"), r"\\ \\ "),
]


_WRAPPER_RE = re.compile(r"\\(?:begin|end)\{(?:aligned|align|gather|split|equation|multline)\*?\}")
_JUNK_RE = re.compile(r"\\(?:label\{[^}]*\}|nonumber|notag)|&")
_ROWSEP_RE = re.compile(r"\\\\(?:\[[^\]]*\])?")


def _normalize(expr: str) -> str:
    for pattern, repl in _SUBSTITUTIONS:
        expr = pattern.sub(repl, expr)
    # mathtext can't handle raw newlines
    return " ".join(expr.split())


def _split_lines(expr: str) -> list:
    """mathtext has no align/aligned; strip the wrappers and alignment marks
    and render each row (split on \\\\) as its own line."""
    expr = _WRAPPER_RE.sub("", expr)
    expr = _JUNK_RE.sub(" ", expr)
    lines = [_normalize(row) for row in _ROWSEP_RE.split(expr)]
    return [ln for ln in lines if ln] or [""]


def _render_mathtext(expr: str, display: bool) -> bytes:
    """Render a LaTeX expression to PNG bytes using matplotlib's mathtext."""
    fontsize = 22 if display else 18
    lines = _split_lines(expr)
    n = len(lines)

    fig = Figure(figsize=(8, 0.9 * n))
    for i, line in enumerate(lines):
        fig.text(0, 1 - (i + 0.5) / n, f"${line}$", fontsize=fontsize, color="black", va="center")
    buf = io.BytesIO()
    # Opaque white, not transparent: black-on-transparent is invisible in
    # Discord's dark theme.
    fig.savefig(
        buf,
        format="png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
        transparent=False,
    )
    return buf.getvalue()


def split_and_render(text: str):
    """
    Walk `text` and return a list of ("text", str) and ("image", bytes)
    items, in original order.

    Any LaTeX block that fails to render (bad syntax, unsupported command)
    is emitted back as plain text so nothing is silently dropped.
    """
    items = []
    pos = 0

    for m in _LATEX_RE.finditer(text):
        if m.start() > pos:
            items.append(("text", text[pos:m.start()]))

        if m.group("code") is not None:
            items.append(("text", m.group(0)))
            pos = m.end()
            continue

        if m.group("env_body") is not None:
            expr, display = m.group("env_body"), True
        elif m.group("display_dollar") is not None:
            expr, display = m.group("display_dollar"), True
        elif m.group("display_bracket") is not None:
            expr, display = m.group("display_bracket"), True
        elif m.group("inline_paren") is not None:
            expr, display = m.group("inline_paren"), False
        else:
            expr, display = m.group("inline_dollar"), False

        expr = expr.strip()
        try:
            items.append(("image", _render_mathtext(expr, display)))
        except Exception:
            # Rendering failed; keep the original source visible.
            items.append(("text", m.group(0)))

        pos = m.end()

    if pos < len(text):
        items.append(("text", text[pos:]))

    return items


def contains_latex(text: str) -> bool:
    """True if `text` has math outside of code spans/fences."""
    return any(m.group("code") is None for m in _LATEX_RE.finditer(text))
