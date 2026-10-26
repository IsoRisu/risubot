# latexrenderer.py
"""Render LaTeX expressions found in a text string to PNG images."""

import ast
import io
import re

import matplotlib
import matplotlib.backends.backend_agg

matplotlib.use("Agg")  # headless; required for systemd
from matplotlib.figure import Figure  # no pyplot: it keeps global state and isn't thread-safe

# Matches, in priority order:
#   ``` ... ``` / ` ... `   (code; passed through untouched so `$HOME` etc. isn't mangled)
#   [[plot: sin(x); x^2 from -5 to 5]]   (function graph, see render_plot_spec)
#   [[dfa: start: q0; q0 -a-> q1]]       (graphs, automata, Turing machines, trees: see graphrenderer.py)
#   $$ ... $$   (display, multiline ok)
#   \[ ... \]   (display)
#   \( ... \)   (inline)
#   $ ... $     (inline; "costs $5 and $10" is not math: the opening $ must be followed
#                by a non-space and the closing $ must not follow a space or precede a digit)
_LATEX_RE = re.compile(
    r"(?P<code>```.*?```|`[^`\n]+`)"
    r"|\[\[plot:\s*(?P<plot>.+?)\s*\]\]"
    r"|\[\[(?P<gkind>graph|digraph|tree|automaton|dfa|nfa|fsm|tm|turing|pda):\s*(?P<gspec>.+?)\s*\]\]"
    r"|\\begin\{(?P<env>equation|align|gather|multline|eqnarray)\*?\}(?P<env_body>.+?)\\end\{(?P=env)\*?\}"
    r"|(\$\$(?P<display_dollar>.+?)\$\$)"
    r"|(\\\[(?P<display_bracket>.+?)\\\])"
    r"|(\\\((?P<inline_paren>.+?)\\\))"
    r"|(?<![\w$])\$(?P<inline_dollar>[^\s$][^$]*?)(?<![\s\\])\$(?![\d$])",
    re.DOTALL,
)

# matplotlib's mathtext is a LaTeX subset. Map common commands it lacks onto
# ones it has, so more expressions render instead of falling back to source.
_SUBSTITUTIONS = [
    (re.compile(r"\\[dt]frac\b"), r"\\frac"),
    # mathtext has a real \text{} that keeps spaces and ä/ö; \mathrm would eat the spaces.
    (re.compile(r"\\(?:mbox|textrm|textnormal|textbf|textit|textsf|texttt)\s*\{"), r"\\text{"),
    (re.compile(r"\\operatorname\*?\s*\{"), r"\\mathrm{"),
    (re.compile(r"\\boldsymbol\s*\{"), r"\\mathbf{"),
    (re.compile(r"\\(?:displaystyle|textstyle|scriptstyle|limits|nolimits)\b"), ""),
    (re.compile(r"\\(?:left|right)\s*\."), ""),
    (re.compile(r"\\(?:big|Big|bigg|Bigg)[lrm]?(?![A-Za-z])"), ""),     # size modifiers
    # short spellings mathtext doesn't know (\leq etc. it does)
    (re.compile(r"\\le(?![A-Za-z])"), r"\\leq"),
    (re.compile(r"\\ge(?![A-Za-z])"), r"\\geq"),
    (re.compile(r"\\ne(?![A-Za-z])"), r"\\neq"),
    (re.compile(r"\\iff(?![A-Za-z])"), r"\\Leftrightarrow"),
    (re.compile(r"\\implies(?![A-Za-z])"), r"\\Rightarrow"),
    (re.compile(r"\\impliedby(?![A-Za-z])"), r"\\Leftarrow"),
    (re.compile(r"\\[lr]vert(?![A-Za-z])"), "|"),
    (re.compile(r"\\(?:[lr]Vert|Vert)(?![A-Za-z])"), r"\\|"),
    (re.compile(r"\\[;:!]"), r"\\,"),
    (re.compile(r"\\quad\b|\\qquad\b"), r"\\ \\ "),
]


_WRAPPER_RE = re.compile(r"\\(?:begin|end)\{(?:aligned|align|gather|split|equation|multline)\*?\}")
_JUNK_RE = re.compile(r"\\(?:label\{[^}]*\}|tag\*?\{[^}]*\}|nonumber|notag)|&")
_CASES_RE = re.compile(r"\\begin\{cases\}(.*?)\\end\{cases\}", re.DOTALL)
_ROWSEP_RE = re.compile(r"\\\\(?:\[[^\]]*\])?")


def _normalize(expr: str) -> str:
    for pattern, repl in _SUBSTITUTIONS:
        expr = pattern.sub(repl, expr)
    # mathtext can't handle raw newlines
    return " ".join(expr.split())


def _split_lines(expr: str) -> list:
    """mathtext has no align/aligned; strip the wrappers and alignment marks
    and render each row (split on \\\\) as its own line."""
    m = _CASES_RE.search(expr)
    if m:
        # mathtext has no \\begin{cases}: draw "prefix", then one row per case,
        # each led by a brace, with the "&" between value and condition as a comma.
        rows = [r for r in _ROWSEP_RE.split(m.group(1)) if r.strip()]
        raw = [expr[:m.start()], *(r"\{\ " + r.replace("&", r",\ \ ") for r in rows), expr[m.end():]]
    else:
        expr = _WRAPPER_RE.sub("", expr)
        raw = _ROWSEP_RE.split(expr)
    lines = [_normalize(_JUNK_RE.sub(" ", row)) for row in raw]
    return [ln for ln in lines if ln] or [""]


def _render_mathtext(expr: str, display: bool) -> bytes:
    """Render a LaTeX expression to PNG bytes using matplotlib's mathtext."""
    fontsize = 22 if display else 18
    lines = _split_lines(expr)
    n = len(lines)

    fig = Figure(figsize=(8, 0.8 * n))
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


# ---------------------------------------------------------------------------
# Plain-text shortcut for simple inline math
#
# Discord can't put an image in the middle of a sentence, so every inline
# formula rendered as a PNG splits the message into text / image / text.
# Simple inline math (x^2, a_n, \varepsilon > 0, n \to \infty, \mathbb{R}^n)
# is converted to Unicode text instead; anything it can't fully convert
# (fractions, roots, sums, limits, ...) returns None and gets rendered as an
# image as before.
# ---------------------------------------------------------------------------

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "varrho": "ϱ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_SYMBOLS = {
    "infty": "∞", "cdot": "·", "times": "×", "pm": "±", "mp": "∓", "div": "÷",
    "ldots": "…", "dots": "…", "cdots": "⋯", "partial": "∂", "nabla": "∇",
    "forall": "∀", "exists": "∃", "emptyset": "∅", "varnothing": "∅",
    "cup": "∪", "cap": "∩", "setminus": "∖", "neg": "¬", "wedge": "∧", "vee": "∨",
    "circ": "∘", "angle": "∠", "ell": "ℓ", "hbar": "ℏ",
}
# Relations/arrows get a space on each side so "x\in A" doesn't become "x∈A".
_RELATIONS = {
    "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "Leftrightarrow": "⇔", "leftrightarrow": "↔", "mapsto": "↦",
    "leq": "≤", "geq": "≥", "neq": "≠", "approx": "≈", "equiv": "≡", "sim": "∼",
    "propto": "∝", "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂", "subseteq": "⊆",
    "supset": "⊃", "supseteq": "⊇", "ll": "≪", "gg": "≫",
}
_FUNCS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh",
    "cosh", "tanh", "ln", "log", "exp", "lim", "sup", "inf", "max", "min", "det",
    "dim", "ker", "deg", "gcd", "arg",
}
_BLACKBOARD = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ"}

# Only characters with widely-supported glyphs; anything else falls back to an image.
_SUP = {ord(k): v for k, v in zip("0123456789+-=()nixy", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱˣʸ")}
_SUB = {ord(k): v for k, v in zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")}

_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")
_SUP_RE = re.compile(r"\^(?:\{([^{}]*)\}|([0-9A-Za-z+\-]))")
_SUB_RE = re.compile(r"_(?:\{([^{}]*)\}|([0-9A-Za-z+\-]))")


def _script(table):
    def repl(m):
        body = m.group(1) if m.group(1) is not None else m.group(2)
        body = body.replace(" ", "")
        if body and all(ord(c) in table for c in body):
            return body.translate(table)
        return m.group(0)  # leave it; the leftover check will reject the expression
    return repl


def _to_unicode(expr: str):
    """Return plain Unicode text for a simple inline expression, else None."""
    s = _normalize(expr)
    s = re.sub(r"\\mathbb\s*\{([A-Za-z])\}", lambda m: _BLACKBOARD.get(m.group(1), m.group(0)), s)
    s = re.sub(r"\\(?:mathrm|text)\s*\{([^{}]*)\}", r"\1", s)

    def command(m, _s=s):
        name = m.group(1)
        if name in _GREEK:
            return _GREEK[name]
        if name in _SYMBOLS:
            return _SYMBOLS[name]
        if name in _RELATIONS:
            return f" {_RELATIONS[name]} "
        if name in _FUNCS:
            nxt = _s[m.end():m.end() + 1]
            return name + (" " if nxt and (nxt.isalnum() or nxt == "\\") else "")
        return m.group(0)

    s = _COMMAND_RE.sub(command, s)
    s = s.replace("\\,", "\u2009").replace("\\ ", " ")
    s = _SUP_RE.sub(_script(_SUP), s)
    s = _SUB_RE.sub(_script(_SUB), s)
    s = re.sub(r" {2,}", " ", s).strip()

    if (not s or re.search(r"[\\{}^_*~`]", s) or "||" in s
            or re.match(r"(?:[>#]|[-+] |\d+[.)] )", s)):
        return None  # unconverted LaTeX left over, or would trigger Discord markdown
    return s


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

        if m.group("plot") is not None:
            try:
                items.append(("image", render_plot_spec(m.group("plot"))))
            except Exception:
                items.append(("text", m.group(0)))  # keep the source visible
            pos = m.end()
            continue

        if m.group("gspec") is not None:
            try:
                import graphrenderer  # lazy: keeps the math path independent of it
                items.append(("image", graphrenderer.render_graph_spec(m.group("gspec"), m.group("gkind"))))
            except Exception:
                items.append(("text", m.group(0)))  # keep the source visible
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
        if not display:
            plain = _to_unicode(expr)
            if plain is not None:
                items.append(("text", plain))
                pos = m.end()
                continue
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

# ---------------------------------------------------------------------------
# Extended matplotlib renderers  (matplotlib ^3.8)
#
# All functions return PNG bytes. No pyplot, no global figure state.
# No aliased imports anywhere. Optional deps (mpl_toolkits, Pillow) raise
# clear errors when missing.
# ---------------------------------------------------------------------------

# --- figure helper ---------------------------------------------------------

def _new_fig(figsize=(8, 5), **savefig_kwargs):
    """Create a Figure with an Agg canvas, return (fig, save) where
    save() -> PNG bytes."""
    fig = matplotlib.figure.Figure(figsize=figsize)
    matplotlib.backends.backend_agg.FigureCanvasAgg(fig)  # explicit canvas
    buf = io.BytesIO()

    savefig_kwargs.setdefault("format", "png")
    savefig_kwargs.setdefault("dpi", 200)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("pad_inches", 0.15)
    savefig_kwargs.setdefault("facecolor", "white")
    savefig_kwargs.setdefault("transparent", False)

    def save():
        fig.savefig(buf, **savefig_kwargs)
        return buf.getvalue()

    return fig, save


# ---------------------------------------------------------------------------
# Core plotting
# ---------------------------------------------------------------------------

# Whitelisted function names -> numpy callables (resolved lazily).
_PLOT_FUNCS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "arcsin", "acos": "arccos", "atan": "arctan",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "exp": "exp", "sqrt": "sqrt", "cbrt": "cbrt", "abs": "abs", "sign": "sign",
    "floor": "floor", "ceil": "ceil",
    "ln": "log", "log": "log", "log10": "log10", "log2": "log2",  # log(x) is the natural log
}
_PLOT_RANGE_RE = re.compile(
    r"\s+(?:from|väliltä|välillä)\s+(-?\d+(?:\.\d+)?)\s+(?:to|-|\.\.)\s+(-?\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)


_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def _plot_preprocess(expr: str) -> str:
    """Turn calculator-style input into valid Python: x^2, x², 2x, xy, 3(x+1), (x+1)(x-1), π."""
    s = expr.strip()
    s = re.sub(r"^(?:y|f\s*\(\s*x\s*\))\s*=\s*", "", s)
    s = re.sub(r"⁻?[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "**(" + m.group(0).translate(_SUPERSCRIPTS) + ")", s)
    for a, b in (("^", "**"), ("·", "*"), ("×", "*"), ("−", "-"), ("π", "pi")):
        s = s.replace(a, b)
    s = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)(?![eE][+-]?\d)\s*(?=[A-Za-z(])", r"\1*", s)  # 2x, 3(x+1)
    s = re.sub(r"\)\s*(?=[\w(])", ")*", s)                                              # (a)(b), (a)x
    s = re.sub(r"(?<![\w.])([xy])(?=[xy](?![\w.]))", r"\1*", s)                         # xy -> x*y
    s = re.sub(r"\b([xy])\s*(?=\()", r"\1*", s)                                         # x(x+1)
    return s


def _safe_eval(expr: str, x, y=None):
    """Evaluate a function of x (and y, for curves) without Python's eval: only numbers,
    x, y, pi, e, + - * / % **, unary +/-, and the one-argument functions in _PLOT_FUNCS."""
    import numpy
    s = _plot_preprocess(expr)
    if not s or len(s) > 200:
        raise ValueError("expression is empty or too long")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        raise ValueError(f"can't parse {expr.strip()!r}") from None
    if sum(1 for _ in ast.walk(tree)) > 120:
        raise ValueError("expression is too complex")

    binops = {ast.Add: numpy.add, ast.Sub: numpy.subtract, ast.Mult: numpy.multiply,
              ast.Div: numpy.divide, ast.Pow: numpy.power, ast.Mod: numpy.mod}
    consts = {"pi": numpy.pi, "e": numpy.e}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return numpy.float64(n.value)  # float64, so 10**10**10 can't hang Python's bigints
        if isinstance(n, ast.Name):
            if n.id == "x":
                return x
            if n.id == "y":
                if y is None:
                    raise ValueError("y only works in an equation or inequality, e.g. x^2 + y^2 = 1 or y > x^2")
                return y
            if n.id in consts:
                return numpy.float64(consts[n.id])
            raise ValueError(f"unknown name {n.id!r} (the variables are x and y)")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in binops:
            return binops[type(n.op)](ev(n.left), ev(n.right))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in _PLOT_FUNCS and len(n.args) == 1 and not n.keywords):
            return getattr(numpy, _PLOT_FUNCS[n.func.id])(ev(n.args[0]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            if n.func.id in _PLOT_FUNCS:
                raise ValueError(f"{n.func.id}() takes exactly one argument")
            raise ValueError(f"unknown function {n.func.id!r}")
        raise ValueError("unsupported syntax (use + - * / ^ and functions like sin(x))")

    return ev(tree)


_COMPARISON_RE = re.compile(r"<=|>=|<|>")


def _plot_kind(expr: str):
    """Classify one input -> (kind, payload, op).
      explicit  'sin(x)', 'y = f(x)'                  payload = f(x) text
      implicit  'x^2+y^2=1', 'x*y=1', 'x=2'           payload = 'lhs - rhs' (curve where it is 0)
      region    'x^2+y^2<1', 'y>=x^2', 'x+y<=2'       payload = 'lhs - rhs', op = '<', '<=', '>', '>='
    """
    e = expr.strip().replace("≤", "<=").replace("≥", ">=")
    ops = _COMPARISON_RE.findall(e)
    if ops:
        if len(ops) != 1 or "=" in _COMPARISON_RE.sub("", e):
            raise ValueError("use exactly one comparison (<, >, <= or >=)")
        lhs, rhs = (p.strip() for p in _COMPARISON_RE.split(e))
        if not lhs or not rhs:
            raise ValueError(f"incomplete inequality {e!r}")
        return "region", f"({lhs})-({rhs})", ops[0]

    if "=" not in e:
        return "explicit", e, None
    if e.count("=") != 1:
        raise ValueError("use exactly one '=' in an equation")
    lhs, rhs = (p.strip() for p in e.split("="))
    if not lhs or not rhs:
        raise ValueError(f"incomplete equation {e!r}")
    if (lhs == "y" or re.fullmatch(r"f\s*\(\s*x\s*\)", lhs)) \
            and not re.search(r"(?<![A-Za-z_])y(?![A-Za-z_])", _plot_preprocess(rhs)):
        return "explicit", rhs, None
    return "implicit", f"({lhs})-({rhs})", None


def _zero_set_mask(Z):
    """Grid cells where Z crosses (or hits) zero, i.e. where the curve passes."""
    import numpy
    m = (Z == 0)
    m[:, :-1] |= Z[:, :-1] * Z[:, 1:] < 0
    m[:-1, :] |= Z[:-1, :] * Z[1:, :] < 0
    return m


def _auto_window(payloads):
    """Pick a square window that frames the implicit curves. A curve that runs off
    the default ±10 view (lines, hyperbolas) just gets ±10."""
    import numpy
    n = 601
    for R in (10.0, 100.0):
        g = numpy.linspace(-R, R, n)
        X, Y = numpy.meshgrid(g, g)
        xs, ys = [], []
        with numpy.errstate(all="ignore"):
            for p in payloads:
                Z = numpy.broadcast_to(numpy.asarray(_safe_eval(p, X, Y), dtype=float), X.shape)
                Z = numpy.where(numpy.isfinite(Z), Z, numpy.nan)
                m = _zero_set_mask(Z)
                xs.append(X[m])
                ys.append(Y[m])
        px, py = numpy.concatenate(xs), numpy.concatenate(ys)
        if px.size:
            lo_x, hi_x, lo_y, hi_y = px.min(), px.max(), py.min(), py.max()
            if R == 10.0 and max(abs(lo_x), abs(hi_x), abs(lo_y), abs(hi_y)) > 0.97 * R:
                return (-R, R, -R, R)
            half = max(max(hi_x - lo_x, hi_y - lo_y) / 2 * 1.25, 0.5)
            cx, cy = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
            return (float(cx - half), float(cx + half), float(cy - half), float(cy + half))
    return (-10.0, 10.0, -10.0, 10.0)


def render_function(expression, xmin=None, xmax=None, n=800,
                    title=None, figsize=(8, 5), **savefig_kwargs):
    """Plot graphs. `expression` is one string or a list of up to 6, in calculator syntax:
      - functions of x:   'sin(x) + x^2/10', '2x+1', 'y = ln(x)'
      - implicit curves:  'x^2 + y^2 = 1', 'x*y = 1', '(x-1)^2/4 + y^2 = 1'
      - regions:          'x^2 + y^2 < 1', 'y >= x^2', 'x + y <= 2'  (shaded; the boundary
                          is dashed for < and >, solid for <= and >=)
    Input is parsed by a restricted evaluator, never passed to eval(). xmin/xmax (used
    for both axes when there is a curve or region) default to -10..10, or to a window
    fitted around the boundary. Asymptotes (tan, 1/x) are broken up and the y-range
    is clipped so one huge value doesn't flatten the rest."""
    import numpy
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    exprs = [expression] if isinstance(expression, str) else list(expression)
    if not 1 <= len(exprs) <= 6:
        raise ValueError("give between 1 and 6 functions")
    parsed = [(*_plot_kind(e), e.strip().replace("$", "")) for e in exprs]  # kind, payload, op, label
    planar = [p for k, p, _, _ in parsed if k != "explicit"]  # everything drawn in the x-y plane

    if xmin is None or xmax is None:
        if planar:
            x0, x1, y0, y1 = _auto_window(planar)
        else:
            x0, x1, y0, y1 = -10.0, 10.0, None, None
    else:
        x0, x1 = float(xmin), float(xmax)
        if not x0 < x1 or max(abs(x0), abs(x1)) > 1e6:
            raise ValueError("x range must satisfy min < max, within ±1,000,000")
        y0, y1 = (x0, x1) if planar else (None, None)

    x = numpy.linspace(x0, x1, int(n))
    explicit_ys = {}
    fields = {}
    with numpy.errstate(all="ignore"):
        for idx, (kind, payload, op, label) in enumerate(parsed):
            if kind == "explicit":
                y = numpy.broadcast_to(numpy.asarray(_safe_eval(payload, x), dtype=float), x.shape).copy()
                y[~numpy.isfinite(y) | (numpy.abs(y) > 1e9)] = numpy.nan
                explicit_ys[idx] = y

        ylim = None
        finite_parts = [y[~numpy.isnan(y)] for y in explicit_ys.values()]
        finite = numpy.concatenate(finite_parts) if finite_parts else numpy.array([])
        if finite.size:
            lo, hi = numpy.percentile(finite, [2, 98])
            span = max(float(hi - lo), 1e-9)
            for y in explicit_ys.values():  # cut the line where it jumps across an asymptote
                dy = numpy.abs(numpy.diff(y))
                cross = numpy.sign(y[:-1]) != numpy.sign(y[1:])
                y[1:][cross & (dy > 1.5 * span)] = numpy.nan
            if float(finite.max() - finite.min()) > 3 * span:
                ylim = (float(lo) - 0.25 * span, float(hi) + 0.25 * span)

        if planar:
            g = numpy.linspace(0, 1, 700)
            X, Y = numpy.meshgrid(x0 + (x1 - x0) * g, y0 + (y1 - y0) * g)
            for idx, (kind, payload, op, label) in enumerate(parsed):
                if kind == "explicit":
                    continue
                Z = numpy.broadcast_to(numpy.asarray(_safe_eval(payload, X, Y), dtype=float), X.shape).copy()
                Z[~numpy.isfinite(Z)] = numpy.nan
                if kind == "implicit":
                    ok = _zero_set_mask(Z).any()
                else:  # region: some point must satisfy the inequality
                    ok = {"<": Z < 0, "<=": Z <= 0, ">": Z > 0, ">=": Z >= 0}[op].any()
                if not ok:
                    raise ValueError(f"{label!r} has no points in this range")
                fields[idx] = (numpy.ma.masked_invalid(Z), bool(_zero_set_mask(Z).any()))

    fig, save = _new_fig(figsize, **savefig_kwargs)
    ax = fig.add_subplot(111)
    handles = []
    for idx, (kind, _, op, label) in enumerate(parsed):
        color = f"C{idx % 10}"
        if kind == "explicit":
            ax.plot(x, explicit_ys[idx], color=color)
            handles.append(Line2D([0], [0], color=color, linewidth=2, label=label))
        elif kind == "implicit":
            ax.contour(X, Y, fields[idx][0], levels=[0], colors=[color], linewidths=2)
            handles.append(Line2D([0], [0], color=color, linewidth=2, label=label))
        else:
            Z, has_boundary = fields[idx]
            if op in ("<", "<="):
                levels = [float(Z.min()) - 1.0, 0.0]
            else:
                levels = [0.0, float(Z.max()) + 1.0]
            ax.contourf(X, Y, Z, levels=levels, colors=[color], alpha=0.3)
            if has_boundary:
                ax.contour(X, Y, Z, levels=[0], colors=[color], linewidths=2,
                           linestyles="dashed" if op in ("<", ">") else "solid")
            handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.4, label=label))
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlim(x0, x1)
    if planar:
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")  # circles must look like circles
    elif ylim:
        ax.set_ylim(*ylim)
    if len(parsed) > 1:
        ax.legend(handles=handles)
    elif not title:
        k, _, _, label = parsed[0]
        title = label if k != "explicit" or "=" in label else "y = " + label
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return save()


def parse_plot_spec(spec: str):
    """'sin(x); x^2+y^2<1 from -5 to 5' -> (['sin(x)', 'x^2+y^2<1'], -5.0, 5.0).
    xmin/xmax are None when no range is given (render_function then picks one)."""
    spec = spec.strip()
    xmin = xmax = None
    m = _PLOT_RANGE_RE.search(spec)
    if m:
        xmin, xmax = float(m.group(1)), float(m.group(2))
        spec = spec[:m.start()]
    exprs = [p.strip() for p in spec.split(";") if p.strip()]
    if not exprs:
        raise ValueError("no function given")
    return exprs, xmin, xmax


def render_plot_spec(spec: str) -> bytes:
    """Entry point for the bot's !plot command and the [[plot: ...]] tag."""
    exprs, xmin, xmax = parse_plot_spec(spec)
    return render_function(exprs, xmin, xmax)
