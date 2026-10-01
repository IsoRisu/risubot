# graphrenderer.py
"""Render graphs, finite automata, Turing machines, pushdown automata and trees
to PNG bytes, from a small text description.

Pure matplotlib + numpy (both already dependencies of latexrenderer.py), so
there is nothing extra to install and nothing that needs a system binary.

Why not neo4j-viz (python-graph-visualization)? It produces interactive
HTML/JavaScript widgets for notebooks and browsers. A Discord bot can only post
static images, and it has no notion of start arrows, accepting double circles,
self-loops with transition labels, or a Turing machine tape. This module draws
those directly.

Description language (one statement per line or separated by ';'):

    A -> B                 directed edge          A -- B     undirected edge
    A -> B : 5             edge label (weight)    A <-> B    arrows both ways
    q0 -a-> q1             edge label between the dashes (automata)
    q1 -0/1,R-> q2         Turing machine rule: read/write,move
    q0 -a,b-> q0           several symbols on one edge (a self-loop is fine)
    start: q0              start state (arrow drawn into it)
    accept: q2, q3         accepting states (double circle)
    nodes: a, b, c         isolated nodes (nodes in edges are created for you)
    highlight: q1          fill these nodes orange (current state, a found path)
    layout: lr             circle | spring | lr | tb | tree  (sensible default per kind)
    title: My automaton
    tape: 1 0 1 1 _        Turing machine tape snapshot below the diagram
    head: 2                0-based index of the cell under the head
    state: q1              label shown under the head

Node names are words (letters, digits, ' and _). q_0 and q_accept are drawn as
subscripts. Put other names in double quotes: "start here" -> "end".

Entry points:
    render_graph_spec(spec, kind="graph") -> PNG bytes
    parse_graph_spec(spec, kind="graph")  -> GraphSpec
"""

import io
import math
import re
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.backends.backend_agg
import numpy
from matplotlib.figure import Figure  # no pyplot: global state, not thread-safe
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path

MAX_NODES = 30
MAX_EDGES = 100
MAX_SPEC_CHARS = 4000
MAX_LABEL_CHARS = 40

# Spec "kind" (the tag or command name) -> (family, default layout)
_KINDS = {
    "graph": ("graph", None), "digraph": ("graph", None), "verkko": ("graph", None),
    "tree": ("tree", "tree"), "puu": ("tree", "tree"),
    "automaton": ("automaton", "lr"), "dfa": ("automaton", "lr"), "nfa": ("automaton", "lr"),
    "fsm": ("automaton", "lr"), "tm": ("automaton", "lr"), "turing": ("automaton", "lr"),
    "pda": ("automaton", "lr"), "automata": ("automaton", "lr"),
}
_LAYOUT_ALIASES = {
    "circle": "circle", "circular": "circle", "ring": "circle",
    "spring": "spring", "force": "spring",
    "lr": "lr", "ltr": "lr", "layered": "lr", "horizontal": "lr",
    "tb": "tb", "td": "tb", "vertical": "tb",
    "tree": "tree",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_NAME = r"""(?:"[^"]+"|[\w'’′]+)"""
# NAME  [<]-label-[>]  NAME  [: label]      e.g.  q0 -a,b-> q1,  A -- B,  A <-> B : 3
_EDGE_RE = re.compile(
    rf"""^(?P<a>{_NAME})\s*(?P<l><)?-(?P<lab>[^"]*?)-*(?P<r>>)?\s*(?P<b>{_NAME})\s*(?::\s*(?P<w>.*))?$"""
)
_KEYWORD_RE = re.compile(
    r"^(start|initial|accept|accepting|final|finals|nodes|layout|title|highlight|tape|head|state)\s*[:=]\s*(.*)$",
    re.IGNORECASE,
)
_NAME_RE = re.compile(_NAME)

_LABEL_COMMANDS = {
    "varepsilon": "ε", "epsilon": "ε", "lambda": "λ", "delta": "δ", "Sigma": "Σ", "sigma": "σ",
    "Gamma": "Γ", "gamma": "γ", "to": "→", "rightarrow": "→", "leftarrow": "←", "alpha": "α",
    "beta": "β", "pi": "π", "mu": "μ", "cdot": "·", "times": "×", "emptyset": "∅",
}


@dataclass
class GraphSpec:
    family: str
    nodes: list = field(default_factory=list)
    groups: list = field(default_factory=list)  # [a, b, label, head_a, head_b], merged below
    start: str = None
    accept: list = field(default_factory=list)
    highlight: list = field(default_factory=list)
    layout: str = None
    title: str = None
    tape: list = field(default_factory=list)
    head: int = None
    state: str = None


def _unquote(name: str) -> str:
    return name[1:-1] if name.startswith('"') else name


def _clean_label(text: str) -> str:
    s = (text or "").strip().replace("$", "")
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _LABEL_COMMANDS.get(m.group(1), m.group(1)), s)
    s = re.sub(r"(?<![\w])(?:eps|epsilon)(?![\w])", "ε", s)
    s = re.sub(r"(?<![\w])lambda(?![\w])", "λ", s)
    s = s.replace("\\", "")
    return s[:MAX_LABEL_CHARS]


def _names(text: str) -> list:
    return [_unquote(m.group(0)) for m in _NAME_RE.finditer(text)]


def parse_graph_spec(spec: str, kind: str = "graph") -> GraphSpec:
    kind = (kind or "graph").strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"unknown diagram type {kind!r}")
    family, default_layout = _KINDS[kind]
    spec = re.sub(r"^```\w*|```$", "", spec.strip()).strip()
    if not spec:
        raise ValueError("empty description")
    if len(spec) > MAX_SPEC_CHARS:
        raise ValueError("description too long")

    g = GraphSpec(family=family, layout=default_layout)
    seen = set()

    def add_node(name):
        if name not in seen:
            seen.add(name)
            g.nodes.append(name)

    for raw in re.split(r"[;\n]+", spec):
        st = raw.strip()
        if not st:
            continue

        kw = _KEYWORD_RE.match(st)
        if kw:
            key, val = kw.group(1).lower(), kw.group(2).strip()
            if key in ("start", "initial"):
                names = _names(val)
                if not names:
                    raise ValueError("start: needs a state name")
                g.start = names[0]
                add_node(g.start)
            elif key in ("accept", "accepting", "final", "finals"):
                for n in _names(val):
                    g.accept.append(n)
                    add_node(n)
            elif key == "nodes":
                for n in _names(val):
                    add_node(n)
            elif key == "highlight":
                for n in _names(val):
                    g.highlight.append(n)
                    add_node(n)
            elif key == "layout":
                lay = _LAYOUT_ALIASES.get(val.lower())
                if not lay:
                    raise ValueError(f"unknown layout {val!r} (use circle, spring, lr, tb or tree)")
                g.layout = lay
            elif key == "title":
                g.title = _clean_label(val)
            elif key == "tape":
                cells = re.split(r"[\s,|]+", val.strip())
                cells = [c for c in cells if c]
                if len(cells) == 1 and len(cells[0]) > 1:
                    cells = list(cells[0])  # "1011" -> 1 0 1 1
                g.tape = [_clean_label(c)[:4] for c in cells][:25]
            elif key == "head":
                try:
                    g.head = int(val)
                except ValueError:
                    raise ValueError("head: needs a cell number, e.g. head: 2") from None
            elif key == "state":
                g.state = _clean_label(val)
            continue

        m = _EDGE_RE.match(st)
        if m:
            a, b = _unquote(m.group("a")), _unquote(m.group("b"))
            label = _clean_label(" ".join(x for x in (m.group("lab"), m.group("w")) if x and x.strip()))
            head_a, head_b = bool(m.group("l")), bool(m.group("r"))
            if head_a and not head_b:  # "A <- B" means B -> A
                a, b, head_a, head_b = b, a, False, True
            add_node(a)
            add_node(b)
            g.groups.append([a, b, label, head_a, head_b])
            continue

        if re.fullmatch(rf"{_NAME}(?:\s*,\s*{_NAME})*", st):
            for n in _names(st):
                add_node(n)
            continue

        raise ValueError(f"can't parse {st!r}")

    if not g.nodes:
        raise ValueError("no nodes")
    if len(g.nodes) > MAX_NODES:
        raise ValueError(f"too many nodes (max {MAX_NODES})")
    if len(g.groups) > MAX_EDGES:
        raise ValueError(f"too many edges (max {MAX_EDGES})")
    for n in g.accept + g.highlight:
        if n not in seen:
            raise ValueError(f"unknown node {n!r}")
    return g


def _merge_edges(g: GraphSpec) -> list:
    """Merge parallel edges (same ends, same arrow style) into one with a joint label.
    Returns [(a, b, label, head_a, head_b)]."""
    idx = {n: i for i, n in enumerate(g.nodes)}
    merged = {}
    for a, b, label, ha, hb in g.groups:
        if ha == hb and idx[a] > idx[b]:  # undirected or two-way: canonical orientation
            a, b = b, a
        merged.setdefault((a, b, ha, hb), [])
        if label and label not in merged[(a, b, ha, hb)]:
            merged[(a, b, ha, hb)].append(label)
    out = []
    for (a, b, ha, hb), labels in merged.items():
        short = all(len(x) <= 3 and "/" not in x for x in labels)
        out.append((a, b, (", " if short else "\n").join(labels), ha, hb))
    return out


# ---------------------------------------------------------------------------
# Layouts. Each returns {name: numpy.array([x, y])} with ~2 units between neighbours.
# ---------------------------------------------------------------------------

def _forward_adjacency(g, edges):
    adj = {n: [] for n in g.nodes}
    for a, b, _, ha, hb in edges:
        if a == b:
            continue
        if hb or not ha:
            adj[a].append(b)
        if ha or not hb:
            adj[b].append(a)
    return adj


def _layout_circle(g, edges, r):
    n = len(g.nodes)
    if n == 1:
        return {g.nodes[0]: numpy.zeros(2)}
    R = max(1.8, 2.0 / (2 * math.sin(math.pi / n)))
    return {
        name: numpy.array([R * math.cos(math.pi / 2 - 2 * math.pi * i / n),
                           R * math.sin(math.pi / 2 - 2 * math.pi * i / n)])
        for i, name in enumerate(g.nodes)
    }


def _layout_spring(g, edges, r):
    names = g.nodes
    n = len(names)
    if n == 1:
        return {names[0]: numpy.zeros(2)}
    idx = {nm: i for i, nm in enumerate(names)}
    A = numpy.zeros((n, n))
    for a, b, *_ in edges:
        if a != b:
            A[idx[a], idx[b]] = A[idx[b], idx[a]] = 1.0
    rng = numpy.random.default_rng(7)  # fixed seed: same input, same picture
    ang = 2 * numpy.pi * numpy.arange(n) / n
    pos = numpy.c_[numpy.cos(ang), numpy.sin(ang)] * math.sqrt(n) + rng.normal(0, 0.1, (n, 2))
    k, iters = 1.7, 300
    for it in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = numpy.linalg.norm(delta, axis=2)
        numpy.fill_diagonal(dist, 1.0)
        dist = numpy.maximum(dist, 0.01)
        disp = ((k * k / dist ** 2)[:, :, None] * delta).sum(axis=1)       # repulsion
        disp -= ((dist / k)[:, :, None] * delta * A[:, :, None]).sum(axis=1)  # attraction
        disp -= 0.05 * pos                                                  # keep components together
        length = numpy.maximum(numpy.linalg.norm(disp, axis=1), 1e-9)
        temp = k * (1 - it / iters)
        pos += disp / length[:, None] * numpy.minimum(length, temp)[:, None]
    d = numpy.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    numpy.fill_diagonal(d, numpy.inf)
    pos *= max(1.0, 2.0 / max(d.min(), 1e-6))
    return {nm: pos[i] for i, nm in enumerate(names)}


def _bfs_levels(g, edges, roots):
    adj = _forward_adjacency(g, edges)
    level, order = {}, []
    queue = []
    for root in list(roots) + list(g.nodes):  # unreachable nodes start new BFS trees
        if root in level:
            continue
        level[root] = 0 if root in roots or not level else min(level.values())
        queue = [root]
        order.append(root)
        while queue:
            cur = queue.pop(0)
            for nxt in adj[cur]:
                if nxt not in level:
                    level[nxt] = level[cur] + 1
                    order.append(nxt)
                    queue.append(nxt)
    return level, order, adj


def _layout_layered(g, edges, r, horizontal=True):
    roots = [g.start] if g.start else _roots(g, edges)
    level, order, _ = _bfs_levels(g, edges, roots)
    layers = {}
    for nm in order:
        layers.setdefault(level[nm], []).append(nm)
    nbrs = {n: set() for n in g.nodes}
    for a, b, *_ in edges:
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    pos_in = {nm: i for L in layers.values() for i, nm in enumerate(L)}
    keys = sorted(layers)
    for sweep in range(4):  # barycentre ordering to cut crossings
        for L in (keys if sweep % 2 == 0 else keys[::-1]):
            prev = L - 1 if sweep % 2 == 0 else L + 1
            def bary(nm):
                ps = [pos_in[x] for x in nbrs[nm] if level[x] == prev]
                return sum(ps) / len(ps) if ps else pos_in[nm]
            layers[L].sort(key=bary)
            for i, nm in enumerate(layers[L]):
                pos_in[nm] = i
    big = max(r.values())
    step_level, step_in = 2.4 + 1.4 * big, 1.8 + 1.0 * big
    out = {}
    for L, members in layers.items():
        m = len(members)
        for i, nm in enumerate(members):
            across = ((m - 1) / 2 - i) * step_in
            along = L * step_level
            out[nm] = numpy.array([along, across]) if horizontal else numpy.array([-across, -along])
    return out


def _roots(g, edges):
    """Nodes nothing points at (for directed input); otherwise the first node."""
    has_in = set()
    arrows = False
    for a, b, _, ha, hb in edges:
        if hb:
            has_in.add(b)
            arrows = True
        if ha:
            has_in.add(a)
            arrows = True
    roots = [n for n in g.nodes if n not in has_in] if arrows else []
    return roots or [g.nodes[0]]


def _layout_tree(g, edges, r):
    roots = [g.start] if g.start else _roots(g, edges)
    adj = _forward_adjacency(g, edges)
    parent, children = {}, {n: [] for n in g.nodes}
    for root in list(roots) + list(g.nodes):
        if root in parent:
            continue
        parent[root] = None
        queue = [root]
        while queue:
            cur = queue.pop(0)
            for nxt in adj[cur]:
                if nxt not in parent:
                    parent[nxt] = cur
                    children[cur].append(nxt)
                    queue.append(nxt)
    big = max(r.values())
    step_x, step_y = 1.8 + 1.0 * big, 2.0 + 0.8 * big
    counter = [0.0]
    out = {}

    def place(node, depth):
        kids = children[node]
        if not kids:
            x = counter[0]
            counter[0] += 1.0
        else:
            xs = [place(k, depth + 1) for k in kids]
            x = (xs[0] + xs[-1]) / 2
        out[node] = numpy.array([x * step_x, -depth * step_y])
        return x

    for root in [n for n in g.nodes if parent[n] is None]:
        place(root, 0)
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

_FILL, _EDGE_COLOR, _INK, _HILITE = "#dbe9f6", "#2b5c8a", "#222222", "#ffd59e"


def _display_name(name: str):
    """(text, approx_length). q_0 / q_accept become real subscripts."""
    m = re.fullmatch(r"([A-Za-z]+)_\{?([A-Za-z0-9]+)\}?", name)
    if m:
        return (rf"$\mathrm{{{m.group(1)}}}_{{\mathrm{{{m.group(2)}}}}}$",
                len(m.group(1)) + 0.7 * len(m.group(2)))
    return name.replace("$", ""), len(name)


def _angle_gap(a, b):
    d = abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    return d


def _free_angle(used, preferred):
    best, best_score = preferred, -1e9
    for i in range(24):
        a = i * math.pi / 12
        gap = min((_angle_gap(a, u) for u in used), default=math.pi)
        score = min(gap, 1.2) - 0.1 * _angle_gap(a, preferred)
        if score > best_score:
            best, best_score = a, score
    return best


def _bezier_clear(p0, p1, rad, others):
    """Does the quadratic arc3 curve p0->p1 stay clear of the other nodes?"""
    d = p1 - p0
    ctrl = (p0 + p1) / 2 + rad * numpy.array([d[1], -d[0]])
    for t in numpy.linspace(0.08, 0.92, 24):
        pt = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * ctrl + t ** 2 * p1
        for c, rr in others:
            if numpy.linalg.norm(pt - c) < rr + 0.12:
                return False
    return True


def render_graph(g: GraphSpec) -> bytes:
    edges = _merge_edges(g)
    disp = {n: _display_name(n) for n in g.nodes}
    radius = {n: min(1.1, 0.38 + 0.065 * max(0.0, disp[n][1] - 2)) for n in g.nodes}

    layout = g.layout or ("circle" if len(g.nodes) <= 6 else "spring")
    if layout == "circle":
        pos = _layout_circle(g, edges, radius)
    elif layout == "spring":
        pos = _layout_spring(g, edges, radius)
    elif layout == "lr":
        pos = _layout_layered(g, edges, radius, horizontal=True)
    elif layout == "tb":
        pos = _layout_layered(g, edges, radius, horizontal=False)
    else:
        pos = _layout_tree(g, edges, radius)

    centre = numpy.mean([pos[n] for n in g.nodes], axis=0)
    fig = Figure()
    matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    pts = []        # everything that must fit in the picture
    texts = []      # label artists, resized once the final scale is known
    arrows = []     # arrow artists, ditto

    def include(p, pad=0.0):
        pts.append((p[0] - pad, p[1] - pad))
        pts.append((p[0] + pad, p[1] + pad))

    def add_label(p, text, **kw):
        t = ax.text(p[0], p[1], text, ha="center", va="center", color=_INK, zorder=6, **kw)
        texts.append(t)
        lines = text.count("\n") + 1
        include(p, 0.0)
        pts.append((p[0] - 0.09 * max(len(x) for x in text.split("\n")), p[1] - 0.2 * lines))
        pts.append((p[0] + 0.09 * max(len(x) for x in text.split("\n")), p[1] + 0.2 * lines))
        return t

    def edge_label(p, text):
        add_label(p, text, bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.92))

    # --- nodes -------------------------------------------------------------
    outer = {}
    for n in g.nodes:
        c, rr = pos[n], radius[n]
        fill = _HILITE if n in g.highlight else _FILL
        circ = Circle(c, rr, fc=fill, ec=_EDGE_COLOR, lw=2.0, zorder=3)
        ax.add_patch(circ)
        outer[n] = circ
        if n in g.accept:
            ax.add_patch(Circle(c, rr * 0.8, fc="none", ec=_EDGE_COLOR, lw=1.6, zorder=3))
        add_label(c, disp[n][0], fontweight="bold" if g.family == "tree" else "normal")
        include(c, rr + 0.05)

    # --- used angles around each node (for loops / start arrow) ---------------
    used = {n: [] for n in g.nodes}
    for a, b, *_ in edges:
        if a != b:
            d = pos[b] - pos[a]
            used[a].append(math.atan2(d[1], d[0]))
            used[b].append(math.atan2(-d[1], -d[0]))

    def away(n):
        d = pos[n] - centre
        return math.atan2(d[1], d[0]) if numpy.linalg.norm(d) > 1e-6 else math.pi

    # --- start arrow ---------------------------------------------------------
    if g.start:
        pref = {"lr": math.pi, "tb": math.pi / 2, "tree": math.pi / 2}.get(layout, away(g.start))
        th = _free_angle(used[g.start], pref)
        used[g.start].append(th)
        u = numpy.array([math.cos(th), math.sin(th)])
        c, rr = pos[g.start], radius[g.start]
        tail, tip = c + (rr + 1.0) * u, c + rr * u
        arr = FancyArrowPatch(tail, tip, arrowstyle="-|>,head_length=0.5,head_width=0.2",
                              mutation_scale=14, lw=1.8, color=_INK, shrinkA=0, shrinkB=0, zorder=2)
        ax.add_patch(arr)
        arrows.append(arr)
        include(tail, 0.1)

    # --- edges -----------------------------------------------------------------
    pair_groups = {}
    for e in edges:
        a, b, _, ha, hb = e
        if a != b:
            pair_groups.setdefault(frozenset((a, b)), []).append(e)

    def style(ha, hb):
        return {(False, False): "-", (False, True): "-|>", (True, True): "<|-|>"}[(ha, hb)]

    label_spots = []

    # curvature first, so self-loops can steer clear of the curved edges
    rads = {}
    for e in edges:
        a, b, label, ha, hb = e
        if a == b:
            continue
        siblings = pair_groups[frozenset((a, b))]
        p0, p1 = pos[a], pos[b]
        if len(siblings) == 1:
            rad = 0.0
            others = [(pos[n], radius[n] + (0.9 if any(x[0] == x[1] == n for x in edges) else 0.0))
                      for n in g.nodes if n not in (a, b)]
            if not _bezier_clear(p0, p1, 0.0, others):  # a node sits on the line: bend around it
                rad = next((c for c in (-0.3, 0.3, -0.55, 0.55) if _bezier_clear(p0, p1, c, others)), -0.3)
        else:
            first = min(a, b, key=g.nodes.index)
            base = [0.3, -0.3, 0.55, -0.55, 0.8, -0.8][siblings.index(e) % 6]
            # arc3 bends to one side of the *travel* direction, so a reversed edge needs the opposite sign
            rad = base * (1 if a == first else -1)
        rads[e] = rad
        if rad:
            d = p1 - p0
            for n, other in ((a, p1), (b, p0)):
                v = (pos[a] + pos[b]) / 2 + 0.5 * rad * numpy.array([d[1], -d[0]]) - pos[n]
                used[n].append(math.atan2(v[1], v[0]))

    for e in edges:
        a, b, label, ha, hb = e
        ast = style(ha, hb) + (",head_length=0.5,head_width=0.2" if (ha or hb) else "")
        if a == b:  # self-loop
            pref = away(a) if layout in ("circle", "spring") else (math.pi / 2 if layout == "lr" else 0.0)
            th = _free_angle(used[a], pref)
            used[a].append(th)
            c, rr = pos[a], radius[a]
            u = lambda ang: numpy.array([math.cos(ang), math.sin(ang)])
            h = 0.9 * rr + 0.45
            P0, P3 = c + rr * u(th - 0.5), c + rr * u(th + 0.5)
            P1, P2 = c + (rr + 1.35 * h) * u(th - 0.62), c + (rr + 1.35 * h) * u(th + 0.62)
            path = Path([P0, P1, P2, P3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
            arr = FancyArrowPatch(path=path, arrowstyle=ast, mutation_scale=14, lw=1.8,
                                  color=_INK, shrinkA=0, shrinkB=0, zorder=2)
            ax.add_patch(arr)
            arrows.append(arr)
            apex = 0.125 * P0 + 0.375 * P1 + 0.375 * P2 + 0.125 * P3
            include(apex, 0.1)
            if label:
                lines = label.count("\n") + 1
                label_spots.append(apex)
                edge_label(apex + (0.03 + 0.15 * lines) * u(th), label)
            continue

        p0, p1, rad = pos[a], pos[b], rads[e]
        arr = FancyArrowPatch(p0, p1, patchA=outer[a], patchB=outer[b], arrowstyle=ast,
                              connectionstyle=f"arc3,rad={rad}", mutation_scale=14, lw=1.8,
                              color=_INK, shrinkA=0, shrinkB=0, zorder=2)
        ax.add_patch(arr)
        arrows.append(arr)
        d = p1 - p0
        ctrl = (p0 + p1) / 2 + rad * numpy.array([d[1], -d[0]])
        include((p0 + p1) / 2 + 0.5 * rad * numpy.array([d[1], -d[0]]), 0.1)
        if label:
            # middle of the edge, unless another label is already there (crossing edges)
            for t in (0.5, 0.33, 0.67, 0.22, 0.78):
                spot = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * ctrl + t ** 2 * p1
                if all(numpy.linalg.norm(spot - q) > 0.55 for q in label_spots):
                    break
            label_spots.append(spot)
            edge_label(spot, label)

    # --- tape (Turing machine) -------------------------------------------------
    x_all = [p[0] for p in pts]
    y_all = [p[1] for p in pts]
    if g.tape:
        cw = 0.85
        cx = (min(x_all) + max(x_all)) / 2
        x0 = cx - cw * len(g.tape) / 2
        ty = min(y_all) - 1.3
        for i, sym in enumerate(g.tape):
            at_head = g.head == i
            ax.add_patch(Rectangle((x0 + i * cw, ty - cw / 2), cw, cw, fc=_HILITE if at_head else "white",
                                   ec=_INK, lw=1.8, zorder=3))
            t = ax.text(x0 + (i + 0.5) * cw, ty, sym, ha="center", va="center", color=_INK, zorder=6)
            texts.append(t)
        include((x0, ty - cw / 2 - 0.2))
        include((x0 + cw * len(g.tape), ty + cw / 2))
        if g.head is not None and 0 <= g.head < len(g.tape):
            hx = x0 + (g.head + 0.5) * cw
            arr = FancyArrowPatch((hx, ty - cw / 2 - 0.75), (hx, ty - cw / 2 - 0.05),
                                  arrowstyle="-|>,head_length=0.5,head_width=0.2", mutation_scale=14,
                                  lw=1.8, color=_INK, shrinkA=0, shrinkB=0, zorder=2)
            ax.add_patch(arr)
            arrows.append(arr)
            if g.state:
                add_label((hx, ty - cw / 2 - 1.05), _display_name(g.state)[0])

    x_all = [p[0] for p in pts]
    y_all = [p[1] for p in pts]
    x0, x1 = min(x_all) - 0.3, max(x_all) + 0.3
    y0, y1 = min(y_all) - 0.3, max(y_all) + 0.3
    if g.title:
        t = ax.text((x0 + x1) / 2, y1 + 0.35, g.title, ha="center", va="center",
                    fontweight="bold", color=_INK)
        texts.append(t)
        y1 += 0.8
    # never absurdly small: a lone node still gets a sensible canvas
    for lo, hi, minimum, setter in ((x0, x1, 4.0, "x"), (y0, y1, 2.6, "y")):
        if hi - lo < minimum:
            mid = (hi + lo) / 2
            lo, hi = mid - minimum / 2, mid + minimum / 2
            if setter == "x":
                x0, x1 = lo, hi
            else:
                y0, y1 = lo, hi

    w_d, h_d = x1 - x0, y1 - y0
    unit = min(0.85, 13.0 / w_d, 11.0 / h_d)  # inches per data unit
    unit = max(unit, 0.3)
    scale = min(1.0, unit / 0.85)
    fontsize = max(7.0, 13.0 * scale)
    for t in texts:
        t.set_fontsize(fontsize)
    for a_ in arrows:
        a_.set_mutation_scale(14 * max(scale, 0.55))
        a_.set_linewidth(1.8 * max(scale, 0.6))
    for p in ax.patches:
        p.set_clip_on(False)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    fig.set_size_inches(w_d * unit, h_d * unit)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.12,
                facecolor="white", transparent=False)
    return buf.getvalue()


def render_graph_spec(spec: str, kind: str = "graph") -> bytes:
    """Entry point for the [[graph: ...]] / [[dfa: ...]] / [[tm: ...]] tags and the !diagram command."""
    return render_graph(parse_graph_spec(spec, kind))
