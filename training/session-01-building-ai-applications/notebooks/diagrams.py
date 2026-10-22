"""Architecture diagrams for the session 01 notebooks.

build_notebooks.py draws each diagram with render_svg() and embeds it in the notebook's markdown as an
image, so the notebooks themselves import nothing from here. The usage below is for trying a diagram out.

    import diagrams
    diagrams.show("prompt")                      # notebook 01
    diagrams.show("knowledge")                   # notebook 02
    diagrams.show("retrieval")                   # notebook 03
    diagrams.show("tools")                       # notebook 04
    diagrams.show("agent")                       # notebook 05
    diagrams.show("verification")                # notebook 06
    diagrams.show("architecture", stage=2)       # the six components, one highlighted

Two engines draw the same description:

    "svg"        the default: a small layout written here; needs nothing beyond IPython
    "graphviz"   the Graphviz `dot` program through the `graphviz` package (needs `dot` installed)
    "auto"       Graphviz when `dot` is installed, otherwise "svg"

A flow without a `grid` is laid out in columns by the longest path to each node. A flow with a `grid`
places each node at (column, row); an edge may carry a third item, "above" or "below", to route it
around the nodes in its row (used for the arrows that return to an earlier box), or "elbow" to step down to
a lower row.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from html import escape

from IPython.display import SVG, display

FILL, EDGE, TEXT = "#e8f0fa", "#2f5f9e", "#16315a"
HOT_FILL, HOT_TEXT = "#2f5f9e", "#ffffff"
DIM_FILL, DIM_EDGE, DIM_TEXT = "#f3f4f6", "#8a94a3", "#6b7280"
FONT = "Helvetica, Arial, sans-serif"
DASH = ' stroke-dasharray="5 4"'

ALT = {
    "prompt": "Diagram: the question and the context feed the prompt, which goes to the model and produces the answer.",
    "knowledge": "Diagram: documents, metadata, chunks, embeddings, then the vector store. Below, built in notebook 03: the question and the retrieved passages from the vector store feed the prompt, then the model and the answer.",
    "retrieval": "Diagram: the question goes to retrieval, a search of the vector store that returns the closest chunks as passages; the passages and the question go into the prompt, then to generation, where the model writes an answer with citations.",
    "tools": "Diagram: the question goes to a search of the store, which returns passages for the prompt; the prompt goes to the model. New in this notebook: the model can call the tool get_indicator with an indicator, a country and optional start and end years; the data it returns goes back into the prompt; the model answers with text or a structured output, a filled-in form that other software can read. Separately, the same tool is served over MCP to any MCP client, such as this notebook, a chatbot or a desktop assistant.",
    "agent": "Diagram: the question goes to a model that chooses the next step. It can call search_documents, the search of notebook 03 now offered as a tool, which returns passages, or get_indicator, the tool of notebook 04, which returns data. The results go back to the model, which writes the answer. New in this notebook: the model chooses the steps, a step limit and a trace bound and record the loop, and the answer ends on the question of whether its numbers can be trusted, which notebook 06 takes up.",
    "lab": "Diagram: your documents are split and stored; a question retrieves the closest passages; a small open model, running in the notebook, writes the answer from them; every number of the answer is looked up in the passages and gets a mark; the answer is shown with its sources. If no passage is close enough, the application says it could not find the answer.",
    "verification": "Diagram: the question goes to the agent with tools of notebook 05, which writes an answer with claim tags; the verifier, pcn-core, puts a mark on each number: verified, flagged or unmarked.",
}


@dataclass
class Flow:
    """Nodes as (id, label, state) with state "normal", "hot", or "dim"; edges as (from, to) or (from, to, route)."""

    nodes: list[tuple[str, str, str]]
    edges: list[tuple]
    grid: dict[str, tuple[int, int]] | None = None
    caption: str = ""

    def state(self, node_id: str) -> str:
        return next(s for i, _, s in self.nodes if i == node_id)


def flow(name: str, stage: int | None = None) -> Flow:
    if name == "prompt":
        return Flow(
            [
                ("q", "question", "normal"),
                ("c", "context", "normal"),
                ("p", "prompt\ninstructions + context + question", "hot"),
                ("m", "model", "normal"),
                ("a", "answer", "normal"),
            ],
            [("q", "p"), ("c", "p"), ("p", "m"), ("m", "a")],
        )
    if name == "knowledge":
        return Flow(
            [
                ("d", "documents", "normal"),
                ("md", "metadata", "normal"),
                ("ch", "chunks", "normal"),
                ("e", "embeddings", "normal"),
                ("v", "vector store", "hot"),
                ("q", "question", "dim"),
                ("p", "prompt\ninstructions +\nretrieved passages", "dim"),
                ("m", "model", "dim"),
                ("a", "answer", "dim"),
            ],
            [("d", "md"), ("md", "ch"), ("ch", "e"), ("e", "v"), ("v", "p", "elbow"), ("q", "p"), ("p", "m"), ("m", "a")],
            grid={"d": (0, 0), "md": (1, 0), "ch": (2, 0), "e": (3, 0), "v": (4, 0), "q": (0, 1), "p": (3, 1), "m": (4, 1), "a": (5, 1)},
            caption="Solid: built in this notebook. Dashed: built in notebook 03",
        )
    if name == "retrieval":
        return Flow(
            [
                ("q", "question", "normal"),
                ("r", "retrieval\n(search the store)", "hot"),
                ("p", "passages\n(chunks found)", "hot"),
                ("pr", "prompt\ninstructions + passages\n+ question", "normal"),
                ("m", "generation\n(the model writes)", "normal"),
                ("a", "answer\nwith citations", "normal"),
            ],
            [("q", "r"), ("r", "p"), ("p", "pr"), ("q", "pr", "below"), ("pr", "m"), ("m", "a")],
            grid={"q": (0, 0), "r": (1, 0), "p": (2, 0), "pr": (3, 0), "m": (4, 0), "a": (5, 0)},
        )
    if name == "tools":
        return Flow(
            [
                ("q", "question", "normal"),
                ("s", "retrieval\n(search the store,\nreturns passages)", "normal"),
                ("pr", "prompt\ninstructions + passages\n+ question", "normal"),
                ("m", "model", "normal"),
                ("a", "answer, or a\nstructured output\n(a filled-in form)", "hot"),
                ("d", "data", "hot"),
                ("t", "tool call:\nget_indicator(indicator,\ncountry, start, end)", "hot"),
                ("ms", "the same tool\nserved over MCP", "hot"),
                ("c", "any MCP client\n(this notebook, a chatbot,\na desktop assistant)", "hot"),
            ],
            [("q", "s"), ("s", "pr"), ("pr", "m"), ("m", "a"), ("m", "t"), ("t", "d"), ("d", "pr"), ("ms", "c")],
            grid={"q": (0, 0), "s": (1, 0), "pr": (2, 0), "m": (3, 0), "a": (4, 0), "d": (2, 1), "t": (3, 1), "ms": (3, 2), "c": (4, 2)},
            caption="Light boxes: built in notebooks 01 to 03. Dark boxes: added in this notebook.",
        )
    if name == "agent":
        return Flow(
            [
                ("q", "question", "normal"),
                ("m1", "the model\nchooses the next step", "hot"),
                ("s", "search_documents\n(the search of 03,\nnow a tool)", "hot"),
                ("g", "get_indicator\n(the tool of 04)", "normal"),
                ("ps", "passages", "normal"),
                ("dt", "data", "normal"),
                ("m2", "the model writes", "normal"),
                ("a", "answer", "normal"),
                ("lim", "step limit\nand a trace", "hot"),
                ("chk", "can we trust\nthe numbers?", "hot"),
            ],
            [("q", "m1"), ("m1", "s"), ("m1", "g"), ("s", "ps"), ("g", "dt"), ("ps", "m2"), ("dt", "m2"), ("m2", "a"), ("m1", "lim"), ("a", "chk")],
            grid={"q": (0, 0), "m1": (1, 0), "s": (2, 0), "ps": (3, 0), "m2": (4, 0), "a": (5, 0), "lim": (1, 1), "g": (2, 1), "dt": (3, 1), "chk": (5, 1)},
            caption="Light boxes: built in notebooks 01 to 04. Dark boxes: added in this notebook.",
        )
    if name == "verification":
        return Flow(
            [
                ("q", "question", "normal"),
                ("ag", "agent with tools\n(notebook 05)", "normal"),
                ("an", "answer with\nclaim tags", "hot"),
                ("pcn", "verifier\n(pcn-core)", "hot"),
                ("mk", "a mark on each number:\nverified, flagged, unmarked", "hot"),
            ],
            [("q", "ag"), ("ag", "an"), ("an", "pcn"), ("pcn", "mk")],
            grid={"q": (0, 0), "ag": (1, 0), "an": (2, 0), "pcn": (3, 0), "mk": (4, 0)},
            caption="Light boxes: built in notebooks 01 to 05. Dark boxes: added in this notebook.",
        )
    if name == "lab":
        return Flow(
            [
                ("d", "your documents\n(or the session's)", "hot"),
                ("sp", "split into chunks\n(notebook 02)", "normal"),
                ("st", "store\n(notebook 02)", "normal"),
                ("q", "your question", "normal"),
                ("rt", "retrieve the closest\npassages (notebook 03)", "normal"),
                ("nf", "no close passage:\nsay so", "hot"),
                ("m", "small open model\nwrites the answer", "hot"),
                ("ck", "each number looked up\nin the passages", "hot"),
                ("a", "answer, marks\nand sources", "hot"),
            ],
            [("d", "sp"), ("sp", "st"), ("q", "rt"), ("st", "rt"), ("rt", "m"), ("rt", "nf"), ("m", "ck"), ("ck", "a")],
            grid={"d": (0, 0), "sp": (1, 0), "st": (2, 0), "q": (2, 1), "rt": (3, 0), "m": (4, 0), "ck": (5, 0), "a": (6, 0), "nf": (4, 1)},
            caption="Light boxes: built in notebooks 02 and 03. Dark boxes: new in the lab.",
        )
    if name == "architecture":
        if stage is None:
            raise ValueError("architecture needs stage=1..6")
        labels = [
            "Instructions\n& context",
            "Prepared\nknowledge",
            "Retrieval",
            "Tools &\nstructured output",
            "Workflow\nor agent",
            "Verification\n& evaluation",
        ]
        nodes = [(f"s{i}", t, "hot" if i == stage else ("dim" if i > stage else "normal")) for i, t in enumerate(labels, 1)]
        return Flow(nodes, [(f"s{i}", f"s{i + 1}") for i in range(1, 6)])
    raise ValueError(f"unknown diagram {name!r}")


# ---- engine: svg -----------------------------------------------------------

def _ranks(f: Flow) -> dict[str, int]:
    rank = {i: 0 for i, _, _ in f.nodes}
    for _ in f.nodes:
        for e in f.edges:
            rank[e[1]] = max(rank[e[1]], rank[e[0]] + 1)
    return rank


def _size(label: str) -> tuple[int, int]:
    lines = label.split("\n")
    return max(112, int(max(len(t) for t in lines) * 7.6) + 30), 24 + 19 * len(lines)


def _layout(f: Flow, size: dict[str, tuple[int, int]], gap_x: int, gap_y: int, pad: int, top: int):
    """Positions as {id: (x, y, w, h)}, the canvas width, and the y of the bottom of the lowest node."""
    if f.grid is None:
        rank = _ranks(f)
        columns: dict[int, list[str]] = {}
        for i, _, _ in f.nodes:
            columns.setdefault(rank[i], []).append(i)
        col_w = {r: max(size[i][0] for i in ids) for r, ids in columns.items()}
        col_h = {r: sum(size[i][1] for i in ids) + gap_y * (len(ids) - 1) for r, ids in columns.items()}
        height = max(col_h.values())
        pos, x = {}, pad
        for r in sorted(columns):
            y = top + (height - col_h[r]) / 2
            for i in columns[r]:
                w, h = size[i]
                pos[i] = (x + (col_w[r] - w) / 2, y, w, h)
                y += h + gap_y
            x += col_w[r] + gap_x
        return pos, x - gap_x + pad, top + height
    n_cols = max(c for c, _ in f.grid.values()) + 1
    n_rows = max(r for _, r in f.grid.values()) + 1
    col_w = [max([size[i][0] for i, (c, _) in f.grid.items() if c == k] or [0]) for k in range(n_cols)]
    row_h = [max([size[i][1] for i, (_, r) in f.grid.items() if r == k] or [0]) for k in range(n_rows)]
    xs = [pad + sum(col_w[:k]) + gap_x * k for k in range(n_cols)]
    ys = [top + sum(row_h[:k]) + gap_y * k for k in range(n_rows)]
    pos = {}
    for i, (c, r) in f.grid.items():
        w, h = size[i]
        pos[i] = (xs[c] + (col_w[c] - w) / 2, ys[r] + (row_h[r] - h) / 2, w, h)
    return pos, xs[-1] + col_w[-1] + pad, ys[-1] + row_h[-1]


def render_svg(f: Flow) -> str:
    size = {i: _size(t) for i, t, _ in f.nodes}
    routed = [e for e in f.edges if len(e) > 2]
    pad = 18
    top = pad + (26 if any(e[2] == "above" for e in routed) else 0)
    gap_y = 56 if routed and f.grid else 26
    pos, width, bottom = _layout(f, size, 44, gap_y, pad, top)
    below = 26 if any(e[2] == "below" for e in routed) else 0
    note = 24 if f.caption else 0
    height = bottom + below + note + pad

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="max-width:100%;height:auto" font-family="{FONT}">',
        f'<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{EDGE}"/></marker>'
        f'<marker id="ahd" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{DIM_EDGE}"/></marker></defs>',
        f'<rect width="{width}" height="{height}" rx="10" fill="#ffffff" stroke="#e1e6ee"/>',
    ]
    for e in f.edges:
        a, b = e[0], e[1]
        route = e[2] if len(e) > 2 else None
        ax, ay, aw, ah = pos[a]
        bx, by, bw, bh = pos[b]
        dim = f.state(b) == "dim"
        acx, bcx = ax + aw / 2, bx + bw / 2
        if route == "below":
            yr = max(ay + ah, by + bh) + 18
            shape = f'<polyline points="{acx},{ay + ah} {acx},{yr} {bcx},{yr} {bcx},{by + bh + 2}" fill="none"'
        elif route == "above":
            yr = min(ay, by) - 18
            shape = f'<polyline points="{acx},{ay} {acx},{yr} {bcx},{yr} {bcx},{by - 2}" fill="none"'
        elif route == "elbow":
            ym = (ay + ah + by) / 2
            shape = f'<polyline points="{acx},{ay + ah} {acx},{ym} {bcx},{ym} {bcx},{by - 2}" fill="none"'
        elif f.grid is not None and abs((ay + ah / 2) - (by + bh / 2)) < 1 and bx < ax:
            shape = f'<line x1="{ax}" y1="{ay + ah / 2}" x2="{bx + bw + 2}" y2="{by + bh / 2}"'
        elif f.grid is None or abs((ay + ah / 2) - (by + bh / 2)) < 1:
            shape = f'<line x1="{ax + aw}" y1="{ay + ah / 2}" x2="{bx - 2}" y2="{by + bh / 2}"'
        elif by > ay:
            shape = f'<line x1="{acx}" y1="{ay + ah}" x2="{bcx}" y2="{by - 2}"'
        else:
            shape = f'<line x1="{acx}" y1="{ay}" x2="{bcx}" y2="{by + bh + 2}"'
        out.append(
            f'{shape} stroke="{DIM_EDGE if dim else EDGE}" stroke-width="1.8"'
            f'{DASH if dim else ""} marker-end="url(#{"ahd" if dim else "ah"})"/>'
        )
    for i, label, state in f.nodes:
        x, y, w, h = pos[i]
        fill, edge, color = {"hot": (HOT_FILL, HOT_FILL, HOT_TEXT), "dim": (DIM_FILL, DIM_EDGE, DIM_TEXT)}.get(state, (FILL, EDGE, TEXT))
        lines = label.split("\n")
        tspans = "".join(
            f'<tspan x="{x + w / 2}" y="{y + h / 2 + (k - (len(lines) - 1) / 2) * 19 + 5}">{escape(t)}</tspan>' for k, t in enumerate(lines)
        )
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{edge}" stroke-width="1.6"'
            f'{DASH if state == "dim" else ""}/>'
            f'<text text-anchor="middle" font-size="14" fill="{color}" font-weight="{600 if state == "hot" else 500}">{tspans}</text>'
        )
    if f.caption:
        out.append(f'<text x="{pad}" y="{height - pad + 2}" font-size="12" fill="{DIM_TEXT}">{escape(f.caption)}</text>')
    out.append("</svg>")
    return "".join(out)


# ---- engine: graphviz ------------------------------------------------------

def to_dot(f: Flow) -> str:
    lines = [
        "digraph {",
        '  rankdir=LR; bgcolor="white"; pad=0.2; nodesep=0.35; ranksep=0.45;',
        f'  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=13 fillcolor="{FILL}" color="{EDGE}" fontcolor="{TEXT}" margin="0.2,0.12"];',
        f'  edge [color="{EDGE}" arrowsize=0.8];',
    ]
    for i, label, state in f.nodes:
        extra = {
            "hot": f' fillcolor="{HOT_FILL}" color="{HOT_FILL}" fontcolor="{HOT_TEXT}"',
            "dim": f' style="rounded,dashed,filled" fillcolor="{DIM_FILL}" color="{DIM_EDGE}" fontcolor="{DIM_TEXT}"',
        }.get(state, "")
        text = label.replace("\n", "\\n")
        lines.append(f'  {i} [label="{text}"{extra}];')
    for e in f.edges:
        style = f' [style=dashed color="{DIM_EDGE}"]' if f.state(e[1]) == "dim" else ""
        lines.append(f"  {e[0]} -> {e[1]}{style};")
    lines.append("}")
    return "\n".join(lines)


def graphviz_available() -> bool:
    try:
        import graphviz  # noqa: F401
    except ImportError:
        return False
    return shutil.which("dot") is not None


def render_graphviz(f: Flow) -> str:
    import graphviz

    return graphviz.Source(to_dot(f)).pipe(format="svg", encoding="utf-8")


# ---- display ---------------------------------------------------------------

def show(name: str, stage: int | None = None, engine: str = "svg") -> None:
    """Display a diagram in the notebook. `engine` is "svg" (default), "graphviz", or "auto"."""
    f = flow(name, stage)
    if engine not in ("auto", "svg", "graphviz"):
        raise ValueError("engine must be 'auto', 'svg', or 'graphviz'")
    use_graphviz = engine != "svg" and graphviz_available()
    if engine == "graphviz" and not use_graphviz:
        print("Graphviz is not available here (needs the `graphviz` package and the `dot` program); drawing with the built-in SVG engine.")
    display(SVG(data=render_graphviz(f) if use_graphviz else render_svg(f)))
