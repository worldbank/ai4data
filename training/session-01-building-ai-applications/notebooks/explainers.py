"""Animated SVG explainers for notebook 02: chunking and embeddings.

build_notebooks.py draws each one with chunking_svg() or embeddings_svg() and embeds it in the notebook's
markdown as an image, so the notebooks import nothing from here. The animations are SMIL, built into the SVG:
they loop, need no packages, and show their last frame where animation is not supported.

The chunk boundaries come from the splitters the notebook uses, run on assets/documents/kenya-digital.md.
The similarity scores on the map are the ones the notebook prints; the dot positions and the example
vector values are illustrative, and the picture says so.

    python explainers.py        # writes chunking.svg and embeddings.svg to the current folder
"""
import base64
import re
from html import escape
from pathlib import Path

import frontmatter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).resolve().parent
DOC = HERE.parent / "assets" / "documents" / "kenya-digital.md"

INK, BLUE, PALE, GREY, LIGHT = "#16315a", "#2f5f9e", "#e8f0fa", "#6b7280", "#f3f4f6"
CHUNK_COLORS = ["#4c78c8", "#e08a2e", "#3a9d6b", "#8a5cc2"]
SANS = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
FADE = 0.4


def anim(s, e, total, base):
    """An element visible from s to e seconds of a loop of `total` seconds; `base` is its opacity without animation."""
    pts = [(0.0, 0.0)]
    if s > 0:
        pts += [(s, 0.0)]
    pts += [(s + FADE, 1.0), (max(e, s + FADE), 1.0)]
    if e < total:
        pts += [(min(e + FADE, total), 0.0)]
    if pts[-1][0] < total:
        pts += [(total, pts[-1][1])]
    keys, vals = [], []
    for t, v in pts:
        if keys and abs(t - keys[-1]) < 1e-9:
            vals[-1] = v
            continue
        keys.append(t)
        vals.append(v)
    kt = ";".join(f"{k / total:.4f}" for k in keys)
    vv = ";".join(f"{v:g}" for v in vals)
    return f'opacity="{base}"><animate attributeName="opacity" dur="{total}s" repeatCount="indefinite" keyTimes="{kt}" values="{vv}"/>'


def group(inner, s, e, total, base):
    return f"<g {anim(s, e, total, base)}{inner}</g>"


# ----------------------------------------------------------------- chunking
def wrap(text, cols=78):
    """Wrapped lines as (start, end, text) in original offsets, with a gap flag for a new paragraph."""
    lines, pos = [], 0
    for para in re.split(r"\n\n", text):
        start = text.index(para, pos)
        pos = start + len(para)
        words = [(m.start() + start, m.end() + start, m.group()) for m in re.finditer(r"\S+", para)]
        cur = []
        for w in words:
            width = sum(len(x[2]) for x in cur) + len(cur) + len(w[2])
            if cur and width > cols:
                lines.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur), False))
                cur = []
            cur.append(w)
        if cur:
            lines.append((cur[0][0], cur[-1][1], " ".join(x[2] for x in cur), False))
        lines[-1] = lines[-1][:3] + (True,)  # paragraph end
    return lines


def chunking_svg():
    post = frontmatter.load(DOC)
    text = post.content.strip()
    windows = [
        (c.metadata["start_index"], c.metadata["start_index"] + len(c.page_content))
        for c in RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, add_start_index=True).split_documents([Document(page_content=text)])
    ]
    heads = [m.start() for m in re.finditer(r"^## .*$", text, re.M)]
    sections = [(0, heads[1])] + [(heads[i], heads[i + 1] if i + 1 < len(heads) else len(text)) for i in range(1, len(heads))]
    names = ["Internet use", "Mobile subscriptions", "Population", "Reading the two indicators together"]

    lines = wrap(text)
    cw, lh, gap, x0, y0 = 7.2, 17, 10, 24, 78
    ys, y = [], y0
    for (_, _, _, para_end) in lines:
        ys.append(y)
        y += lh + (gap if para_end else 0)
    doc_bottom = y
    W, H, T = 960, int(doc_bottom + 80), 17.0

    def overlay(rng, color, opacity=0.30):
        out = []
        for (ls, le, t, _), ly in zip(lines, ys):
            a, b = max(rng[0], ls), min(rng[1], le)
            if a < b:
                out.append(f'<rect x="{x0 + (a - ls) * cw:.1f}" y="{ly - 12:.1f}" width="{(b - a) * cw:.1f}" height="{lh}" rx="3" fill="{color}" fill-opacity="{opacity}"/>')
        return "".join(out)

    def cut_mark(pos):
        for (ls, le, t, _), ly in zip(lines, ys):
            if ls <= pos <= le:
                x = x0 + (pos - ls) * cw
                return f'<line x1="{x:.1f}" y1="{ly - 14:.1f}" x2="{x:.1f}" y2="{ly + 7:.1f}" stroke="#c0392b" stroke-width="2"/>'
        return ""

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
        f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
        f'<text x="{x0}" y="34" font-size="17" font-weight="700" fill="{INK}">Chunking: the same brief, cut two ways</text>',
        f'<text x="{x0}" y="54" font-size="12.5" fill="{GREY}">kenya-digital.md, 1,301 characters. The chunk boundaries below come from the splitters the notebook uses.</text>',
    ]
    for (ls, le, t, _), ly in zip(lines, ys):
        weight = "700" if t.startswith("#") else "400"
        parts.append(f'<text x="{x0}" y="{ly}" font-family="{MONO}" font-size="12" font-weight="{weight}" fill="{INK}" textLength="{len(t) * cw:.1f}" lengthAdjust="spacing" xml:space="preserve">{escape(t)}</text>')

    # animated layers: fixed windows
    t_w0, step, w_end = 2.6, 1.3, 10.4
    legend_x = 650
    win_notes = [
        "ends where a paragraph ends",
        "ends on a heading, cut off from its paragraph",
        "starts with that heading again (the overlap)",
        "mixes two sections in one chunk",
    ]
    for i, rng in enumerate(windows):
        s = t_w0 + i * step
        inner = overlay(rng, CHUNK_COLORS[i]) + (cut_mark(rng[1]) if i < len(windows) - 1 else "")
        parts.append(group(inner, s, w_end, T, 0))
        ly = 96 + i * 62
        leg = (
            f'<rect x="{legend_x}" y="{ly - 14}" width="16" height="16" rx="3" fill="{CHUNK_COLORS[i]}" fill-opacity="0.55"/>'
            f'<text x="{legend_x + 24}" y="{ly}" font-size="13.5" font-weight="700" fill="{INK}">Window {i + 1} · {rng[1] - rng[0]} characters</text>'
            f'<text x="{legend_x + 24}" y="{ly + 18}" font-size="12.5" fill="{GREY}">{escape(win_notes[i])}</text>'
        )
        parts.append(group(leg, s, w_end, T, 0))
    # animated layers: sections
    t_s0, s_end = 10.8, 16.4
    for i, rng in enumerate(sections):
        s = t_s0 + i * 1.0
        parts.append(group(overlay(rng, CHUNK_COLORS[i]), s, s_end, T, 1))
        ly = 96 + i * 62
        leg = (
            f'<rect x="{legend_x}" y="{ly - 14}" width="16" height="16" rx="3" fill="{CHUNK_COLORS[i]}" fill-opacity="0.55"/>'
            f'<text x="{legend_x + 24}" y="{ly}" font-size="13.5" font-weight="700" fill="{INK}">{escape(names[i])}</text>'
            f'<text x="{legend_x + 24}" y="{ly + 18}" font-size="12.5" fill="{GREY}">{rng[1] - rng[0]} characters · one idea, with its heading</text>'
        )
        parts.append(group(leg, s, s_end, T, 1))
    # captions
    cy = H - 26
    caps = [
        ("A  ·  The document: one country brief, with headings", 0, 2.7, 0),
        ("B  ·  Fixed windows: cut every 400 characters, whatever the text means", 2.7, 10.5, 0),
        ("C  ·  Section chunks: one per heading, so each chunk keeps its meaning and can be cited", 10.5, 16.5, 1),
    ]
    for label, s, e, base in caps:
        parts.append(group(f'<text x="{x0}" y="{cy}" font-size="14.5" font-weight="700" fill="{BLUE}">{escape(label)}</text>', s, e, T, base))
    parts.append("</svg>")
    return "".join(parts), windows, sections


# --------------------------------------------------------------- embeddings
def embeddings_svg():
    W, H, T = 960, 430, 19.0
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
        f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{BLUE}"/></marker></defs>',
        f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
        f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">Embeddings: from text to a position on a map of meaning</text>',
    ]
    # phase 1: the chunk card
    card = (
        f'<rect x="24" y="76" width="262" height="168" rx="10" fill="{PALE}" stroke="{BLUE}" stroke-width="1.6"/>'
        f'<text x="38" y="100" font-size="13" font-weight="700" fill="{BLUE}">Kenya · Internet use</text>'
        + "".join(
            f'<text x="38" y="{124 + 19 * i}" font-size="12.5" fill="{INK}">{escape(t)}</text>'
            for i, t in enumerate(
                ["The share of individuals using", "the internet in Kenya rose from", "7.2 percent of the population in", "2010 to 16.6 percent in 2015 and", "24.7 percent in 2020. Growth …"]
            )
        )
    )
    p.append(group(card, 0, 18, T, 1))
    # phase 2: numbers
    nums = (
        f'<line x1="290" y1="160" x2="352" y2="160" stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"/>'
        f'<text x="321" y="146" text-anchor="middle" font-size="11" fill="{GREY}">embedding</text><text x="321" y="176" text-anchor="middle" font-size="11" fill="{GREY}">model</text>'
        f'<rect x="358" y="76" width="130" height="168" rx="10" fill="#ffffff" stroke="{BLUE}" stroke-width="1.6"/>'
        f'<text x="423" y="98" text-anchor="middle" font-size="12.5" font-weight="700" fill="{BLUE}">384 numbers</text>'
        + "".join(
            f'<text x="423" y="{120 + 20 * i}" text-anchor="middle" font-family="{MONO}" font-size="12.5" fill="{INK}">{v}</text>'
            for i, v in enumerate(["0.12", "-0.41", "0.07", "0.33", "-0.18", "0.26", "…"])
        )
    )
    p.append(group(nums, 3.6, 18, T, 1))
    # phase 3: the map
    px0, px1, py0, py1 = 508, 936, 52, 372
    p.append(group(
        f'<rect x="{px0}" y="{py0}" width="{px1 - px0}" height="{py1 - py0}" rx="10" fill="{LIGHT}" stroke="#d5dbe5"/>'
        f'<text x="{px0 + 14}" y="{py0 + 22}" font-size="12.5" font-weight="700" fill="{GREY}">A simplified map of meaning</text>',
        8.0, 18, T, 1))
    qx, qy = 632.0, 214.0
    import math

    items = [  # label, score, angle, radius
        ("Kenya · Internet use", 0.407, -26, 82),
        ("Rwanda · Internet use", 0.398, 26, 86),
        ("Kenya · Mobile subscriptions", 0.290, -46, 140),
        ("Rwanda · Mobile subscriptions", 0.235, 46, 150),
        ("Kenya · Growth", 0.042, -15, 238),
        ("Rwanda · Growth", 0.045, 15, 238),
    ]
    pts = []
    for label, score, ang, r in items:
        pts.append((label, score, qx + r * math.cos(math.radians(ang)), qy + r * math.sin(math.radians(ang))))
    first = pts[0]
    # where the card's chunk lands
    p.append(group(
        f'<path d="M490,150 C560,100 640,100 {first[2] - 10:.1f},{first[3] - 7:.1f}" fill="none" stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ar)"/>'
        f'<text x="560" y="94" font-size="11" fill="{BLUE}">this chunk lands here</text>', 8.4, 13.0, T, 0))
    for i, (label, score, x, y) in enumerate(pts):
        s = 8.4 + 0.55 * i
        top2 = i < 2
        dot = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{BLUE if top2 else PALE}" stroke="{BLUE}" stroke-width="1.6"/>'
            f'<text x="{x:.1f}" y="{y + 25:.1f}" text-anchor="middle" font-size="11" fill="{INK}">{escape(label)}</text>'
        )
        p.append(group(dot, s, 18, T, 1))
    # phase 4: the question
    q = [
        f'<line x1="{qx}" y1="{qy}" x2="{x:.1f}" y2="{y:.1f}" stroke="{BLUE if i < 2 else "#9aa3b2"}" stroke-width="{2 if i < 2 else 1}" stroke-dasharray="{"" if i < 2 else "4 4"}"/>'
        f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="{BLUE if i < 2 else GREY}">{score:.3f}</text>'
        for i, (label, score, x, y) in enumerate(pts)
    ]
    star = (
        f'<polygon points="{qx},{qy - 13} {qx + 4},{qy - 4} {qx + 13},{qy - 4} {qx + 6},{qy + 2} {qx + 9},{qy + 12} {qx},{qy + 6} {qx - 9},{qy + 12} {qx - 6},{qy + 2} {qx - 13},{qy - 4} {qx - 4},{qy - 4}" fill="#e08a2e" stroke="#b86a14" stroke-width="1.2"/>'
        f'<text x="{qx}" y="{qy - 24}" text-anchor="middle" font-size="12" font-weight="700" fill="#b86a14">your question</text>'
        f'<text x="{qx}" y="{qy + 32}" text-anchor="middle" font-size="11" fill="#b86a14">“share of people</text>'
        f'<text x="{qx}" y="{qy + 45}" text-anchor="middle" font-size="11" fill="#b86a14">using the internet”</text>'
    )
    p.append(group("".join(q) + star, 13.2, 18, T, 1))
    p.append(group(
        f'<text x="{px1 - 12}" y="{py1 - 10}" text-anchor="end" font-size="11" fill="{GREY}">Closest two are retrieved. Scores are similarity, from the notebook run; positions are illustrative.</text>',
        13.2, 18, T, 1))
    caps = [
        ("1  ·  A chunk of text from the collection", 0, 3.7, 0),
        ("2  ·  An embedding model turns it into a list of numbers (illustrative values)", 3.7, 8.1, 0),
        ("3  ·  Every chunk becomes a dot: similar meaning lands close together", 8.1, 13.3, 0),
        ("4  ·  A question becomes a dot too, and the closest chunks are the ones retrieved", 13.3, 18.1, 1),
    ]
    for label, s, e, base in caps:
        p.append(group(f'<text x="24" y="{H - 28}" font-size="14.5" font-weight="700" fill="{BLUE}">{escape(label)}</text>', s, e, T, base))
    p.append("</svg>")
    return "".join(p)


def data_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


INSTR, CTX, QST = "#2f5f9e", "#2f8f5b", "#d9822b"   # instructions, context, question


def wrap_lines(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def text_block(x, y, lines, size=12.5, fill=INK, lh=17, weight="400"):
    return "".join(f'<text x="{x}" y="{y + i * lh}" font-size="{size}" fill="{fill}" font-weight="{weight}">{escape(t)}</text>' for i, t in enumerate(lines))


# --------------------------------------------------------------- notebook 01: the three parts of a prompt ------------------------------
def prompt_svg():
    W, H, T = 1000, 430, 17.0
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
         f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{BLUE}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">What goes into one call to a language model</text>']
    parts = [
        ("Instructions", "set by the developer", INSTR, wrap_lines("You are a research assistant for development data. Answer only from the supplied context. Cite the document id…", 44)),
        ("Context", "supplied by the application", CTX, wrap_lines("[kenya-digital] The share of individuals using the internet in Kenya rose from 7.2 percent of the population in 2010 to 16.6 percent in 2015…", 44)),
        ("Question", "asked by the user", QST, wrap_lines("How have internet access and economic development evolved in Rwanda and Kenya? …", 44)),
    ]
    y, centres = 62, []
    for i, (name, who, color, lines) in enumerate(parts):
        s = 0.4 + 1.2 * i
        h = 30 + 17 * len(lines) + 12
        centres.append(y + h / 2)
        block = (f'<rect x="24" y="{y}" width="318" height="{h}" rx="10" fill="{color}" fill-opacity="0.12" stroke="{color}" stroke-width="1.8"/>'
                 f'<text x="38" y="{y + 22}" font-size="13.5" font-weight="700" fill="{color}">{name}</text>'
                 f'<text x="330" y="{y + 22}" text-anchor="end" font-size="11.5" fill="{GREY}">{who}</text>'
                 + text_block(38, y + 44, lines, size=12, fill=INK, lh=17))
        p.append(group(block, s, 16.3, T, 1))
        y += h + 14
    # the prompt card
    bands = [150, 188, 226]
    card = ("".join(f'<line x1="342" y1="{c}" x2="404" y2="{b}" stroke="{col}" stroke-width="2" marker-end="url(#ar)"/>' for c, b, col in zip(centres, bands, [INSTR, CTX, QST]))
            + f'<rect x="398" y="118" width="150" height="150" rx="10" fill="#ffffff" stroke="{BLUE}" stroke-width="1.8"/>'
            f'<text x="473" y="108" text-anchor="middle" font-size="13" font-weight="700" fill="{BLUE}">One prompt</text>'
            + "".join(f'<rect x="414" y="{132 + 38 * i}" width="126" height="30" rx="6" fill="{c}" fill-opacity="0.85"/>'
                      f'<text x="477" y="{152 + 38 * i}" text-anchor="middle" font-size="12.5" font-weight="700" fill="#ffffff">{n}</text>'
                      for i, (n, c) in enumerate([("instructions", INSTR), ("context", CTX), ("question", QST)])))
    p.append(group(card, 5.2, 16.3, T, 1))
    model = (f'<line x1="548" y1="190" x2="598" y2="190" stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"/>'
             f'<rect x="604" y="160" width="110" height="60" rx="10" fill="{PALE}" stroke="{BLUE}" stroke-width="1.8"/>'
             f'<text x="659" y="196" text-anchor="middle" font-size="15" font-weight="700" fill="{BLUE}">model</text>')
    p.append(group(model, 9.4, 16.3, T, 1))
    ans_lines = wrap_lines("In Kenya, the share of individuals using the internet rose from 7.2 percent of the population in 2010 to 35.0 percent in 2024 [kenya-digital].", 30)
    answer = (f'<line x1="714" y1="190" x2="764" y2="190" stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"/>'
              f'<rect x="770" y="112" width="206" height="176" rx="10" fill="#ffffff" stroke="{BLUE}" stroke-width="1.8"/>'
              f'<text x="784" y="136" font-size="13" font-weight="700" fill="{BLUE}">Answer</text>'
              + text_block(784, 160, ans_lines, size=12, lh=17)
              + f'<rect x="784" y="{160 + 17 * len(ans_lines) + 4}" width="104" height="20" rx="5" fill="{CTX}" fill-opacity="0.18"/>'
              f'<text x="792" y="{160 + 17 * len(ans_lines) + 18}" font-size="11.5" font-weight="700" fill="{CTX}">cites its source</text>')
    p.append(group(answer, 12.6, 16.3, T, 1))
    caps = [
        ("1  ·  Three parts go into every call, and each is set by someone different", 0, 5.1, 0),
        ("2  ·  The application puts them together into one prompt", 5.1, 9.3, 0),
        ("3  ·  The model writes its answer from that prompt", 9.3, 16.4, 1),
    ]
    for label, s, e, base in caps:
        p.append(group(f'<text x="24" y="{H - 24}" font-size="14.5" font-weight="700" fill="{BLUE}">{escape(label)}</text>', s, e, T, base))
    p.append("</svg>")
    return "".join(p)


# --------------------------------------------------------------- notebook 01: without and with context -------------------------------------------------------
def before_after_svg():
    W, H = 1000, 330
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
         f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{BLUE}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">The same question, without and with context</text>',
         f'<line x1="500" y1="56" x2="500" y2="{H - 20}" stroke="#d5dbe5" stroke-width="1.5"/>']
    def panel(x0, title, color, inputs, answer_lines, tags):
        out = [f'<text x="{x0}" y="72" font-size="14.5" font-weight="700" fill="{color}">{title}</text>']
        for i, t in enumerate(inputs):
            out.append(f'<rect x="{x0}" y="{92 + 34 * i}" width="150" height="28" rx="6" fill="{PALE}" stroke="{BLUE}"/>'
                       f'<text x="{x0 + 75}" y="{111 + 34 * i}" text-anchor="middle" font-size="12" fill="{INK}">{escape(t)}</text>')
        mid = 92 + 34 * (len(inputs) - 1) / 2 + 14
        out.append(f'<line x1="{x0 + 150}" y1="{mid}" x2="{x0 + 190}" y2="{mid}" stroke="{BLUE}" stroke-width="2" marker-end="url(#ar)"/>'
                   f'<rect x="{x0 + 196}" y="{mid - 22}" width="70" height="44" rx="8" fill="{PALE}" stroke="{BLUE}" stroke-width="1.6"/>'
                   f'<text x="{x0 + 231}" y="{mid + 5}" text-anchor="middle" font-size="13" font-weight="700" fill="{BLUE}">model</text>')
        out.append(f'<rect x="{x0}" y="196" width="440" height="92" rx="10" fill="#ffffff" stroke="{color}" stroke-width="1.8"/>')
        out.append(text_block(x0 + 12, 218, answer_lines, size=12, lh=17))
        out.append("".join(f'<rect x="{x0 + 12 + 130 * i}" y="262" width="120" height="20" rx="5" fill="{c}" fill-opacity="0.16"/>'
                           f'<text x="{x0 + 72 + 130 * i}" y="276" text-anchor="middle" font-size="11.5" font-weight="700" fill="{c}">{escape(t)}</text>' for i, (t, c) in enumerate(tags)))
        return "".join(out)
    p.append(panel(24, "Without context", "#9a3a2f", ["question"],
                   wrap_lines("“Kenya is one of East Africa's most connected economies. Internet penetration is often reported above 80 percent of the population when measured by subscriptions…”", 66),
                   [("no source", "#9a3a2f"), ("no date", "#9a3a2f"), ("blends measures", "#9a3a2f")]))
    p.append(panel(524, "With the four country briefs", CTX, ["question", "briefs as context"],
                   wrap_lines("“In Kenya, the share of individuals using the internet rose from 7.2 percent of the population in 2010 to 35.0 percent in 2024 [kenya-digital]…”", 66),
                   [("cites its source", CTX), ("dated", CTX), ("one measure", CTX)]))
    p.append("</svg>")
    return "".join(p)


# --------------------------------------------------------------- notebook 01: what context costs ---------------------------------------------------
def desk_svg():
    W, H = 1000, 300
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">Why the application has to choose what the model reads</text>']
    # desk
    p.append(f'<rect x="40" y="70" width="420" height="150" rx="12" fill="{LIGHT}" stroke="{BLUE}" stroke-width="2"/>'
             f'<text x="52" y="94" font-size="13" font-weight="700" fill="{BLUE}">The model’s context window: room for a limited number of pages</text>')
    for i in range(4):
        p.append(f'<rect x="{64 + i * 34}" y="{118 + (i % 2) * 6}" width="26" height="36" rx="3" fill="#ffffff" stroke="{CTX}" stroke-width="1.6"/>')
    p.append(f'<text x="64" y="182" font-size="12.5" fill="{INK}">The four briefs used in this notebook:</text>'
             f'<text x="64" y="200" font-size="12.5" font-weight="700" fill="{CTX}">about 1,024 tokens. They fit, with room to spare.</text>')
    # collection
    p.append(f'<text x="560" y="94" font-size="13" font-weight="700" fill="#9a3a2f">A real collection: thousands of documents</text>')
    for r in range(5):
        for c in range(18):
            p.append(f'<rect x="{556 + c * 24}" y="{108 + r * 28}" width="18" height="24" rx="2" fill="#ffffff" stroke="#9a3a2f" stroke-width="1" opacity="0.8"/>')
    p.append(f'<line x1="480" y1="150" x2="540" y2="150" stroke="#9a3a2f" stroke-width="2" stroke-dasharray="5 4"/>'
             f'<text x="510" y="140" text-anchor="middle" font-size="22" fill="#9a3a2f">✕</text>'
             f'<text x="560" y="262" font-size="12.5" fill="{INK}">It cannot all go on the desk, and every page costs money and dilutes the rest.</text>'
             f'<text x="24" y="{H - 14}" font-size="14.5" font-weight="700" fill="{BLUE}">So the next notebooks prepare the collection, and the application picks the few pages that matter for each question.</text>')
    p.append("</svg>")
    return "".join(p)


# --------------------------------------------------------------- notebook 02: metadata and text
def metadata_svg():
    meta = frontmatter.load(DOC.parent / "regional-2016.md")
    body = [p.strip() for p in meta.content.split("\n\n") if p.strip()]
    heading, paragraphs = body[0].lstrip("# ").strip(), body[1:]
    fields = [(k, str(meta.metadata[k])) for k in ("id", "title", "type", "country", "date", "source")]
    W, pad, top = 1000, 24, 62
    ORANGE = "#d9822b"

    # left card: the metadata
    left, y, date_box = [], top + 50, None
    for key, value in fields:
        lines = wrap_lines(value, 34)
        h = 17 * len(lines)
        if key == "date":
            date_box = (y - 15, h + 8)
            left.append(f'<rect x="{pad + 10}" y="{y - 15}" width="{360 - 20}" height="{h + 8}" rx="6" fill="{ORANGE}" fill-opacity="0.18" stroke="{ORANGE}" stroke-width="1.5"/>')
        left.append(f'<text x="{pad + 22}" y="{y}" font-size="12.5" font-weight="700" fill="{GREY}">{key}</text>')
        left.append(text_block(pad + 92, y, lines, size=12.5, fill=INK, lh=17, weight="700" if key == "date" else "400"))
        y += h + 12
    left_bottom = y

    # right card: the text
    right, y2, last_box = [], top + 50, None
    right.append(f'<text x="{pad + 400 + 18}" y="{y2}" font-size="13" font-weight="700" fill="{INK}">{escape(heading)}</text>')
    y2 += 26
    for i, para in enumerate(paragraphs):
        lines = wrap_lines(para, 80)
        h = 17 * len(lines)
        if i == len(paragraphs) - 1:
            right.append(f'<rect x="{pad + 400 + 10}" y="{y2 - 15}" width="{W - 2 * pad - 400 - 20}" height="{h + 8}" rx="6" fill="{ORANGE}" fill-opacity="0.18" stroke="{ORANGE}" stroke-width="1.5"/>')
        right.append(text_block(pad + 400 + 18, y2, lines, size=12.5, fill=INK, lh=17))
        y2 += h + 12
    bottom = max(left_bottom, y2) + 6
    card_h = bottom - top
    H = int(bottom + 86)
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
        f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
        f'<text x="{pad}" y="34" font-size="17" font-weight="700" fill="{INK}">One document, two parts</text>',
        f'<rect x="{pad}" y="{top}" width="360" height="{card_h}" rx="10" fill="{INSTR}" fill-opacity="0.07" stroke="{INSTR}" stroke-width="1.8"/>',
        f'<text x="{pad + 14}" y="{top + 24}" font-size="14" font-weight="700" fill="{INSTR}">Metadata</text>',
        f'<text x="{pad + 346}" y="{top + 24}" text-anchor="end" font-size="11.5" fill="{GREY}">the header of the file</text>',
        f'<rect x="{pad + 400}" y="{top}" width="{W - 2 * pad - 400}" height="{card_h}" rx="10" fill="{CTX}" fill-opacity="0.07" stroke="{CTX}" stroke-width="1.8"/>',
        f'<text x="{pad + 414}" y="{top + 24}" font-size="14" font-weight="700" fill="{CTX}">Text</text>',
        f'<text x="{W - pad - 14}" y="{top + 24}" text-anchor="end" font-size="11.5" fill="{GREY}">the body of the file</text>',
    ] + left + right + [
        f'<text x="{pad}" y="{bottom + 28}" font-size="13.5" font-weight="700" fill="#b86a14">An application can filter and rank on this field.</text>',
        f'<text x="{pad + 400}" y="{bottom + 28}" font-size="13.5" font-weight="700" fill="#b86a14">In the text, "latest" reads as current.</text>',
        f'<text x="{pad}" y="{bottom + 62}" font-size="14.5" font-weight="700" fill="{BLUE}">An application can use the date only if it was kept as metadata.</text>',
        "</svg>",
    ]
    return "".join(p)



# --------------------------------------------------------------- notebook 04: the model asks, the application runs the tool -------------------
def toolloop_svg():
    W, H, T = 1000, 600, 20.0
    LIFE = 478  # lifelines end here; the four narration lines sit below
    xs = {"model": 170, "app": 500, "tool": 830}
    green = CTX
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
         f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{BLUE}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">How a model uses a tool: it asks, the application runs it</text>']
    heads = [("model", "The model", "writes text, runs nothing", PALE, BLUE), ("app", "The application", "your code", LIGHT, INK), ("tool", "The tool", "get_indicator, over the data", "#e4f3ea", green)]
    for key, name, sub, fill, col in heads:
        x = xs[key]
        p.append(f'<rect x="{x - 105}" y="56" width="210" height="50" rx="10" fill="{fill}" stroke="{col}" stroke-width="1.8"/>'
                 f'<text x="{x}" y="77" text-anchor="middle" font-size="14.5" font-weight="700" fill="{col}">{name}</text>'
                 f'<text x="{x}" y="96" text-anchor="middle" font-size="11.5" fill="{GREY}">{sub}</text>'
                 f'<line x1="{x}" y1="106" x2="{x}" y2="{LIFE}" stroke="#c9d1de" stroke-width="1.5" stroke-dasharray="5 5"/>')

    def step(n, y, src, dst, label, s, e, col=BLUE):
        x1, x2 = xs[src], xs[dst]
        d = 1 if x2 > x1 else -1
        mid = (x1 + x2) / 2
        w = max(len(label) * 6.6 + 24, 120)
        inner = (f'<line x1="{x1 + 6 * d}" y1="{y}" x2="{x2 - 6 * d}" y2="{y}" stroke="{col}" stroke-width="2.2" marker-end="url(#ar)"/>'
                 f'<rect x="{mid - w / 2}" y="{y - 31}" width="{w}" height="25" rx="6" fill="#ffffff" stroke="{col}" stroke-opacity="0.5"/>'
                 f'<text x="{mid}" y="{y - 14}" text-anchor="middle" font-size="12.5" fill="{INK}">{escape(label)}</text>'
                 f'<circle cx="{x1 + 6 * d - 0}" cy="{y}" r="0"/>')
        p.append(group(inner, s, e, T, 1))

    end = 19.3
    step(1, 150, "app", "model", "question + the tools it may ask for", 0.4, end)
    step(2, 215, "model", "app", "asks: get_indicator(GDP per capita, Rwanda, 2023)", 5.0, end)
    step(3, 280, "app", "tool", "runs it", 9.5, end, green)
    step(4, 335, "tool", "app", "2023: 1,061 · with its source", 11.0, end, green)
    step(5, 400, "app", "model", "hands the result back", 14.3, end)
    step(6, 465, "model", "app", "answer: 1,061 US dollars in 2023 [source]", 16.2, end)
    caps = [
        ("1  ·  The application sends the question, and says which tools the model may ask for", 0, 4.9, 0),
        ("2  ·  The model runs nothing: it replies with a request that names the tool and fills in its details", 4.9, 9.4, 0),
        ("3  ·  The application runs the tool, and the tool returns data", 9.4, 14.2, 0),
        ("4  ·  The application hands the result back, and the model writes its answer from it", 14.2, 20.4, 1),
    ]
    # all four lines are always shown (so a still image reads in full); the current one is highlighted
    for i, (label, s, e, base) in enumerate(caps):
        y = LIFE + 40 + i * 25
        p.append(f'<text x="24" y="{y}" font-size="14" fill="{GREY}">{escape(label)}</text>')
        p.append(group(f'<rect x="20" y="{y - 15}" width="970" height="21" fill="#ffffff"/>'
                       f'<text x="24" y="{y}" font-size="14" font-weight="700" fill="{BLUE}">{escape(label)}</text>'
                       f'<rect x="10" y="{y - 14}" width="5" height="18" rx="2" fill="{BLUE}"/>', s, e, T, base))
    p.append("</svg>")
    return "".join(p)



def workflows_svg():
    """Static picture: a workflow (the order is written in code) beside an agent (the model chooses the next step)."""
    W, H = 1000, 478
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
         f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{BLUE}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">Who decides the next step?</text>']

    def box(x, y, w, h, lines, fill=PALE, stroke=BLUE, bold=False):
        out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
        for i, t in enumerate(lines):
            yy = y + h / 2 + 5 + (i - (len(lines) - 1) / 2) * 17
            weight = 'font-weight="700" ' if bold else ""
        out += f'<text x="{x + w / 2}" y="{yy}" text-anchor="middle" font-size="13.5" {weight}fill="{INK}">{escape(t)}</text>'
        return out

    def arrow(x1, y1, x2, y2, label="", col=BLUE, dy=-8):
        out = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" stroke-width="2.2" marker-end="url(#ar)"/>'
        if label:
            out += f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 + dy}" text-anchor="middle" font-size="11.5" fill="{GREY}">{escape(label)}</text>'
        return out

    # a workflow
    p.append(f'<text x="24" y="72" font-size="15" font-weight="700" fill="{BLUE}">A workflow: you write the order in code</text>')
    xs = [24, 214, 424, 634, 824]
    labels = [["question"], ["1 · search", "the documents"], ["2 · look up", "the four series"], ["3 · the model", "writes the answer"], ["answer"]]
    for x, l in zip(xs, labels):
        p.append(box(x, 90, 152, 56, l, fill=LIGHT if l == ["question"] or l == ["answer"] else PALE))
    for a, b in zip(xs, xs[1:]):
        p.append(arrow(a + 152, 118, b - 2, 118))
    p.append(f'<text x="24" y="172" font-size="13" fill="{GREY}">The same calls, in the same order, on every run. The model is called once, to write.</text>')
    p.append(f'<line x1="24" y1="196" x2="976" y2="196" stroke="#e1e6ee"/>')

    # an agent
    p.append(f'<text x="24" y="228" font-size="15" font-weight="700" fill="{BLUE}">An agent: the model chooses the next step, after seeing what came back</text>')
    p.append(box(24, 296, 130, 56, ["question"], fill=LIGHT))
    p.append(box(214, 276, 170, 96, ["the model", "decides what", "to do next"], fill=INK, stroke=INK))
    p[-1] = p[-1].replace(f'fill="{INK}">', 'fill="#ffffff">').replace(f'<rect x="214" y="276" width="170" height="96" rx="9" fill="#ffffff"', f'<rect x="214" y="276" width="170" height="96" rx="9" fill="{INK}"')
    p.append(box(560, 252, 200, 44, ["search_documents"], fill="#e4f3ea", stroke=CTX))
    p.append(box(560, 352, 200, 44, ["get_indicator"], fill="#e4f3ea", stroke=CTX))
    p.append(arrow(154, 324, 212, 324))
    p.append(arrow(384, 300, 558, 276, "asks", dy=-9))
    p.append(arrow(558, 290, 384, 314, "result comes back", dy=18))
    p.append(arrow(384, 352, 558, 370, "or asks", dy=-9))
    p.append(arrow(558, 384, 384, 366, "result comes back", dy=19))
    p.append(box(830, 296, 146, 56, ["answer"], fill=LIGHT))
    p.append(f'<line x1="384" y1="324" x2="400" y2="324" stroke="none"/>')
    p.append(f'<path d="M299,372 L299,426 L903,426 L903,354" fill="none" stroke="{BLUE}" stroke-width="2.2" marker-end="url(#ar)"/>')
    p.append(f'<text x="600" y="419" text-anchor="middle" font-size="11.5" fill="{GREY}">when it has enough, it answers in text</text>')
    p.append(f'<text x="24" y="456" font-size="13" fill="{GREY}">The calls vary with the question. A step limit stops a loop that never ends; a trace lists every call and result.</text>')
    p.append("</svg>")
    return "".join(p)


def claimtag_svg():
    """Static picture: the parts of a claim tag, and how the verifier compares the number with its record."""
    W, H, CW = 1000, 400, 9.0   # CW: width of one character of the code font at 15px
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" style="max-width:100%;height:auto" font-family="{SANS}">',
         f'<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="{BLUE}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" rx="12" fill="#ffffff" stroke="#e1e6ee"/>',
         f'<text x="24" y="34" font-size="17" font-weight="700" fill="{INK}">A claim tag: a number that says where it came from</text>']
    parts = [("<claim ", None, ""), ('id="IT.NET.USER.ZS:KEN:2024"', "#2f5f9e", "id"), (" ", None, ""), ('policy="rounded" decimals="1"', "#d9822b", "policy"),
             (">", None, ""), ("35.0", "#2f8f5b", "num"), ("</claim>", None, "")]
    x, y, centres = 40, 92, {}
    for text, col, key in parts:
        w = len(text) * CW
        if col:
            p.append(f'<rect x="{x - 1}" y="{y - 19}" width="{w + 2}" height="28" rx="5" fill="{col}" fill-opacity="0.16" stroke="{col}"/>')
            centres[key] = x + w / 2
        p.append(f'<text x="{x}" y="{y}" font-size="15" font-family="{MONO}" fill="{INK}" xml:space="preserve">{escape(text)}</text>')
        x += w
    notes = {"id": ["which record it came from:", "indicator : country : year"], "policy": ["how to compare it:", "rounded to 1 decimal"], "num": ["the number a", "reader sees"]}
    cols = {"id": "#2f5f9e", "policy": "#d9822b", "num": "#2f8f5b"}
    for key, lines in notes.items():
        cx = centres[key]
        p.append(f'<line x1="{cx}" y1="{y + 11}" x2="{cx}" y2="140" stroke="{cols[key]}" stroke-width="1.6"/>')
        for i, t in enumerate(lines):
            p.append(f'<text x="{cx}" y="{158 + i * 17}" text-anchor="middle" font-size="12.5" fill="{cols[key]}" font-weight="{700 if i == 0 else 400}">{escape(t)}</text>')
    p.append(f'<text x="24" y="238" font-size="14.5" font-weight="700" fill="{BLUE}">What the verifier does with it</text>')
    boxes = [(24, "The record holds", "34.9762", PALE, BLUE), (366, "Rounded to 1 decimal", "35.0", PALE, BLUE), (708, "Equals the number shown", "verified", "#e4f3ea", "#2f8f5b")]
    for bx, a, b, fill, col in boxes:
        p.append(f'<rect x="{bx}" y="256" width="268" height="84" rx="10" fill="{fill}" stroke="{col}" stroke-width="1.8"/>'
                 f'<text x="{bx + 134}" y="286" text-anchor="middle" font-size="13" fill="{GREY}">{escape(a)}</text>'
                 f'<text x="{bx + 134}" y="320" text-anchor="middle" font-size="24" font-weight="700" fill="{col}">{escape(b)}</text>')
    p.append(f'<line x1="294" y1="298" x2="364" y2="298" stroke="{BLUE}" stroke-width="2.2" marker-end="url(#ar)"/>')
    p.append(f'<line x1="636" y1="298" x2="706" y2="298" stroke="{BLUE}" stroke-width="2.2" marker-end="url(#ar)"/>')
    p.append(f'<text x="24" y="372" font-size="13" fill="{GREY}">If the rounded record differed from the number shown, the mark would be flagged. With no tag at all, the number is unmarked.</text>')
    p.append("</svg>")
    return "".join(p)


ALT = {
    "chunking": "Animation: the same Kenya brief cut two ways. Fixed 400-character windows end on a stranded heading and mix two sections; section chunks keep each heading with its paragraph.",
    "embeddings": "Animation: a chunk of text becomes a list of numbers, then a dot on a map where similar meaning sits close together; a question becomes a star and the two closest chunks are retrieved.",
    "prompt": "Animation: instructions set by the developer, context supplied by the application and a question asked by the user are combined into one prompt, which goes to the model and produces an answer that cites its source.",
    "before_after": "Diagram: the same question without context gets an answer with no source, no date and a blend of two measures; with the four country briefs as context it gets a dated answer that cites its source.",
    "metadata": "Diagram: one document in two parts. Its metadata, the header of the file, holds the date 2016-06-01; its text sounds current and ends 'The latest available figures are for 2015.' An application can filter and rank on the date field, but it does not read the text for age.",
    "toolloop": "Animation: the application sends the model the question and the list of tools; the model replies with a request for get_indicator with GDP per capita, Rwanda and 2023; the application runs the tool, which returns 1,061 with its source; the application hands that back and the model answers that it was 1,061 US dollars in 2023.",
    "workflows": "Diagram: in a workflow the order of the calls is written in code: search the documents, look up the four series, the model writes the answer, the same on every run. In an agent the model decides the next step: it asks for search_documents or get_indicator, gets the result back, and answers in text when it has enough; a step limit stops a loop that never ends and a trace lists every call.",
    "claimtag": "Diagram: a claim tag wraps a number and carries the claim_id, which names the indicator, the country and the year, and a policy that says how to compare, here rounded to one decimal. The verifier takes the record, 34.9762, rounds it to one decimal, gets 35.0, finds it equal to the number shown, and marks it verified.",
    "desk": "Diagram: the four briefs of this notebook, about 1,024 tokens, fit inside the model's context window, while a real collection of thousands of documents cannot, so the application must choose what the model reads.",
}


if __name__ == "__main__":
    Path("chunking.svg").write_text(chunking_svg()[0])
    Path("embeddings.svg").write_text(embeddings_svg())
