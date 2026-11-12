"""Animated SVG explainers for the session 02 notebooks.

build_notebooks.py draws each picture at build time and embeds it in a Markdown cell as an image, so the
notebooks import nothing from this file. The animations are SMIL, which is part of SVG: they loop, need no
packages, and show every element where animation is not supported.

The texts and codes in the pictures come from the recorded files in ../assets, and the counts are computed
from those files by build_notebooks.py.

    python explainers.py OUT_DIR    # writes every picture as an .svg file, for review
"""

from html import escape

INK, MUTED, FAINT, LINE, SOFT = "#16191D", "#565F68", "#8A939C", "#E2E5E8", "#EEF2F5"
SMALL, SMALL_SOFT, SMALL_TEXT = "#2A78D6", "#E3EEFA", "#1F5FAE"
LARGE, LARGE_SOFT, LARGE_TEXT = "#EB6834", "#FBE6DC", "#A9431B"
OK, OK_SOFT, BAD, BAD_SOFT = "#1A7F37", "#DCF2E3", "#B42318", "#FBE4E1"
SANS = "Helvetica, Arial, sans-serif"
FADE = 0.4


def anim(s, e, total, base=1):
    """Opacity of an element that is visible from s to e seconds of a loop of `total` seconds."""
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


def show(inner, s, e, total):
    return f"<g {anim(s, e, total)}{inner}</g>"


def opening(w, h, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="max-width:100%;height:auto" font-family="{SANS}">',
        f'<defs><marker id="ar" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" refX="10" refY="5.5" orient="auto"><path d="M0,0 L11,5.5 L0,11 z" fill="{MUTED}"/></marker></defs>',
        f'<rect width="{w}" height="{h}" rx="12" fill="#ffffff" stroke="{LINE}"/>',
        f'<text x="24" y="36" font-size="17" font-weight="700" fill="{INK}">{escape(title)}</text>',
    ]


def text(x, y, s, size=13, fill=INK, weight=400, anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>'


def lines(x, y, rows, size=13, fill=INK, weight=400, anchor="start", lh=None):
    lh = lh or size * 1.35
    return "".join(text(x, y + i * lh, r, size, fill, weight, anchor) for i, r in enumerate(rows))


def box(x, y, w, h, fill="#ffffff", stroke=LINE, sw=1.5, rx=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="1.8" marker-end="url(#ar)"/>'


def chip_width(s, size=12):
    return len(s) * size * 0.58 + 16


def chip(x, y, s, fill=SOFT, ink=INK, size=12):
    """A rounded label whose left edge is at x and whose text baseline is at y."""
    w = chip_width(s, size)
    return (f'<rect x="{x}" y="{y - size - 4}" width="{w:.0f}" height="{size + 10}" rx="{(size + 10) / 2}" fill="{fill}"/>'
            + text(x + 8, y, s, size, ink, 600))


def cut(s, n):
    """A text of at most n characters. A longer text ends with an ellipsis."""
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def wrap(s, cols):
    out, cur = [], ""
    for word in s.split():
        if cur and len(cur) + 1 + len(word) > cols:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return out + [cur] if cur else out


def stages_svg(title, stages, legend=True):
    """A row of stages. Each stage is (kind, lines of the name, caption); kind is data, large or small."""
    n = len(stages)
    w, gap, x0 = 960, 22, 24
    bw = (w - 2 * x0 - gap * (n - 1)) / n
    y, bh, total = 62, 78, 2.0 * n + 1
    captions = [wrap(caption, int(bw / 6.6)) for _, _, caption in stages]
    h = y + bh + 14 + 16 * max(len(c) for c in captions) + (40 if legend else 10)
    fills = {"data": (SOFT, FAINT, INK), "large": (LARGE_SOFT, LARGE, LARGE_TEXT), "small": (SMALL_SOFT, SMALL, SMALL_TEXT)}
    p = opening(w, h, title)
    for i, (kind, name, caption) in enumerate(stages):
        x = x0 + i * (bw + gap)
        fill, stroke, ink = fills[kind]
        p.append(box(x, y, bw, bh, fill, stroke))
        p.append(show(box(x - 3, y - 3, bw + 6, bh + 6, "none", stroke, 3.5, 12), 2.0 * i + 0.3, 2.0 * i + 2.0, total))
        p.append(lines(x + bw / 2, y + (38 if len(name) == 2 else 46), name, 14, ink, 700, "middle", 19))
        p.append(lines(x + bw / 2, y + bh + 22, captions[i], 12, MUTED, 400, "middle", 16))
        if i:
            p.append(arrow(x - gap + 2, y + bh / 2, x - 3, y + bh / 2))
    if legend:
        ly = h - 22
        p.append(f'<rect x="24" y="{ly - 11}" width="13" height="13" rx="3" fill="{LARGE_SOFT}" stroke="{LARGE}"/>' + text(44, ly, "Large models", 12, MUTED))
        p.append(f'<rect x="150" y="{ly - 11}" width="13" height="13" rx="3" fill="{SMALL_SOFT}" stroke="{SMALL}"/>' + text(170, ly, "Small model", 12, MUTED))
    p.append("</svg>")
    return "".join(p)


def rewrite_svg(title, code, name, answers, styles):
    """One job title with its code, and the answers that the large model writes from it, one for each style."""
    w, total = 960, 13.0
    rows = [wrap(a, 46) for a in answers]
    card_h = [38 + 19 * len(r) for r in rows]
    h = max(300, 78 + sum(card_h) + 14 * (len(rows) - 1) + 28)
    p = opening(w, h, "The large model rewrites a job title as answers to a survey question")
    cy = 66 + (h - 66 - 28) / 2
    p.append(box(24, cy - 62, 252, 124, SOFT, FAINT))
    p.append(text(40, cy - 36, "Job title on the job board", 12, MUTED))
    p.append(lines(40, cy - 10, wrap(title, 26), 16, INK, 700, lh=21))
    p.append(chip(40, cy + 44, f"{code} · {name}"[:34], "#ffffff", INK))
    p.append(arrow(280, cy, 322, cy))
    p.append(box(326, cy - 34, 150, 68, LARGE_SOFT, LARGE))
    p.append(lines(401, cy - 4, ["Large model", "writes"], 14, LARGE_TEXT, 700, "middle", 19))
    y = 66
    for i, (r, style, ch) in enumerate(zip(rows, styles, card_h)):
        s = 1.2 + 3.0 * i
        p.append(show(arrow(480, cy, 532, y + ch / 2), s, total - 0.8, total))
        card = box(536, y, 400, ch, "#ffffff", LINE) + text(552, y + 22, style, 12, MUTED)
        card += lines(552, y + 44, r, 14.5, INK, 600, lh=19)
        card += chip(936 - 12 - chip_width(code), y + 24, code, SOFT, INK)
        p.append(show(card, s + 0.5, total - 0.8, total))
        y += ch + 14
    p.append(text(24, h - 16, "Each answer takes the code of its job title.", 12.5, MUTED))
    p.append("</svg>")
    return "".join(p)


def check_svg(rows):
    """Each row: (answer, job title, code of the title, code from the second model or None)."""
    w, rh, y0, total = 960, 64, 96, 4.0 * len(rows) + 2
    h = y0 + rh * len(rows) + 44
    p = opening(w, h, "A second model labels each answer on its own")
    for x, label in [(24, "Answer written by the first model"), (470, "Code of the job title"), (640, "Code from the second model"), (862, "Result")]:
        p.append(text(x, 72, label, 12, MUTED, 600))
    p.append(f'<line x1="24" y1="82" x2="936" y2="82" stroke="{LINE}"/>')
    for i, (answer, title, code, second) in enumerate(rows):
        y, s = y0 + i * rh, 0.8 + 4.0 * i
        keep = second == code
        p.append(text(24, y + 20, f"“{answer}”"[:58], 14.5, INK, 600))
        p.append(text(24, y + 40, f"from the job title “{title}”"[:62], 12, MUTED))
        p.append(chip(470, y + 26, code, SOFT, INK, 13))
        p.append(show(chip(640, y + 26, second or "none of the listed groups", LARGE_SOFT, LARGE_TEXT, 13), s, total - 0.6, total))
        verdict = chip(862, y + 26, "kept" if keep else "removed", OK_SOFT if keep else BAD_SOFT, OK if keep else BAD, 13)
        p.append(show(verdict, s + 1.6, total - 0.6, total))
        if not keep:
            p.append(show(f'<line x1="24" y1="{y + 15}" x2="{24 + min(len(answer) + 2, 58) * 6.85:.0f}" y2="{y + 15}" stroke="{BAD}" stroke-width="1.6"/>', s + 1.6, total - 0.6, total))
        p.append(f'<line x1="24" y1="{y + rh - 8}" x2="936" y2="{y + rh - 8}" stroke="{LINE}"/>')
    p.append(text(24, h - 16, "An answer is kept when the second model returns the code of the job title.", 12.5, MUTED))
    p.append("</svg>")
    return "".join(p)


def loop_svg(title, nodes):
    """Four steps on a cycle. Each node is (kind, lines of the name)."""
    w, h, total = 960, 300, 10.0
    fills = {"data": (SOFT, FAINT, INK), "large": (LARGE_SOFT, LARGE, LARGE_TEXT), "small": (SMALL_SOFT, SMALL, SMALL_TEXT)}
    pos = [(150, 70), (590, 70), (590, 196), (150, 196)]
    bw, bh = 220, 70
    p = opening(w, h, title)
    for i, ((kind, name), (x, y)) in enumerate(zip(nodes, pos)):
        fill, stroke, ink = fills[kind]
        p.append(box(x, y, bw, bh, fill, stroke))
        p.append(show(box(x - 3, y - 3, bw + 6, bh + 6, "none", stroke, 3.5, 12), 2.5 * i + 0.2, 2.5 * i + 2.3, total))
        p.append(lines(x + bw / 2, y + (32 if len(name) == 2 else 41), name, 14, ink, 700, "middle", 19))
    p.append(arrow(374, 105, 586, 105))
    p.append(arrow(700, 144, 700, 192))
    p.append(arrow(586, 231, 374, 231))
    p.append(arrow(260, 192, 260, 144))
    p.append("</svg>")
    return "".join(p)


def request_svg(topics, sentences, per_call, requests):
    """One request to the large model: the topics it names, and some of the sentences that came back with their labels."""
    w, total = 960, 4.0 + 2.6 * len(sentences)
    rows = [wrap(s, 68) for s, _ in sentences]
    card_h = [36 + 18 * len(r) for r in rows]
    h = max(70 + 34 + 27 * len(topics) + 40, 70 + sum(card_h) + 12 * (len(rows) - 1) + 44)
    p = opening(w, h, "The large model writes sentences about the topics named in a request")
    p.append(text(24, 72, "Topics named in the request", 12, MUTED, 600))
    for i, t in enumerate(topics):
        p.append(chip(24, 100 + i * 27, t[:34], SOFT, INK, 11.5))
    cy = 100 + 27 * len(topics) / 2 - 14
    p.append(arrow(262, cy, 296, cy))
    p.append(box(300, cy - 34, 130, 68, LARGE_SOFT, LARGE))
    p.append(lines(365, cy - 4, ["Large model", "writes"], 14, LARGE_TEXT, 700, "middle", 19))
    y = 62
    for i, ((_, labels), r, ch) in enumerate(zip(sentences, rows, card_h)):
        s = 1.0 + 2.6 * i
        card = box(470, y, 466, ch, "#ffffff", LINE) + lines(484, y + 21, r, 12.5, INK, 400, lh=18)
        x = 484
        for label in labels:
            card += chip(x, y + ch - 10, label[:34], SMALL_SOFT, SMALL_TEXT, 11)
            x += chip_width(label[:34], 11) + 6
        p.append(show(arrow(434, cy, 466, y + ch / 2), s, total - 0.8, total))
        p.append(show(card, s + 0.4, total - 0.8, total))
        y += ch + 12
    p.append(text(470, y + 12, f"{len(sentences)} of the {per_call} sentences of this request. The first set has {requests:,} requests.", 12.5, MUTED))
    p.append("</svg>")
    return "".join(p)


def hierarchy_svg(levels):
    """The four levels of the classification for one unit group. Each level is (code, name, count of groups at the level, name of the level)."""
    w, h, total = 960, 330, 12.0
    p = opening(w, h, "A unit group and the three levels above it")
    for i, (code, name, count, level) in enumerate(levels):
        x, y, last = 24 + 56 * i, 60 + 62 * i, i == len(levels) - 1
        fill, stroke, ink = (SMALL_SOFT, SMALL, SMALL_TEXT) if last else (SOFT, FAINT, INK)
        card = box(x, y, 520, 48, fill, stroke) + chip(x + 14, y + 30, code, "#ffffff", ink, 13)
        card += text(x + 30 + chip_width(code, 13), y + 30, name, 14.5, ink, 700)
        card += text(x + 540, y + 30, f"one of {count} {level}", 13, MUTED)
        if i:
            card += f'<path d="M{x - 28},{y - 14} V{y + 24} H{x - 4}" fill="none" stroke="{MUTED}" stroke-width="1.8" marker-end="url(#ar)"/>'
        p.append(show(card, 0.4 + 2.0 * i, total, total))
    p.append(show(text(24 + 56 * 3, h - 16, "The small model chooses among the unit groups.", 12.5, MUTED), 0.4 + 2.0 * len(levels), total, total))
    p.append("</svg>")
    return "".join(p)


def predict_svg(answer, tokens, rows, probs):
    """The path of one answer through the small model. rows: (code, name, probability) of the best groups;
    probs: the probability of every group, in the order of the codes."""
    w, h, total = 960, 360, 14.0
    p = opening(w, h, "From an answer to five suggested groups")
    p.append(text(24, 70, "Answer in the survey", 12, MUTED, 600))
    p.append(show(box(24, 82, 250, 52, "#ffffff", LINE) + lines(38, 104, wrap(f"“{answer}”", 32), 14, INK, 600, lh=18), 0.3, total, total))
    p.append(text(24, 168, "Tokens", 12, MUTED, 600))
    x, y, chips = 24, 196, ""
    for t in tokens:
        cw = chip_width(t, 12)
        if x + cw > 276:
            x, y = 24, y + 30
        chips += chip(x, y, t, SOFT, INK, 12)
        x += cw + 5
    p.append(show(chips, 2.0, total, total))
    p.append(show(arrow(282, 190, 318, 190), 3.6, total, total))
    p.append(box(322, 150, 150, 80, SMALL_SOFT, SMALL))
    p.append(lines(397, 185, ["Small model"], 15, SMALL_TEXT, 700, "middle"))
    p.append(text(397, 206, "DeBERTa-v3-small", 11.5, SMALL_TEXT, 400, "middle"))
    p.append(show(box(319, 147, 156, 86, "none", SMALL, 3.5, 12), 3.6, 6.0, total))
    p.append(show(arrow(476, 190, 508, 190), 5.6, total, total))
    probs = [float(v) for v in probs]
    n, top = len(probs), max(probs) or 1
    best = set(sorted(range(n), key=lambda i: -probs[i])[: len(rows)])
    bars = f'<line x1="514" y1="232.5" x2="664" y2="232.5" stroke="{FAINT}" stroke-width="1"/>'
    for i in sorted(range(n), key=lambda i: i in best):   # the best groups are drawn last, on top
        bh = 96 * probs[i] / top
        if bh >= 1 or i in best:
            bx = 516 + 146 * i / max(n - 1, 1)
            bars += f'<line x1="{bx:.1f}" y1="232" x2="{bx:.1f}" y2="{232 - max(bh, 2):.1f}" stroke="{SMALL if i in best else MUTED}" stroke-width="{2.6 if i in best else 1}"/>'
    p.append(show(bars + lines(516, 252, ["one probability for each", f"of the {n} groups"], 11.5, MUTED, lh=15), 6.0, total, total))
    p.append(show(arrow(668, 190, 692, 190), 8.0, total, total))
    p.append(text(696, 70, "The five most probable groups", 12, MUTED, 600))
    for i, (code, name, prob) in enumerate(rows):
        y = 84 + i * 50
        card = box(696, y, 240, 42, "#ffffff", LINE) + chip(706, y + 27, code, SMALL_SOFT, SMALL_TEXT, 12)
        card += text(758, y + 19, cut(name, 27), 12, INK, 600) + text(758, y + 35, f"{prob:.0%}", 12, MUTED)
        p.append(show(card, 8.4 + 0.7 * i, total, total))
    p.append(show(text(24, h - 16, "A coder chooses among the suggestions or enters another code.", 12.5, MUTED), 12.0, total, total))
    p.append("</svg>")
    return "".join(p)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    import build_notebooks as b

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, svg in b.pictures().items():
        (out / f"{name}.svg").write_text(svg, encoding="utf-8")
        print(name, len(svg))
