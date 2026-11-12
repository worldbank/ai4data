"""Pictures drawn from the notebooks' own results.

Each function takes values computed in a notebook cell and displays an SVG picture. The colours follow the
slides of the session: blue for the small model, orange for a large model, green and red for a check.

    suggestions(response, rows, truth)       the groups that the model suggests for one response, as bars
    kept_removed(rows)                        answers kept and removed by the second model's check, by style
    result_bars(title, rows, note)            one score for several models or training sets
    threshold_chart(grid, values, best)       the score on the threshold file at each threshold
    report_view(title, sentences, tops, ...)  the sentences of one abstract, then the mean score of each topic
    answer_path(answer, tokens, rows, probs)  one answer, its tokens, the probability of each group, five suggestions
"""

from html import escape

from IPython.display import HTML, display

from explainers import (BAD, BAD_SOFT, FAINT, INK, LARGE, LARGE_SOFT, LARGE_TEXT, LINE, MUTED, OK, OK_SOFT, SMALL,
                        SMALL_SOFT, SMALL_TEXT, SOFT, anim, box, chip, chip_width, cut, lines, opening, predict_svg, show, text, wrap)

FILL = {"small": SMALL, "large": LARGE, "muted": FAINT}


def grow(x, y, w, h, fill, begin, rx=4):
    """A bar that grows from the left when the picture loads. Without animation support it is drawn at full width."""
    w = max(w, 2)
    return (f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{h}" rx="{rx}" fill="{fill}">'
            f'<animate attributeName="width" from="0" to="{w:.1f}" dur="0.6s" begin="{begin:.2f}s" fill="freeze"/></rect>')


def put(parts):
    display(HTML("".join(parts) + "</svg>"))


def suggestions(response, rows, truth=None, note=""):
    """rows: (code, name of the group, probability), best first. `truth` is the code that the test file gives."""
    w, top, step = 760, 78, 34
    h = top + step * len(rows) + (40 if truth or note else 14)
    p = opening(w, h, f"“{cut(response, 74)}”")
    p.append(text(24, 58, "Groups suggested by the small model and the probability of each", 12.5, MUTED))
    scale = max(r[2] for r in rows) or 1
    for i, (code, name, prob) in enumerate(rows):
        y = top + i * step
        hit = truth is not None and code == truth
        p.append(chip(24, y + 17, code, OK_SOFT if hit else SMALL_SOFT, OK if hit else SMALL_TEXT, 12.5))
        name_rows = wrap(name, 44)[:2]   # a long group name takes two lines
        if len(name_rows) == 1:
            p.append(text(82, y + 17, name, 13, INK, 600 if hit else 400))
        else:
            p.append(lines(82, y + 10, name_rows, 12, INK, 600 if hit else 400, lh=14))
        p.append(grow(420, y + 4, 250 * prob / scale, 18, OK if hit else SMALL, 0.15 * i))
        p.append(text(420 + 250 * prob / scale + 8, y + 18, f"{prob:.0%}", 12.5, INK, 600))
    if truth is not None:
        found = any(code == truth for code, _, _ in rows)
        p.append(text(24, h - 16, f"Code in the test file: {truth}. " + ("It is among the suggestions (green)." if found else "It is absent from the suggestions."), 12.5, MUTED))
    elif note:
        p.append(text(24, h - 16, note, 12.5, MUTED))
    put(p)


def kept_removed(rows, title="Answers kept and removed by the second model's check"):
    """rows: (name of the style, number kept, number removed)."""
    w, top, step = 760, 70, 46
    h = top + step * len(rows) + 34
    p = opening(w, h, title)
    most = max(k + r for _, k, r in rows)
    for i, (name, kept, removed) in enumerate(rows):
        y = top + i * step
        wk, wr = 250 * kept / most, 250 * removed / most
        p.append(text(24, y + 16, cut(name, 28), 13, INK, 600))
        p.append(grow(230, y, wk, 22, OK, 0.2 * i))
        p.append(f'<rect x="{230 + wk + 2:.1f}" y="{y}" width="{max(wr, 2):.1f}" height="22" rx="4" fill="{BAD_SOFT}" stroke="{BAD}" stroke-width="1" {anim(0.6 + 0.2 * i, 99, 99)}</rect>')
        p.append(text(230 + wk + wr + 12, y + 16, f"{kept:,} kept, {removed:,} removed ({kept / (kept + removed):.0%} kept)", 12.5, MUTED))
    y = top + step * len(rows) + 8
    p.append(box(24, y - 11, 12, 12, OK, OK, 0, 3) + text(42, y, "kept", 12, MUTED))
    p.append(box(92, y - 11, 12, 12, BAD_SOFT, BAD, 1, 3) + text(110, y, "removed", 12, MUTED))
    put(p)


def result_bars(title, rows, note="", top_value=None, fmt="{:.3f}", label_chars=42):
    """rows: (label, value, kind), where kind is small, large or muted. A label can hold two lines separated by a line break.
    `label_chars` is the room for a label; a longer label leaves a shorter bar."""
    w, top, step = 760, 62, 44
    x0 = 24 + round(7.29 * label_chars)
    full = 660 - x0
    h = top + step * len(rows) + (34 if note else 10)
    p = opening(w, h, title)
    scale = top_value or max(v for _, v, _ in rows)
    for i, (label, value, kind) in enumerate(rows):
        y = top + i * step
        first, _, second = label.partition("\n")
        p.append(text(24, y + (10 if second else 17), cut(first, label_chars), 13, INK, 600))
        if second:
            p.append(text(24, y + 27, second, 11.5, MUTED))
        p.append(grow(x0, y + 2, full * value / scale, 22, FILL[kind], 0.15 * i))
        p.append(text(x0 + full * value / scale + 8, y + 18, fmt.format(value), 13, INK, 700))
    if note:
        p.append(text(24, h - 14, note, 12, MUTED))
    put(p)


def threshold_chart(grid, values, best, title="Score on the threshold file at each threshold"):
    w, h, left, right, top, bottom = 760, 330, 70, 730, 66, 270
    p = opening(w, h, title)
    hi = max(values) * 1.12
    xs = [left + i * (right - left) / (len(grid) - 1) for i in range(len(grid))]
    ys = [bottom - (bottom - top) * v / hi for v in values]
    for tick in range(0, int(hi * 10) + 1):
        y = bottom - (bottom - top) * (tick / 10) / hi
        p.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{LINE}"/>' + text(left - 10, y + 4, f"{tick / 10:.1f}", 12, MUTED, anchor="end"))
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    p.append(f'<path d="{path}" fill="none" stroke="{SMALL}" stroke-width="2"/>')
    for g, v, x, y in zip(grid, values, xs, ys):
        chosen = g == best
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{7 if chosen else 4}" fill="{SMALL}" stroke="#ffffff" stroke-width="2"/>')
        p.append(text(x, bottom + 20, f"{g:g}", 12, INK if chosen else MUTED, 700 if chosen else 400, "middle"))
        if chosen:
            p.append(text(x, y - 14, f"{v:.3f}", 13, INK, 700, "middle"))
    p.append(text((left + right) / 2, h - 14, "Threshold: the lowest mean probability at which a topic is assigned", 12.5, MUTED, anchor="middle"))
    p.append(text(24, 56, "Micro F1", 12.5, MUTED))
    put(p)


def report_view(title, sentences, tops, means, threshold, catalogue):
    """One abstract. sentences: its sentences; tops: (topic, probability) of the best topic of each sentence;
    means: (topic, mean probability) of the best topics of the abstract; catalogue: the topics in the catalogue."""
    w, x2 = 980, 610
    rows = [wrap(s if len(s) < 230 else s[:227] + "...", 88) for s in sentences]
    hs = [22 + 16 * len(r) + 24 for r in rows]
    total = 1.0 + 1.1 * len(sentences) + 0.6 * len(means) + 6
    left_h = 96 + sum(hs) + 8 * len(hs)
    h = max(left_h, 96 + 38 * len(means) + 110) + 16
    p = opening(w, h, cut(title, 98))
    p.append(text(24, 62, "Sentences of the abstract and the most probable topic of each", 12.5, MUTED, 600))
    p.append(text(x2, 62, "Mean over the sentences", 12.5, MUTED, 600))
    y = 78
    for i, (r, hh, (topic, prob)) in enumerate(zip(rows, hs, tops)):
        s = 0.6 + 1.1 * i
        card = box(24, y, 560, hh, "#ffffff", LINE) + lines(36, y + 20, r, 12, INK, 400, lh=16)
        card += chip(36, y + hh - 9, f"{cut(topic, 40)}  {prob:.2f}", SMALL_SOFT, SMALL_TEXT, 11)
        p.append(show(card, s, total, total))
        y += hh + 8
    t0 = 0.6 + 1.1 * len(sentences)
    scale, bar_w = max(m for _, m in means), 190
    tx = x2 + bar_w * threshold / scale
    for i, (topic, mean) in enumerate(means):
        yy = 80 + i * 38
        assigned, listed = mean >= threshold, topic in catalogue
        row = text(x2, yy + 12, cut(topic, 40), 12.5, INK, 600 if assigned else 400)
        row += f'<rect x="{x2}" y="{yy + 18}" width="{max(bar_w * mean / scale, 2):.1f}" height="12" rx="4" fill="{SMALL if assigned else FAINT}"/>'
        row += text(x2 + bar_w + 12, yy + 29, f"{mean:.2f}", 12, INK, 600)
        if listed:
            row += chip(x2 + bar_w + 52, yy + 29, "in the catalogue", OK_SOFT, OK, 10.5)
        p.append(show(row, t0 + 0.6 * i, total, total))
    yb = 80 + 38 * len(means)
    # The threshold is a tick across each bar, so that it crosses no topic name.
    ticks = "".join(f'<line x1="{tx:.1f}" y1="{94 + i * 38}" x2="{tx:.1f}" y2="{114 + i * 38}" stroke="{INK}" stroke-width="1.4"/>' for i in range(len(means)))
    p.append(show(ticks + text(x2, yb + 14, f"The black mark on a bar shows the threshold of {threshold:g}.", 12, MUTED), t0 + 0.6 * len(means), total, total))
    missed = [c for c in catalogue if c not in [t for t, _ in means]]
    note = ["Blue bars are at or above the threshold.", "The model assigns these topics to the report."]
    if missed:
        note += ["Catalogue topics outside this list:"] + wrap(", ".join(missed), 54)
    p.append(lines(x2, yb + 34, note, 12, MUTED, lh=17))
    put(p)


def answer_path(answer, tokens, rows, probs):
    """One answer on its way through the small model. rows: (code, name, probability) of the best groups;
    probs: the probability of every group, in the order of the codes."""
    display(HTML(predict_svg(answer, tokens, rows, probs)))
