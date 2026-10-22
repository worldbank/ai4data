#!/usr/bin/env python3
"""Build the standalone HTML pages of the training series for the docs site.

Reads the worked-example documents of session 01 and writes a page that shows each
document with its metadata, its role in the session, and the chunks that notebook 02
makes of it, each labelled with the citation the application writes when it uses the
chunk. The site build runs this script before Docusaurus, and the output lands under
website/static so that it is served as is.

    python3 scripts/docs/build_training_pages.py                 # website/static/training/session-01/documents.html
    python3 scripts/docs/build_training_pages.py --artifact OUT  # body-only page with embedded logos, for a Claude artifact

No dependencies beyond the standard library: the documents use headings, paragraphs
and bullet lists only, and the chunking mirrors workshop.split_sections (one chunk per
Markdown section, headings kept, "intro" for text before the first section).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SESSION = REPO / "training" / "session-01-building-ai-applications"
DOCS = SESSION / "assets" / "documents"
SITE_OUT = REPO / "website" / "static" / "training" / "session-01" / "documents.html"
LOGO = REPO / "website" / "static" / "img" / "program-logo.png"
LOGO_DARK = REPO / "website" / "static" / "img" / "program-logo-dark.png"

ROLES = [
    ("evidence", "Relevant evidence",
     "The four country briefs. Their figures are rounded from the indicator data, and a correct answer to the main question cites them.",
     ["kenya-digital", "rwanda-digital", "kenya-economy", "rwanda-economy"]),
    ("definition", "Definition",
     "A note on the definition of the internet-users indicator and on the comparability of its values across years.",
     ["internet-indicator"]),
    ("outdated", "Outdated",
     "A 2016 summary with figures for 2015 and expectations that later data did not confirm. It outranks the current briefs for a question about the latest figures, and its date in the metadata removes it.",
     ["regional-2016"]),
    ("distractor", "Distractors",
     "Two notes on related topics without the requested indicators. The coffee note is the retrieval failure case; the mobile-money note is a plausible wrong passage.",
     ["coffee", "mobile-money"]),
]
NOTE = {
    "kenya-digital": "Relevant evidence for internet use and mobile subscriptions in Kenya.",
    "rwanda-digital": "Relevant evidence for internet use and mobile subscriptions in Rwanda.",
    "kenya-economy": "Relevant evidence for GDP per capita and growth in Kenya.",
    "rwanda-economy": "Relevant evidence for GDP per capita and growth in Rwanda.",
    "internet-indicator": "Definition of the indicator; the source for the question on its metadata.",
    "regional-2016": "Outdated figures; shows why dates in metadata matter.",
    "coffee": "Irrelevant to the question; shows a retrieval failure.",
    "mobile-money": "Related topic without the requested indicators; a plausible distractor.",
}
META_KEYS = ["id", "type", "country", "date", "source", "note"]


def read_document(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    meta = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        meta[key.strip()] = value
    return {"metadata": meta, "content": text[m.end():].strip()}


def split_sections(doc: dict) -> list[dict]:
    """One chunk per section, as MarkdownHeaderTextSplitter does with '#' and '##' and headings kept."""
    chunks, current, section = [], [], None
    has_text = False  # a heading with no text before the next heading is carried into the next chunk
    for line in doc["content"].splitlines():
        if line.startswith("## ") or line.startswith("# "):
            if has_text:
                chunks.append((section, "\n".join(current).strip()))
                current = []
            current.append(line)
            has_text = False
            if line.startswith("## "):
                section = line[3:].strip()
        else:
            current.append(line)
            has_text = has_text or bool(line.strip())
    if current and has_text:
        chunks.append((section, "\n".join(current).strip()))
    year = str(doc["metadata"].get("date", ""))[:4]
    # LangChain's splitter keeps a paragraph's lines joined by a newline, and joins paragraphs with two spaces and a newline
    def normalise(t: str) -> str:
        blocks, block = [], []
        for line in t.splitlines() + [""]:
            if line.strip():
                block.append(line)
            elif block:
                blocks.append("\n".join(block)); block = []
        return "  \n".join(blocks)
    norm = [(s, normalise(t)) for s, t in chunks]
    return [{"section": s or "intro", "text": t, "citation": f"[{doc['metadata']['id']} · {s or 'intro'} · {year}]"} for s, t in norm if t]


def render_markdown(text: str) -> str:
    """Headings, paragraphs and bullet lists; everything else is shown as text."""
    out, para, items = [], [], []

    def flush():
        nonlocal para, items
        if para:
            out.append("<p>" + html.escape(" ".join(para)) + "</p>")
            para = []
        if items:
            out.append("<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in items) + "</ul>")
            items = []

    for line in text.splitlines():
        if line.startswith("## "):
            flush(); out.append(f"<h3>{html.escape(line[3:].strip())}</h3>")
        elif line.startswith("# "):
            flush(); out.append(f"<h2>{html.escape(line[2:].strip())}</h2>")
        elif line.startswith("- "):
            if para:
                flush()
            items.append(line[2:].strip())
        elif not line.strip():
            flush()
        else:
            if items:
                flush()
            para.append(line.strip())
    flush()
    return "".join(out)


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


STYLE = """
/* Layout: a catalogue of the eight documents on the left, grouped by role; the selected document on the right as a sheet, each chunk labelled with its citation. Brand: AI for Data - Data for AI (teal, blue, Inter, JetBrains Mono). */
:root { --bg: #F7F9FB; --sheet: #ffffff; --ink: #16191D; --muted: #565F68; --faint: #8A929A; --rule: #E2E5E8; --teal: #00A898; --blue: #0058B0;
  --chip-evidence: #E3EEFA; --chip-evidence-fg: #0058B0; --chip-definition: #DDF4F0; --chip-definition-fg: #0A7A6E; --chip-outdated: #F3E9DA; --chip-outdated-fg: #7A4A0C; --chip-distractor: #ECEEF1; --chip-distractor-fg: #565F68;
  --sans: 'Inter', 'Helvetica Neue', Arial, sans-serif; --mono: 'JetBrains Mono', Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #14181C; --sheet: #1C2127; --ink: #EEF1F4; --muted: #AEB6BE; --faint: #8A929A; --rule: #2E353D; --teal: #2FC4B2; --blue: #6FA8F0;
  --chip-evidence: #1E3A5F; --chip-evidence-fg: #CFE1FA; --chip-definition: #164840; --chip-definition-fg: #C7F0E9; --chip-outdated: #4A3614; --chip-outdated-fg: #F3DDB4; --chip-distractor: #2E353D; --chip-distractor-fg: #D5DAE0; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #14181C; --sheet: #1C2127; --ink: #EEF1F4; --muted: #AEB6BE; --faint: #8A929A; --rule: #2E353D; --teal: #2FC4B2; --blue: #6FA8F0;
  --chip-evidence: #1E3A5F; --chip-evidence-fg: #CFE1FA; --chip-definition: #164840; --chip-definition-fg: #C7F0E9; --chip-outdated: #4A3614; --chip-outdated-fg: #F3DDB4; --chip-distractor: #2E353D; --chip-distractor-fg: #D5DAE0; color-scheme: dark }
* { box-sizing: border-box }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; padding-block: 0 40px; padding-inline: 0 }
.band { background: var(--sheet); border-bottom: 1px solid var(--rule) }
.band-in, .wrap { max-width: 1200px; margin: 0 auto; padding-inline: clamp(16px, 4vw, 40px) }
.band-in { display: flex; align-items: center; justify-content: space-between; gap: 20px; padding-block: 14px }
.band img { height: 44px; width: auto; max-width: 100% }
.band .where { font-size: 13px; color: var(--muted) }
.band .where a { color: var(--blue); text-decoration: none }
.band .where a:hover { text-decoration: underline }
header.top { display: flex; flex-direction: column; gap: 10px; padding-block: 30px 22px }
.eyebrow { margin: 0; font-size: 12.5px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--teal) }
header.top h1 { margin: 0; font-size: clamp(26px, 3.2vw, 38px); font-weight: 700; letter-spacing: -.01em; line-height: 1.15; text-wrap: balance }
header.top p { margin: 0; max-width: 74ch; color: var(--muted) }
.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 28px; align-items: start }
nav { position: sticky; top: env(safe-area-inset-top, 0px); display: flex; flex-direction: column; gap: 18px }
.group-h { margin: 0 0 4px; font-size: 13px; font-weight: 600 }
.group-p { margin: 0 0 8px; font-size: 13px; color: var(--muted); line-height: 1.4 }
nav ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px }
button.doc { width: 100%; text-align: left; font: inherit; color: var(--ink); background: transparent; border: 1px solid transparent; border-radius: 10px; padding: 8px 10px; display: flex; flex-direction: column; gap: 2px; cursor: pointer }
button.doc:hover { border-color: var(--rule); background: var(--sheet) }
button.doc[aria-current="true"] { background: var(--sheet); border-color: var(--teal); box-shadow: inset 3px 0 0 var(--teal) }
button.doc:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px }
.doc-title { font-weight: 500; line-height: 1.3 }
.doc-id { font-family: var(--mono); font-size: 11.5px; color: var(--faint) }
.chip { display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px }
.chip.evidence { background: var(--chip-evidence); color: var(--chip-evidence-fg) }
.chip.definition { background: var(--chip-definition); color: var(--chip-definition-fg) }
.chip.outdated { background: var(--chip-outdated); color: var(--chip-outdated-fg) }
.chip.distractor { background: var(--chip-distractor); color: var(--chip-distractor-fg) }
article.sheet { background: var(--sheet); border: 1px solid var(--rule); border-radius: 14px; padding: clamp(18px, 3vw, 36px); min-width: 0 }
.role { margin: 0 0 14px; color: var(--muted); font-size: 13.5px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px }
dl.meta { margin: 0 0 18px; border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule) }
.mrow { display: grid; grid-template-columns: 84px minmax(0, 1fr); gap: 12px; padding: 6px 0; border-bottom: 1px dashed var(--rule); font-size: 13px }
.mrow:last-child { border-bottom: 0 }
dt { font-family: var(--mono); color: var(--faint); font-size: 12px; padding-top: 2px }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere }
.chunk-note { margin: 0 0 10px; font-size: 13px; color: var(--muted) }
section.chunk { display: grid; gap: 4px; padding: 14px 0 14px 16px; border-left: 3px solid var(--teal); margin: 10px 0 }
.cite { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px; font-size: 12px; color: var(--muted) }
.cite code { font-family: var(--mono); font-size: 12px; color: var(--teal) }
.prose { font-size: 16px; line-height: 1.6; max-width: 70ch; min-width: 0 }
.prose h2 { font-size: 22px; margin: 0 0 8px; line-height: 1.2; font-weight: 700; letter-spacing: -.01em; text-wrap: balance }
.prose h3 { font-size: 17px; margin: 0 0 6px; font-weight: 600 }
.prose p { margin: 0 0 10px } .prose p:last-child { margin-bottom: 0 }
.prose ul { margin: 0 0 10px; padding-left: 22px } .prose li { margin: 2px 0 }
footer.note { font-size: 12.5px; color: var(--muted); padding-block: 24px 0 }
footer.note code, header.top code { font-family: var(--mono); font-size: 12.5px }
@media (max-width: 760px) { .layout { grid-template-columns: 1fr } nav { position: static } .mrow { grid-template-columns: 72px minmax(0, 1fr) } .band img { height: 36px } }
@media (prefers-reduced-motion: no-preference) { button.doc { transition: background .12s, border-color .12s } }
"""

SCRIPT = """
(function () {
  var ids = %s;
  function select(id, push) {
    if (ids.indexOf(id) < 0) id = ids[0];
    ids.forEach(function (i) {
      document.getElementById(i).hidden = (i !== id);
      document.getElementById('btn-' + i).setAttribute('aria-current', i === id ? 'true' : 'false');
    });
    try { localStorage.setItem('session01-doc', id); } catch (e) {}
    if (push) { try { history.replaceState(null, '', '#' + id); } catch (e) {} }
  }
  document.querySelectorAll('button.doc').forEach(function (b) { b.addEventListener('click', function () { select(b.dataset.id, true); }); });
  var start = (location.hash || '').replace('#', '');
  if (ids.indexOf(start) < 0) { try { start = localStorage.getItem('session01-doc') || ''; } catch (e) { start = ''; } }
  select(start, false);
  window.addEventListener('hashchange', function () { select((location.hash || '').replace('#', ''), false); });
})();
"""


def build(docs: list[dict], logo_light: str, logo_dark: str, site_links: bool) -> str:
    by = {d["metadata"]["id"]: d for d in docs}
    nav, articles = [], []
    for key, label, blurb, ids in ROLES:
        items = "".join(
            f'<li><button type="button" class="doc" id="btn-{i}" data-id="{i}"><span class="doc-title">{html.escape(by[i]["metadata"]["title"])}</span>'
            f'<span class="doc-id">{i} · {by[i]["metadata"]["date"][:4]}</span></button></li>' for i in ids)
        nav.append(f'<section class="group"><h2 class="group-h"><span class="chip {key}">{html.escape(label)}</span></h2><p class="group-p">{html.escape(blurb)}</p><ul>{items}</ul></section>')
        for i in ids:
            d = by[i]; m = d["metadata"]; chunks = split_sections(d)
            meta_rows = "".join(f'<div class="mrow"><dt>{k}</dt><dd>{html.escape(m.get(k, ""))}</dd></div>' for k in META_KEYS)
            body = "".join(
                f'<section class="chunk"><div class="cite"><code>{html.escape(c["citation"])}</code><span>{len(c["text"]):,} characters</span></div>'
                f'<div class="prose">{render_markdown(c["text"])}</div></section>' for c in chunks)
            n = len(chunks)
            articles.append(
                f'<article class="sheet" id="{i}" hidden><p class="role"><span class="chip {key}">{html.escape(label)}</span> {html.escape(NOTE[i])}</p>'
                f'<dl class="meta">{meta_rows}</dl>'
                f'<p class="chunk-note">Notebook 02 splits this document at its headings into {n} chunk{"s" if n != 1 else ""}. '
                f'When the application uses a chunk in an answer, it cites it with the label shown above the chunk.</p>{body}</article>')
    where = ('<a href="https://worldbank.github.io/ai4data/">AI for Data – Data for AI</a> · <a href="https://github.com/worldbank/ai4data/tree/main/training/session-01-building-ai-applications">Training, session 01</a>')
    logo = (f'<picture><source srcset="{logo_dark}" media="(prefers-color-scheme: dark)"><img src="{logo_light}" alt="AI for Data – Data for AI"></picture>')
    ids_json = json.dumps([i for _, _, _, ids in ROLES for i in ids])
    return f"""<title>Session 01 Document Set</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>{STYLE}</style>
<div class="band"><div class="band-in">{logo}<div class="where">{where}</div></div></div>
<div class="wrap">
<header class="top">
  <p class="eyebrow">Session 01 · Building AI applications for research and data</p>
  <h1>The session's document collection</h1>
  <p>Eight documents written for the workshop from World Development Indicators values retrieved on 13 October 2026. The application built in notebooks 02 to 07 searches only this collection, and the worked example's question is answered from it. None of the documents is an official publication.</p>
  <p>Notebook 02 splits each document at its headings into chunks. When the application uses a chunk in an answer, it cites it as <code>[document id · section · year]</code>, and that label is shown above each chunk below.</p>
</header>
<div class="layout">
  <nav aria-label="Documents">{"".join(nav)}</nav>
  <main>{"".join(articles)}</main>
</div>
<footer class="note">Source files: <code>training/session-01-building-ai-applications/assets/documents/</code> in the worldbank/ai4data repository. The figures in the briefs are rounded from <code>indicators.csv</code>; the 2016 summary reproduces the 2015 values.</footer>
</div>
<script>{SCRIPT % ids_json}</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--artifact", type=Path, help="write a body-only page with embedded logos to this path instead of the site page")
    args = ap.parse_args()
    docs = [read_document(p) for p in sorted(DOCS.glob("*.md"))]
    if args.artifact:
        page = build(docs, data_uri(LOGO), data_uri(LOGO_DARK), site_links=True)
        args.artifact.write_text(page, encoding="utf-8")
        print(f"wrote {args.artifact} ({len(page):,} bytes)")
        return 0
    page = build(docs, "/ai4data/img/program-logo.png", "/ai4data/img/program-logo-dark.png", site_links=True)
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<meta name="description" content="The eight documents of the worked example in session 01 of the AI for Data training series, with their metadata, their role, and their retrieval chunks.">\n'
            '<link rel="icon" href="/ai4data/img/favicon.ico">\n' + page.split("\n", 1)[0] + "\n</head>\n<body>\n" + page.split("\n", 1)[1] + "</body>\n</html>\n")
    SITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SITE_OUT.write_text(full, encoding="utf-8")
    print(f"wrote {SITE_OUT.relative_to(REPO)} ({len(full):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
