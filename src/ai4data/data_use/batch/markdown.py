"""Page-delimited markdown: one ``.md`` per document.

Pages are separated by ``\\n\\n<!-- page N -->\\n\\n`` where N is the 0-based PDF
page index. Unlike the playground, every page (including the first) gets a
marker, so blank pages that are skipped do not shift later page numbers. The
reader also accepts the playground format (no marker before the first page,
which is then page 0).
"""

import re

PAGE_DELIM = "\n\n<!-- page {n} -->\n\n"
_PAGE_RE = re.compile(r"(?:^|\n\n)<!-- page (\d+) -->\n\n")
# lone surrogates from broken PDF font encodings crash UTF-8 writes downstream
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def clean_surrogates(text):
    return _SURROGATE_RE.sub("�", text or "")


def pages_to_md(pages):
    """``[{"page": idx, "text": str}]`` -> markdown. Blank pages are dropped."""
    out = []
    for p in pages:
        text = clean_surrogates((p.get("text") or "").strip())
        if text:
            out.append(PAGE_DELIM.format(n=p["page"]).lstrip("\n") + text)
    return "\n\n".join(out)


def md_to_pages(md_text):
    """Markdown -> ``[{"page": idx, "text": str}]`` (blank pages dropped)."""
    md_text = md_text or ""
    parts = _PAGE_RE.split(md_text)
    pages = []
    if parts[0].strip():  # playground format: text before the first marker is page 0
        pages.append({"page": 0, "text": parts[0]})
    for i in range(1, len(parts), 2):
        pages.append({"page": int(parts[i]), "text": parts[i + 1]})
    return [p for p in pages if p["text"].strip()]
