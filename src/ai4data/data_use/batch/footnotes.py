"""Footnote-to-chunk linkage for GLiNER extraction input.

Ported from ``ai4data-playground/data_use/footnotes.py`` (behaviour unchanged).

Problem: pymupdf4llm keeps a page's footnote *definitions* at the page tail
while markers (``<sup>N</sup>``) sit mid-page. The 384/50 sliding window then
puts marker and definition in different chunks (~20% of PAD pages), so the
extractor sees dangling markers and dataset mentions in definitions lose the
claim they support.

Fix: :func:`attach_footnotes` inserts each definition *inline*, right after
its marker (``<sup>1</sup> [^1: Defined as ...]``) — the markers stay in the
text, so traceability is preserved, and marker + definition can never be
separated by trimming or truncation. Entity offsets are stored
chunk-relative (into ``input_text``), so inline insertion does not affect
the ``(corpus_id, page, chunk, start, end)`` attribute join; page-coordinate
mapping for display/PDF tracing goes through :func:`map_to_page` and the
per-chunk ``footnotes`` links.

Budget: the binding limit at inference is GLiNER's 384 *whitespace-word*
window (not subwords), counted with GLiNER's own split over the enriched
text, insertions included. The redundant overlap tail absorbs pressure
first, then core tail (never past the last insertion). Chunk ``(page,
index, start, end)`` identity is unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

# GLiNER's own word split (the binding 384-word window at inference)
WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")

logger = logging.getLogger(__name__)

MARK_RE = re.compile(r"<sup>(.*?)</sup>")
_STRIP_EMPH = re.compile(r"^(\*\*|\*|__|_)+|(\*\*|\*|__|_)+$")


def find_markers(text: str) -> List[tuple]:
    """``[(number, match_start, match_end)]`` footnote markers in order.

    Handles emphasis inside the tag (``<sup>**11**</sup>``), multi-refs
    (``<sup>5,6</sup>`` → 5 and 6), and ignores ordinals (``<sup>2nd</sup>``)
    which cannot map to numbered definitions.
    """
    out = []
    for m in MARK_RE.finditer(text):
        inner = _STRIP_EMPH.sub("", m.group(1))
        for tok in re.split(r"[,;]", inner):
            tok = tok.strip()
            if re.fullmatch(r"\d+", tok):
                out.append((tok, m.start(), m.end()))
    return out
_ENTRY_RE = re.compile(r"^(\d+)\s+(.*\S)\s*$")
_LINK_ENTRY_RE = re.compile(r"^\[(\d+)\.?\s+(.*)\]\(\S+\)\s*$")
# chart-caption notes use dotted numbers ("1. The poverty rate ...").
# Parsed ONLY under strict guards (see _dotted_ok): body text like
# "2. **Lower economic activity ...**" must never match.
_DOTTED_ENTRY_RE = re.compile(r"^(\d+)\.\s+(.*\S)\s*$")
_DOTTED_MAX_CHARS = 400  # measured on markup/URL-stripped text


def _clean_len(text: str) -> int:
    """Length of the linguistic content: markup tags and bare URLs removed
    (a link-heavy chart note reads ~250 words but measures ~450 raw chars)."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return len(text.strip())
_FOOTER_RE = re.compile(r"^Page \d+ of \d+\s*$")

MAX_WORDS_DEFAULT = 384


def _entry_start(line: str) -> Optional[str]:
    """Footnote number if *line* opens a definition entry, else None."""
    m = _ENTRY_RE.match(line) or _LINK_ENTRY_RE.match(line)
    return m.group(1) if m else None


def _remainder(stripped: str) -> str:
    """Definition body without the leading number / link brackets."""
    m = _ENTRY_RE.match(stripped) or _LINK_ENTRY_RE.match(stripped)
    assert m is not None
    remainder = m.group(2).strip()
    # unwrap '[N text](url)' -> keeps label, drops bare URL
    link = re.match(r"^\[(.*)\]\(\S+\)$", remainder)
    if link is not None:
        remainder = link.group(1).strip()
    return remainder


def _plausible(num: str) -> bool:
    """Footnote numbers are small; 4+-digit line-starts are years, not notes."""
    return not (len(num) >= 4 and num.isdigit())


def parse_footnotes(page_text: str) -> Dict[str, Dict[str, object]]:
    """Parse ``{number: {"text", "start", "end"}}`` footnote definitions.

    Two block shapes are accepted (first wins on duplicate numbers):

    - *pure clusters*: blank-separated blocks whose every non-blank line
      opens an entry — covers contiguous PAD blocks, blank-separated
      entries, and mid-page column footnotes in two-column layouts;
    - *trailing run*: from the last entry opener backward across
      blank-separated pure clusters, parsed with continuation-line support
      for wrapped definitions; ends at a ``Page X of Y`` footer or text end.

    Numbers are page-scoped (first wins on duplicates). 4+-digit openers
    (years like ``2023 ...``) are skipped. Dotted chart-caption notes
    (``1. The poverty rate ...``) are parsed only when the number is
    referenced by a marker on the page AND the entry is brief — long
    numbered body paragraphs (``2. **Lower economic activity ...**``)
    are excluded by length.
    """
    lines = page_text.split("\n")
    # line index -> char offset
    offs = []
    pos = 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1

    # blank-separated blocks of non-blank line indices
    blocks: List[List[int]] = []
    cur: List[int] = []
    for i, ln in enumerate(lines):
        if ln.strip():
            cur.append(i)
        elif cur:
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    def block_opens(block: List[int]) -> List[str]:
        return [_entry_start(lines[i].strip()) for i in block]

    def is_pure(block: List[int]) -> bool:
        opens = block_opens(block)
        return bool(opens) and all(n is not None for n in opens)

    # numbers actually referenced on this page: gates wrapped mid-page blocks
    page_markers = {num for num, _, _ in find_markers(page_text)}

    # trailing run: last block with an opener, extended backward across
    # blank-separated pure clusters (wrapped definitions need the run).
    run_end = run_start = None
    for b, block in enumerate(blocks):
        if any(n is not None for n in block_opens(block)):
            run_end = b
    if run_end is not None:
        run_start = run_end
        while run_start > 0 and is_pure(blocks[run_start - 1]):
            run_start -= 1

    # line ranges parsed with continuation support (trailing run) vs
    # opener-only (earlier pure clusters)
    run_lines = set()
    if run_start is not None:
        for b in range(run_start, run_end + 1):
            run_lines.update(blocks[b])
    pure_lines = set()
    # gated blocks: non-pure block whose FIRST line opens an entry for a
    # referenced marker (wrapped mid-page definitions); continuations stay
    # inside the block, further openers need referenced numbers too
    gated_lines = set()
    line_block = {}
    for block in blocks:
        for i in block:
            line_block[i] = block
        if is_pure(block) and not (run_lines and block[0] in run_lines):
            pure_lines.update(block)
            continue
        opens = block_opens(block)
        if opens and opens[0] in page_markers:
            gated_lines.update(block)

    defns: Dict[str, Dict[str, object]] = {}
    cur_num: Optional[str] = None
    cur_lines: List[str] = []
    cur_start = 0
    cur_block: Optional[List[int]] = None

    def flush(end_line: int) -> None:
        if cur_num is not None and cur_num not in defns and cur_lines:
            text = "\n".join(cur_lines).strip()
            if text:
                end = offs[end_line] - 1 if end_line < len(offs) else len(page_text)
                defns[cur_num] = {
                    "text": text,
                    "start": cur_start,
                    "end": end,
                }

    # dotted chart-caption notes first: block-first "N. ..." lines whose
    # number is referenced on the page and whose entry is brief. Consumed
    # lines are hidden from the main loop below.
    consumed = set()
    for block in blocks:
        m = _DOTTED_ENTRY_RE.match(lines[block[0]].strip())
        if not m or m.group(1) not in page_markers or m.group(1) in defns:
            continue
        # chart notes are plain small text; fully emphasized lines
        # ("**Lower economic activity ...**") are body paragraphs/headings
        if re.match(r"^[*_~<|#]", m.group(2).strip()):
            continue
        parts = [m.group(2).strip()]
        consumed.add(block[0])
        for i in block[1:]:
            s = lines[i].strip()
            if (_entry_start(s) or _DOTTED_ENTRY_RE.match(s)
                    or _FOOTER_RE.match(s)):
                break
            parts.append(s)
            consumed.add(i)
        text = "\n".join(parts).strip()
        if text and _clean_len(text) <= _DOTTED_MAX_CHARS:
            defns[m.group(1)] = {"text": text, "start": offs[block[0]],
                                 "end": offs[block[0]] + len(lines[block[0]])}

    for i, ln in enumerate(lines):
        if i in consumed:
            continue
        stripped = ln.strip()
        if not stripped or _FOOTER_RE.match(stripped):
            flush(i)
            cur_num, cur_lines, cur_block = None, [], None
            if _FOOTER_RE.match(stripped):
                break
            continue
        num = _entry_start(stripped)
        gated_here = i in gated_lines and (
            line_block.get(i, [i])[0] == i or (num in page_markers))
        if (num is not None and _plausible(num)
                and (i in run_lines or i in pure_lines or gated_here)):
            flush(i)
            cur_num, cur_lines, cur_start = num, [_remainder(stripped)], offs[i]
            cur_block = line_block.get(i)
        elif num is not None:
            # opener-shaped line we refuse (implausible year, or outside any
            # footnote block): close any open entry, never swallow body text
            flush(i)
            cur_num, cur_lines, cur_block = None, [], None
        elif cur_num is not None and (i in run_lines
                                      or line_block.get(i) == cur_block):
            cur_lines.append(stripped)
        elif cur_num is not None:
            # continuation-looking line outside the entry's block: close the
            # entry rather than swallowing body text
            flush(i)
            cur_num, cur_lines, cur_block = None, [], None
    flush(len(lines))
    return defns


def _word_spans(text: str):
    return [(m.start(), m.end()) for m in WORD_RE.finditer(text)]


def attach_footnotes(
    page_text: str,
    chunks: List[Dict[str, object]],
    tokenizer=None,
    max_tokens: int = 384,
    max_words: int = MAX_WORDS_DEFAULT,
) -> List[Dict[str, object]]:
    """Insert resolved footnote definitions into marker-bearing chunks.

    Layout of an enriched chunk is ``core (+ inline insertions) + overlap
    tail``: each definition is spliced right after its marker
    (``<sup>1</sup> [^1: Defined as ...]``), so marker and definition can
    never be separated by trimming or truncation. Markers stay in the text
    for traceability. The overlap tail is redundant by construction (the
    next chunk opens with the same page region), so it absorbs word-budget
    pressure first — and any residual GLiNER-side truncation would eat
    duplicated tail, never body or footnotes.

    Only markers in the chunk *core* (up to the next chunk's start) are
    linked; markers in the overlap tail belong to the next chunk, which
    links them itself. Chunk ``(start, end)`` identity is preserved; entity
    offsets stay chunk-relative (into ``input_text``), so the
    ``(corpus_id, page, chunk, start, end)`` join is unaffected — page
    mapping for display/PDF tracing goes through :func:`map_to_page`.

    Args:
        page_text: full page markdown (chunks hold offsets into it).
        chunks: ``{"start", "end", "text"}`` dicts (e.g. from
            :func:`ai4data.data_use.batch.chunking.chunk_text`).
        tokenizer: HF tokenizer for the ``over_budget_sub`` stat (optional).
        max_tokens: subword budget for the stat only. The binding inference
            limit is ``max_words`` (GLiNER truncates whitespace-words).
        max_words: whitespace-word budget counted with GLiNER's own split
            (``WORD_RE``) over the *enriched* text, insertions included.

    Returns enriched chunk dicts with ``text`` (core + inline insertions +
    tail), ``ov_start`` (page offset where the overlap tail begins),
    ``tail_from`` (enriched offset where the tail begins), ``tail_len``
    (kept tail length), ``footnotes`` (``[{n, def_start, def_end, at_start,
    at_end, ps, prefix}]`` with ``at_*`` offsets into the enriched text,
    ``ps`` the page offset of the insertion point), ``trimmed_tail_chars``
    / ``trimmed_core_chars``, ``footnotes_dropped_budget``,
    ``over_budget_sub`` and ``unresolved``.
    """
    defns = parse_footnotes(page_text)
    enriched = []
    n = len(chunks)
    for i, c in enumerate(chunks):
        start, end, body = c["start"], c["end"], c["text"]
        # overlap tail = page region shared with the next chunk's head
        ov = chunks[i + 1]["start"] if i + 1 < n else end
        ov = max(start, min(ov, end))
        core, tail = body[:ov - start], body[ov - start:]
        base = {**c, "footnotes": [], "ov_start": ov,
                "tail_from": ov - start, "tail_len": len(tail),
                "trimmed_tail_chars": 0, "trimmed_core_chars": 0,
                "footnotes_dropped_budget": False,
                "over_budget_sub": False, "unresolved": []}
        # Link markers whose *opening* lies in the core (a marker split by
        # the core/tail boundary still belongs to this chunk); tail-only
        # markers belong to the next chunk, which links them itself.
        # Markers already co-located with their definition need nothing.
        seen, wanted = set(), []
        for num, ms, me in find_markers(body):
            if ms < len(core) and num not in seen:
                seen.add(num)
                wanted.append((num, me))
        todo = [(m, me) for m, me in wanted if m in defns
                and not (defns[m]["start"] >= start and defns[m]["end"] <= end)]
        base["unresolved"] = [m for m, _ in wanted if m not in defns]
        if not todo:
            enriched.append(base)
            continue
        # a linked marker straddling the core/tail boundary extends the core
        # to swallow the full match, so insertions never split a tag; the
        # next chunk still opens at ov, so overlap coverage is intact.
        # (tail_from/tail_len track the extension: the dropped-budget path
        # below reuses base and must stay consistent.)
        core_end = max([ov - start] + [me for _, me in todo])
        if core_end > ov - start:
            core, tail = body[:core_end], body[core_end:]
            base["ov_start"] = start + core_end
            base["tail_from"] = core_end
            base["tail_len"] = len(tail)
        # splice insertions right after their markers (first occurrence order)
        inserts = []  # (core_offset, line)
        for m, me in todo:
            inserts.append((me, f" [^{m}: {defns[m]['text']}]"))
        inserts.sort()
        pieces, cursor = [], 0
        for me, line in inserts:
            pieces.append(core[cursor:me])
            pieces.append(line)
            cursor = me
        pieces.append(core[cursor:])
        core_ins = "".join(pieces)
        links = []
        at = 0
        for (m, me), (_, line) in zip(todo, inserts):
            # at_* in core_ins coordinates; ps = page offset of insertion
            at = core_ins.index(line, at)
            links.append({"n": m, "def_start": defns[m]["start"],
                          "def_end": defns[m]["end"],
                          "at_start": at, "at_end": at + len(line),
                          "ps": start + me,
                          "prefix": len(f" [^{m}: ")})
            at += len(line)

        # word budget over the true enriched string (same split GLiNER
        # uses): trim the redundant overlap tail first, then core tail
        # (never past the last insertion end).
        joined = core_ins + tail
        Li = len(core_ins)
        spans = _word_spans(joined)
        core_idx = [k for k, (s, e) in enumerate(spans) if s < Li]
        tail_idx = [k for k, (s, e) in enumerate(spans) if s >= Li]
        keep_core_n, keep_tail_n = len(core_idx), len(tail_idx)
        if len(spans) > max_words:
            drop = len(spans) - max_words
            cut = min(len(tail_idx), drop)
            keep_tail_n -= cut
            drop -= cut
            if drop > 0:
                last_ins_end = max(l["at_end"] for l in links)
                must = sum(1 for k in core_idx if spans[k][1] <= last_ins_end)
                keep_core_n = max(len(core_idx) - drop, must)
        # drop links stranded by core trimming (only when insertions alone
        # exceed the window — pathological); survivors keep valid at_*.
        if keep_core_n < len(core_idx):
            cut_at = spans[core_idx[keep_core_n]][0]
            links = [l for l in links if l["at_end"] <= cut_at]
        core_kept = (core_ins[:spans[core_idx[keep_core_n]][0]].rstrip()
                     if keep_core_n < len(core_idx) else core_ins)
        tail_kept = (tail[:spans[tail_idx[keep_tail_n]][0] - Li].rstrip()
                     if keep_tail_n < len(tail_idx) else tail)

        text = core_kept + tail_kept
        links = [l for l in links if l["at_end"] <= len(core_kept)]
        if len(_word_spans(text)) > max_words or not links:
            # insertions alone exceed the window, or nothing survived:
            # keep the body, metadata only.
            enriched.append({**base, "footnotes_dropped_budget": True})
            continue
        over_sub = None
        if tokenizer is not None:
            over_sub = len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens
        enriched.append({**base, "text": text, "footnotes": links,
                         "tail_from": len(core_kept),
                         "tail_len": len(tail_kept),
                         "trimmed_core_chars": len(core) + sum(
                             len(l) for _, l in inserts) - len(core_kept),
                         "trimmed_tail_chars": len(tail) - len(tail_kept),
                         "over_budget_sub": bool(over_sub) if over_sub is not None else False,
                         "over_budget_sub_unknown": over_sub is None})
    return enriched


def map_span_to_page(chunk: Dict[str, object], s: int, e: int):
    """Map a chunk-relative ``[s, e)`` span to page offsets, or ``None``.

    Understands plain :func:`ai4data.data_use.batch.chunking.chunk_text` dicts (identity)
    and enriched dicts from :func:`attach_footnotes` (inline insertions +
    overlap tail). Spans inside an insertion map into the definition's true
    page range; spans straddling regions return ``None`` (callers must keep
    such spans rather than guess).
    """
    start = chunk.get("start")
    if start is None or e <= s:
        return None
    links = chunk.get("footnotes") or []
    tail_from = chunk.get("tail_from")
    if not links and tail_from is None:
        return (start + s, start + e)
    text, tail_len = chunk.get("text", ""), chunk.get("tail_len", 0) or 0
    tail_from = len(text) if tail_from is None else tail_from
    for link in links:
        a, b = link["at_start"], link["at_end"]
        if a <= s and e <= b:
            prefix = link.get("prefix", 0)
            base = link["def_start"] + len(link["n"]) + 1
            return (base + (s - a - prefix), base + (e - a - prefix))
        if a < e and s < b:
            return None  # straddles an insertion
    if s >= tail_from:
        if e <= tail_from + tail_len:
            ov = chunk.get("ov_start", chunk.get("end"))
            if ov is None:
                return None
            return (ov + (s - tail_from), ov + (e - tail_from))
        return None
    ins = sum(l["at_end"] - l["at_start"] for l in links if l["at_end"] <= s)
    return (start + s - ins, start + e - ins)


def map_to_page(chunk: Dict[str, object], s: int, e: int) -> tuple:
    """Map an ``(s, e)`` span in enriched chunk text back to page offsets.

    Layout is inline insertions + overlap tail: body spans subtract
    preceding insertions, footnote spans map into the definition's true
    page range via the recorded links, tail spans map via ``ov_start``.
    Spans straddling regions clamp to the chunk window end.
    """
    mapped = map_span_to_page(chunk, s, e)
    if mapped is not None:
        return mapped
    # straddles a region boundary: clamp to the chunk window end
    return (chunk["start"] + s, chunk["end"])
