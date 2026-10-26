"""Tokenizer-aware chunking and cross-chunk dedupe (ported from the playground)."""

from .footnotes import WORD_RE, map_span_to_page


def chunk_text(text, tokenizer, max_tokens=384, overlap=50, unit="words"):
    """Sliding window of ``max_tokens`` units; chunks carry char ``start``/``end``.

    ``unit="words"`` (default) counts GLiNER's own words (``WORD_RE``, every
    punctuation mark is a word), which is the window the model really has: GLiNER
    truncates anything beyond 384 words silently. ``unit="tokens"`` counts
    subword tokens like the playground; those chunks can exceed the word window
    on markup-heavy text (``**``, ``<br>``, tables) and lose their tail.
    """
    if not text:
        return []
    if unit == "words":
        offsets = [m.span() for m in WORD_RE.finditer(text)]
        ids = offsets
    elif unit == "tokens":
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
    else:
        raise ValueError(f"unit must be 'words' or 'tokens', got {unit!r}")
    if not ids:
        return []
    step = max(1, max_tokens - overlap)
    chunks = []
    for start_tok in range(0, len(ids), step):
        end_tok = min(start_tok + max_tokens, len(ids))
        c_start = offsets[start_tok][0]
        c_end = offsets[end_tok - 1][1]
        chunks.append({"start": c_start, "end": c_end, "text": text[c_start:c_end]})
        if end_tok == len(ids):
            break
    return chunks


def _field(ent, name):
    return ent[name] if isinstance(ent, dict) else getattr(ent, name)


def chunk_page_span(chunk, s, e):
    """Map a chunk-relative ``[s, e)`` span to page coordinates, or ``None``."""
    return map_span_to_page(chunk, s, e)


def dedupe_overlap_entities(chunks, entities_per_chunk):
    """Drop cross-chunk duplicate entity spans via page coordinates.

    The sliding window shows overlap regions to two adjacent chunks, so the same
    occurrence is often extracted twice. Per identical ``(label, page_start,
    page_end)`` only the highest-score copy is kept (ties: earlier chunk).
    Unmappable spans and distinct occurrences of the same surface are kept.

    Returns ``(kept_per_chunk, dropped)``; *dropped* is a list of
    ``(chunk_idx, entity_idx)``.
    """
    occurrences = {}
    for ci, ents in enumerate(entities_per_chunk):
        for ei, ent in enumerate(ents or []):
            try:
                mapped = chunk_page_span(chunks[ci], _field(ent, "start"), _field(ent, "end"))
            except (KeyError, AttributeError, TypeError):
                mapped = None
            if mapped is None:
                continue
            key = (_field(ent, "label"), mapped[0], mapped[1])
            occurrences.setdefault(key, []).append((float(_field(ent, "score")), ci, ei))
    drop = set()
    for occ in occurrences.values():
        if len(occ) < 2:
            continue
        occ.sort(key=lambda t: (-t[0], t[1], t[2]))
        for _, ci, ei in occ[1:]:
            drop.add((ci, ei))
    kept = [
        [ent for ei, ent in enumerate(ents or []) if (ci, ei) not in drop]
        for ci, ents in enumerate(entities_per_chunk)
    ]
    return kept, sorted(drop)
