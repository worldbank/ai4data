"""
Schema-aware serialization with field-order permutation and field dropout.

This is the core of permutation-invariant fine-tuning (PI-FT). A structured
record is a set of labeled fields. To embed it with a text encoder the fields
must be flattened into one string, which forces a choice of field order. PI-FT
removes the model's dependence on that choice by serializing each record under a
freshly sampled field order (and with random dropout of non-essential fields) at
every training step, so meaning binds to the field labels rather than to
position.

Two modes:
  - canonical (``permute=False``): a deterministic field order; use this when
    you build the search index and at evaluation time.
  - augmented (``permute=True`` with an ``rng``): the field order is shuffled and
    non-protected fields are dropped at probability ``field_dropout``; use this
    during training.

The serializer is driven entirely by a :class:`pift.config.Config`, so it works
on any catalogue's native field names with no hardcoded schema.
"""

from __future__ import annotations

import random
from typing import Optional

from .config import Config, FieldSpec

# Generous bound on an elastic field's raw text before budget allocation
# truncates it; only guards against pathological multi-thousand-word inputs.
_ELASTIC_HARD_CAP = 8000


def _clean(text, max_chars: int) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())[:max_chars]


def _get_path(record: dict, path: str):
    """Resolve a dotted path (e.g. ``source.organization``) into a record."""
    cur = record
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _extract(record: dict, spec: FieldSpec) -> str:
    """Turn one configured field into a plain text value.

    Supports three extraction shapes via ``spec.extract['type']``:
      - ``scalar`` (default): the value at ``spec.key`` rendered as text.
      - ``list_scalar``: a list of scalars joined by ``sep``.
      - ``list_join``: a list of dicts; ``subkey`` is pulled from each and the
        results are de-duplicated and joined by ``sep``.
    """
    value = _get_path(record, spec.key)
    if value is None:
        return ""
    etype = (spec.extract or {}).get("type", "scalar")
    sep = (spec.extract or {}).get("sep", ", ")
    if etype == "scalar":
        return _clean(value, spec.effective_cap(_ELASTIC_HARD_CAP))
    if etype == "list_scalar":
        items = [str(v) for v in value if v not in (None, "")]
        return _clean(sep.join(dict.fromkeys(items)), spec.effective_cap(_ELASTIC_HARD_CAP))
    if etype == "list_join":
        subkey = spec.extract.get("subkey")
        items = []
        for entry in value or []:
            v = entry.get(subkey) if isinstance(entry, dict) else None
            if v:
                items.append(str(v))
        return _clean(sep.join(dict.fromkeys(items)), spec.effective_cap(_ELASTIC_HARD_CAP))
    raise ValueError(f"unknown extract type: {etype!r} for field {spec.key!r}")


def build_segments(record: dict, config: Config) -> list[tuple[str, str, str]]:
    """Return ``(label, key, text)`` for every populated, configured field.

    ``build_segments`` is deterministic (no sampling), so callers that serialize
    the same record many times (dynamic per-access augmentation during training)
    can build segments once and reuse them via :func:`render_segments`.
    """
    segments: list[tuple[str, str, str]] = []
    for spec in config.fields:
        text = _extract(record, spec)
        if text:
            segments.append((spec.label, spec.key, text))
    return segments


def _fair_allocate(lengths: list[int], budget: int) -> list[int]:
    """Max-min fair split of ``budget`` over items of the given full lengths.

    Items shorter than their fair share take their full length and release the
    surplus to longer items, so a short description leaves more room for a long
    methodology and vice versa.
    """
    n = len(lengths)
    alloc = [0] * n
    remaining = max(0, budget)
    for rank, i in enumerate(sorted(range(n), key=lambda j: lengths[j])):
        share = remaining // (n - rank)
        take = min(lengths[i], share)
        alloc[i] = take
        remaining -= take
    return alloc


def _fit_budget(
    segments: list[tuple[str, str, str]],
    config: Config,
    elastic_labels: set,
    raw_mode: bool,
) -> list[tuple[str, str, str]]:
    """Truncate only the elastic fields so the rendered string fits the budget."""
    if not segments:
        return segments
    total_chars = config.serialization["total_chars"]
    sep = config.serialization["separator"]

    def label(c: str, k: str) -> str:
        return k if raw_mode else c

    elastic_idx = [i for i, (c, _k, _t) in enumerate(segments) if c in elastic_labels]
    if not elastic_idx:
        return segments

    n = len(segments)
    overhead = len(sep) * (n - 1)
    fixed = sum(
        len(label(c, k)) + 2 + len(t)
        for i, (c, k, t) in enumerate(segments) if i not in set(elastic_idx)
    )
    elastic_label_cost = sum(
        len(label(segments[i][0], segments[i][1])) + 2 for i in elastic_idx
    )
    budget = total_chars - overhead - fixed - elastic_label_cost
    allocs = _fair_allocate([len(segments[i][2]) for i in elastic_idx], budget)

    out = list(segments)
    for j, i in enumerate(elastic_idx):
        c, k, t = segments[i]
        if allocs[j] < len(t):
            cut = t[:allocs[j]]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            out[i] = (c, k, cut.rstrip())
    return out


def render_segments(
    segments: list[tuple[str, str, str]],
    config: Config,
    rng: Optional[random.Random] = None,
    permute: bool = False,
    field_dropout: float = 0.0,
    protect: Optional[set] = None,
) -> str:
    """Render pre-built segments (dropout, budget, permutation, formatting).

    Does not mutate ``segments``. ``protect`` adds field labels that must
    survive dropout for this call (used to keep a query facet's evidence in its
    positive document).
    """
    raw_mode = config.serialization["label_scheme"] == "key"
    keep = config.protected_labels | (protect or set())

    if rng is not None and field_dropout > 0:
        segments = [
            seg for seg in segments
            if seg[0] in keep or rng.random() > field_dropout
        ]

    segments = _fit_budget(segments, config, config.elastic_labels, raw_mode)

    if rng is not None and permute:
        segments = segments[:]
        rng.shuffle(segments)

    return config.serialization["separator"].join(
        f"{(key if raw_mode else label)}: {text}"
        for label, key, text in segments
    )


def serialize(
    record: dict,
    config: Config,
    rng: Optional[random.Random] = None,
    permute: bool = False,
    field_dropout: float = 0.0,
    protect: Optional[set] = None,
) -> str:
    """Serialize one record to a single string.

    canonical:  ``serialize(record, config)``
    augmented:  ``serialize(record, config, rng=rng, permute=True, field_dropout=0.15)``
    """
    return render_segments(
        build_segments(record, config),
        config,
        rng=rng,
        permute=permute,
        field_dropout=field_dropout,
        protect=protect,
    )
