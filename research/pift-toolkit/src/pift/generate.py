"""
Synthetic query generation: the supervision signal.

For every record we ask an LLM to write several search queries for which that
record is a correct result, each targeting a different facet of the metadata
(a short keyword search, a natural-language question, a definition question, a
methodology question, and so on). The record is the ground-truth positive for
each of its queries.

Why generate instead of mining click logs: logs only cover what users have
already found, so they give no signal for the long tail of records nobody has
searched yet. Generating queries gives full coverage of every record and every
facet, in every language you ask for, including questions no user has issued.

Output: a JSON-Lines file of ``{query_id, query, facet, lang, record_id}``.
Train and eval queries are written separately; the eval set is generated with a
different (stronger) model so the benchmark does not just measure generator
style.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .llm import get_client
from .records import load_records, is_eval
from .serialize import serialize, _extract

SYSTEM_PROMPT = """You generate realistic search queries for a data catalogue. \
You will receive the full metadata of ONE record. Generate search queries for \
which this record is a correct retrieval result.

Rules:
- Each query is something a real researcher, analyst, journalist, or student would type.
- Follow the requested facet for each query.
- Vary phrasing, specificity, and persona. Do not copy the record's title verbatim \
into more than one query.
- If a language is specified, write every query in that language, but keep codes, \
acronyms, and proper nouns as-is.
- Return ONLY a JSON array: [{"facet": "...", "query": "..."}, ...]"""

USER_TEMPLATE = """Record metadata:
{doc}

Generate exactly {n} queries, one for each of these facets: {facets}.{language_clause}
Return ONLY the JSON array."""


def _facet_clause(config: Config, facets: list[str]) -> str:
    # Give the model a one-line gloss per facet so it knows what to target.
    glosses = {
        "keyword": "2-6 word keyword search",
        "natural": "complete natural-language question",
        "definition": "question about what the record means or measures",
        "methodology": "question about how the data is collected or computed",
        "geo": "query naming a covered place",
        "year": "query naming a covered year or period",
        "geo_year": "query combining a covered place and a covered year",
        "unit": "query referencing the measurement unit or scale",
        "source": "query referencing the source organization",
        "frequency": "query referencing the update frequency",
        "thematic": "broad topical or policy framing",
    }
    lines = [f"- {f}: {glosses.get(f, 'a relevant query for this facet')}" for f in facets]
    return "\n".join(lines)


def _pick_facets(record: dict, config: Config, n: int, rng: random.Random) -> list[str]:
    """Sample facets, skipping those whose evidence field is empty in this record."""
    available = []
    for facet in config.facets:
        needed = config.facets[facet]  # field labels this facet depends on
        ok = True
        for label in needed:
            key = next((f.key for f in config.fields if f.label == label), None)
            if key and not _extract(record, next(f for f in config.fields if f.key == key)):
                ok = False
                break
        if ok:
            available.append(facet)
    if not available:
        available = ["keyword", "natural"]
    rng.shuffle(available)
    # cycle facets if n exceeds the number available
    return [available[i % len(available)] for i in range(n)]


def _parse_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def _heuristic_queries(record: dict, config: Config, facets: list[str],
                       lang: str) -> list[dict]:
    """Offline, no-API query construction from the record's own fields."""
    title = ""
    for f in config.fields:
        if f.role == "protected":
            title = _extract(record, f)
            break
    out = []
    for facet in facets:
        if facet == "keyword":
            q = " ".join(title.split()[:4])
        elif facet == "natural":
            q = f"What data is available on {title.lower()}?"
        else:
            labels = config.facets.get(facet, [])
            ev = ""
            for label in labels:
                spec = next((f for f in config.fields if f.label == label), None)
                if spec:
                    ev = _extract(record, spec)
                    if ev:
                        break
            q = f"{title} {ev}".strip() if ev else title
        if q:
            out.append({"facet": facet, "query": q})
    return out


def generate_split(config: Config, split: str, limit: int = 0,
                   workers: int = 8, out_path: str | None = None) -> str:
    """Generate queries for ``split`` in {"train", "eval"}; returns output path."""
    gen = config.generation
    records = load_records(config)
    ev_frac = gen["eval_fraction"]
    ids = [rid for rid in records if (is_eval(rid, ev_frac) == (split == "eval"))]
    ids.sort()
    if limit:
        ids = ids[:limit]

    provider = gen["eval_provider"] if split == "eval" else gen["provider"]
    model = gen["eval_model"] if split == "eval" else gen["model"]
    client = get_client(provider, model)
    languages = gen["languages"]
    n = gen["queries_per_record"]
    rng = random.Random(42 if split == "train" else 43)

    out_path = out_path or f"data/{split}_queries.jsonl"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def work(rid: str) -> list[dict]:
        record = records[rid]
        facets = _pick_facets(record, config, n, rng)
        rows: list[dict] = []
        for lang in languages:
            if provider in ("heuristic", "none", "offline"):
                items = _heuristic_queries(record, config, facets, lang)
            else:
                doc = serialize(record, config)[:3500]
                clause = "" if lang == "en" else f"\nWrite every query in {lang}."
                user = USER_TEMPLATE.format(
                    doc=doc, n=len(facets),
                    facets=_facet_clause(config, facets),
                    language_clause=clause,
                )
                items = _parse_array(client.complete(SYSTEM_PROMPT, user))
            for j, it in enumerate(items):
                q = (it.get("query") or "").strip()
                if not q:
                    continue
                rows.append({
                    "query_id": f"{rid}-{lang}-{j}",
                    "query": q,
                    "facet": it.get("facet", "unknown"),
                    "lang": lang,
                    "record_id": rid,
                })
        return rows

    all_rows: list[dict] = []
    if provider in ("heuristic", "none", "offline") or workers <= 1:
        for rid in ids:
            all_rows.extend(work(rid))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for rows in pool.map(work, ids):
                all_rows.extend(rows)

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[generate] {split}: {len(all_rows):,} queries over {len(ids):,} records "
          f"({provider}:{model}) -> {out_path}")
    return out_path
