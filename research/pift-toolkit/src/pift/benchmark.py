"""
Benchmark: compare several models on the same held-out set.

Evaluates each model (base or fine-tuned) on the same queries and corpus and
prints a leaderboard with Recall@k, MRR, nDCG@10, the order-robustness delta,
and, optionally, the graded judge score. The corpus serialization is independent
of the model, so it is built once and reused; the judge cache is shared across
models, so each pair is scored at most once no matter how many models are
compared.

Outputs a console table, a Markdown table, and a JSON file with the full metrics
for every model.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .encoder import Encoder, resolve_model
from .evaluate import (
    canonical_corpus, permuted_corpus, retrieve, load_queries, _metrics_from,
)
from .records import load_records

# Columns shown in the leaderboard and whether a higher value is better.
COLUMNS = [
    ("recall@1", True), ("recall@5", True), ("recall@10", True),
    ("mrr", True), ("ndcg@10", True),
]


def benchmark(config: Config, queries_path: str, models: list[str | None],
              robustness: bool = True, device: str | None = None,
              judge=None, judge_top_k: int = 10,
              out_dir: str = "data/benchmark") -> list[dict]:
    records = load_records(config)
    ids = list(records)
    queries = load_queries(queries_path, set(ids))
    canon = canonical_corpus(records, ids, config)
    perm = permuted_corpus(records, ids, config) if robustness else None
    print(f"[benchmark] {len(models)} models on {len(queries):,} queries / {len(ids):,} records")

    rows = []
    for model in models:
        model_id, qp, dp = resolve_model(model, config)
        enc = Encoder(model_id, query_prefix=qp, doc_prefix=dp, device=device)
        retrieved = retrieve(enc, canon, ids, queries)
        m = _metrics_from(retrieved, queries)
        row = {"model": model_id, **{k: m[k] for k, _ in COLUMNS}}
        if judge is not None:
            from .judge import judged_at_k
            pairs = [(r, q["query"]) for r, q in zip(retrieved, queries)]
            row[f"judged@{judge_top_k}"] = judged_at_k(judge, pairs, records, k=judge_top_k)
        if robustness:
            perm_retrieved = retrieve(enc, perm, ids, queries)
            pm = _metrics_from(perm_retrieved, queries)
            row["order_delta"] = pm["ndcg@10"] - m["ndcg@10"]
        rows.append(row)
        print(f"  done: {model_id}")

    rows.sort(key=lambda r: r["ndcg@10"], reverse=True)
    _write(rows, queries, out_dir, judge_top_k if judge is not None else None, robustness)
    return rows


def _fmt(rows, judge_k, robustness):
    cols = [c for c, _ in COLUMNS]
    headers = ["model"] + cols
    if judge_k is not None:
        headers.append(f"judged@{judge_k}")
    if robustness:
        headers.append("order Δ")
    # best value per column to mark with **bold** in markdown
    best = {}
    for c in cols + ([f"judged@{judge_k}"] if judge_k is not None else []):
        best[c] = max(r.get(c, float("-inf")) for r in rows)
    if robustness:
        best["order_delta"] = max(r.get("order_delta", float("-inf")) for r in rows)

    def cell(r, key, md):
        if key not in r:
            return ""
        v = f"{r[key]:+.3f}" if key == "order_delta" else f"{r[key]:.3f}"
        is_best = (key in best and abs(r[key] - best[key]) < 1e-9)
        return f"**{v}**" if (md and is_best) else v

    def render(md: bool):
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        if md:
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            cells = [r["model"]] + [cell(r, c, md) for c in cols]
            if judge_k is not None:
                cells.append(cell(r, f"judged@{judge_k}", md))
            if robustness:
                cells.append(cell(r, "order_delta", md))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    return render(md=False), render(md=True)


def _write(rows, queries, out_dir, judge_k, robustness):
    console, md = _fmt(rows, judge_k, robustness)
    print("\n" + console + "\n")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leaderboard.md").write_text(
        f"# Benchmark\n\n{len(queries):,} held-out queries.\n\n{md}\n")
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"[benchmark] wrote {out}/leaderboard.md and {out}/results.json")
