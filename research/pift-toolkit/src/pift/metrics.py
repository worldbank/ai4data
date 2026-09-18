"""
Retrieval metrics for a single labeled positive per query.

Given, for each query, the ranked list of retrieved record ids and the id of the
one correct record, compute Recall@k, MRR, and nDCG@k. With one binary positive,
nDCG@k reduces to ``1/log2(rank+1)`` when the positive is in the top k.
"""

from __future__ import annotations

import math


def _rank_of(retrieved: list[str], positive: str) -> int | None:
    for i, rid in enumerate(retrieved):
        if rid == positive:
            return i + 1
    return None


def evaluate_rankings(rankings: list[tuple[list[str], str]],
                      ks=(1, 5, 10)) -> dict:
    """``rankings`` is a list of (retrieved_ids, positive_id). Returns mean metrics."""
    n = len(rankings)
    if n == 0:
        return {}
    recall = {k: 0 for k in ks}
    mrr = 0.0
    ndcg = {k: 0.0 for k in ks}
    for retrieved, positive in rankings:
        rank = _rank_of(retrieved, positive)
        if rank is None:
            continue
        mrr += 1.0 / rank
        for k in ks:
            if rank <= k:
                recall[k] += 1
                ndcg[k] += 1.0 / math.log2(rank + 1)
    out = {f"recall@{k}": recall[k] / n for k in ks}
    out["mrr"] = mrr / n
    out.update({f"ndcg@{k}": ndcg[k] / n for k in ks})
    out["n_queries"] = n
    return out


def format_metrics(m: dict) -> str:
    keys = [k for k in m if k != "n_queries"]
    parts = [f"{k}={m[k]:.3f}" for k in keys]
    return f"n={m.get('n_queries', 0):,}  " + "  ".join(parts)
