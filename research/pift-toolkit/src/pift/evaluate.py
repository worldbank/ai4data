"""
Evaluation: held-out retrieval quality, the order-robustness test, and an
optional graded LLM-judge score.

Three things can be measured:

  1. Retrieval quality on the held-out queries against the full corpus indexed
     under the canonical (deterministic) field order: Recall@k, MRR, nDCG@k
     against the single labeled positive.

  2. Order robustness: rebuild the index under a different, fixed field order and
     re-evaluate. Nothing else changes, so any drop is pure field-order
     sensitivity. A permutation-invariant model should stay flat here.

  3. Graded relevance (optional, ``judge=...``): an LLM scores the top-k
     retrieved records on a 0-3 rubric, which captures usefulness the sparse
     single-positive labels miss. Verdicts are cached and shared across models.

Queries are scored against the WHOLE corpus, so the ranking task is realistic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import Config
from .encoder import Encoder, resolve_model
from .metrics import evaluate_rankings, format_metrics
from .records import load_records
from .serialize import build_segments, render_segments, serialize


def retrieve(enc: Encoder, corpus_texts, corpus_ids, queries, top_k=10):
    """Return a list of retrieved-id lists, aligned to ``queries``."""
    D = enc.encode_documents(corpus_texts, batch_size=64, show=True)
    Q = enc.encode_queries([q["query"] for q in queries], batch_size=128, show=True)
    out = []
    block = 2000
    for s in range(0, len(queries), block):
        sims = Q[s:s + block] @ D.T
        order = np.argsort(-sims, axis=1)[:, :top_k]
        for row in order:
            out.append([corpus_ids[j] for j in row])
    return out


def permuted_corpus(records, ids, config: Config, seed=1234):
    """Serialize every record under one fixed random field order (no dropout)."""
    texts = []
    for rid in ids:
        rng = random.Random(seed + hash(rid) % 10_000)
        texts.append(render_segments(
            build_segments(records[rid], config), config, rng=rng, permute=True))
    return texts


def canonical_corpus(records, ids, config: Config):
    return [serialize(records[rid], config) for rid in ids]


def load_queries(queries_path: str, valid_ids: set) -> list[dict]:
    queries = [json.loads(l) for l in Path(queries_path).read_text().splitlines() if l.strip()]
    return [q for q in queries if q["record_id"] in valid_ids]


def _metrics_from(retrieved, queries):
    return evaluate_rankings([(r, q["record_id"]) for r, q in zip(retrieved, queries)])


def evaluate(config: Config, queries_path: str, model: str | None = None,
             robustness: bool = True, device: str | None = None,
             judge=None, judge_top_k: int = 10) -> dict:
    records = load_records(config)
    ids = list(records)
    model_id, qp, dp = resolve_model(model, config)
    enc = Encoder(model_id, query_prefix=qp, doc_prefix=dp, device=device)
    queries = load_queries(queries_path, set(ids))
    print(f"[evaluate] {model_id} on {len(queries):,} queries / {len(ids):,} records")

    retrieved = retrieve(enc, canonical_corpus(records, ids, config), ids, queries)
    canon_m = _metrics_from(retrieved, queries)
    print(f"  canonical : {format_metrics(canon_m)}")
    result = {"model": model_id, "canonical": canon_m}

    if judge is not None:
        from .judge import judged_at_k
        pairs = [(r, q["query"]) for r, q in zip(retrieved, queries)]
        jk = judged_at_k(judge, pairs, records, k=judge_top_k)
        print(f"  judged@{judge_top_k} (0-3): {jk:.3f}")
        result[f"judged@{judge_top_k}"] = jk

    if robustness:
        perm_retrieved = retrieve(enc, permuted_corpus(records, ids, config), ids, queries)
        perm_m = _metrics_from(perm_retrieved, queries)
        delta = perm_m["ndcg@10"] - canon_m["ndcg@10"]
        print(f"  permuted  : {format_metrics(perm_m)}")
        print(f"  order-change delta (nDCG@10): {delta:+.3f}  "
              f"({'robust' if abs(delta) < 0.01 else 'fragile'})")
        result["permuted"] = perm_m
        result["order_delta_ndcg@10"] = delta
    return result
