"""
Hard-negative mining.

For each training query we embed the canonical-serialized corpus with a base
encoder, retrieve the top ranks, and keep high-ranked records that are not the
positive as hard negatives. Two safeguards matter for metadata catalogues, which
are full of near-duplicate records across collections:

  - skip the very top ranks (``skip_top``), which are the most likely to be
    unlabeled true positives;
  - drop any candidate whose embedding is more similar to the positive than
    ``max_sim_to_positive`` (a near-duplicate of the positive is not a real
    negative).

Output: a JSON-Lines file of training triplets, one per query:
``{query_id, query, facet, lang, positive_id, negative_ids}``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .config import Config
from .encoder import Encoder, resolve_model
from .records import load_records
from .serialize import serialize


def mine(config: Config, queries_path: str, out_path: str,
         miner_model: str | None = None, n_negatives: int | None = None,
         skip_top: int = 3, window: int = 30, max_sim_to_positive: float = 0.95,
         device: str | None = None) -> str:
    n_negatives = n_negatives or config.training["n_negatives"]
    records = load_records(config)
    ids = list(records)
    idx_of = {rid: k for k, rid in enumerate(ids)}
    corpus = [serialize(records[rid], config) for rid in ids]

    model_id, qp, dp = resolve_model(miner_model, config)
    enc = Encoder(model_id, query_prefix=qp, doc_prefix=dp, device=device)
    print(f"[mine] embedding {len(corpus):,} records with {model_id} on {enc.device}")
    D = enc.encode_documents(corpus, batch_size=64, show=True)

    rows = [json.loads(l) for l in Path(queries_path).read_text().splitlines() if l.strip()]
    qtexts = [r["query"] for r in rows]
    Q = enc.encode_queries(qtexts, batch_size=128, show=True)

    rng = random.Random(42)
    triplets = []
    sim_drops = 0
    # rank the corpus per query in blocks to bound memory
    block = 2000
    for s in range(0, len(rows), block):
        sims = Q[s:s + block] @ D.T          # (b, N) cosine
        order = np.argsort(-sims, axis=1)[:, : window + 1]
        for bi, r in enumerate(rows[s:s + block]):
            pos = r["record_id"]
            pos_i = idx_of.get(pos)
            if pos_i is None:
                continue
            ranked = [ids[j] for j in order[bi] if ids[j] != pos]
            candidates = ranked[skip_top:window]
            if max_sim_to_positive < 1.0:
                kept = []
                for c in candidates:
                    if float(D[pos_i] @ D[idx_of[c]]) <= max_sim_to_positive:
                        kept.append(c)
                    else:
                        sim_drops += 1
                candidates = kept
            if not candidates:
                continue
            negs = rng.sample(candidates, min(n_negatives, len(candidates)))
            triplets.append({
                "query_id": r["query_id"],
                "query": r["query"],
                "facet": r.get("facet", "unknown"),
                "lang": r.get("lang", "en"),
                "positive_id": pos,
                "negative_ids": negs,
            })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for t in triplets:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[mine] wrote {len(triplets):,} triplets -> {out_path} "
          f"(similarity guard dropped {sim_drops:,} near-duplicate negatives)")
    return out_path
