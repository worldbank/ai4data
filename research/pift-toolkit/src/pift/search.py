"""
Search index and query: deploy a fine-tuned model.

Builds an in-memory index by encoding the canonical serialization of every
record, then answers free-text queries. The index is a normalized embedding
matrix; similarity is a dot product. This is enough for catalogues up to ~100k
records on a laptop. For larger corpora, swap the brute-force search for a
vector database (FAISS, Qdrant, pgvector); the encoding and serialization stay
the same. See ``docs/deployment.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .encoder import Encoder, resolve_model
from .records import load_records
from .serialize import serialize


@dataclass
class Hit:
    record_id: str
    score: float
    record: dict


class SearchIndex:
    def __init__(self, config: Config, model: str | None = None, device: str | None = None):
        self.config = config
        self.records = load_records(config)
        self.ids = list(self.records)
        model_id, qp, dp = resolve_model(model, config)
        self.enc = Encoder(model_id, query_prefix=qp, doc_prefix=dp, device=device)
        texts = [serialize(self.records[rid], config) for rid in self.ids]
        print(f"[search] indexing {len(texts):,} records with {model_id} on {self.enc.device}")
        self.D = self.enc.encode_documents(texts, batch_size=64, show=True)

    def query(self, text: str, top_k: int = 10) -> list[Hit]:
        q = self.enc.encode_queries([text])[0]
        sims = self.D @ q
        order = np.argsort(-sims)[:top_k]
        return [Hit(self.ids[j], float(sims[j]), self.records[self.ids[j]])
                for j in order]


def interactive(config: Config, model: str | None = None, top_k: int = 10,
                device: str | None = None):
    index = SearchIndex(config, model=model, device=device)
    title_field = next((f.key for f in config.fields if f.role == "protected"), None)
    print("\nType a query (empty line to quit).")
    while True:
        try:
            text = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        for rank, hit in enumerate(index.query(text, top_k), 1):
            title = hit.record.get(title_field, hit.record_id) if title_field else hit.record_id
            print(f"  {rank:2d}. [{hit.score:.3f}] {title}  ({hit.record_id})")
