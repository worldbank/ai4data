"""
LLM-as-a-judge: graded relevance scoring of retrieved records.

The binary metrics (Recall@k, MRR, nDCG@k) score retrieval against the single
labeled positive per query. In a near-duplicate-rich catalogue that ground truth
is sparse and understates how useful a result list actually is: a retrieved
record can be a perfectly good answer without being the one labeled positive. A
graded judge addresses this by scoring each retrieved (query, record) pair on a
0-3 rubric, so two systems that both miss the labeled positive can still be told
apart by how relevant their top results are.

Relevance is a property of the (query, record) pair, not of the retriever, so
verdicts are cached on disk and shared across every compared model. Running a
second model over the same queries mostly reuses cached judgements, which is what
makes multi-model benchmarking affordable.

Providers: ``anthropic``, ``openai``, and ``heuristic`` (offline, no API key, a
lexical-overlap proxy for demos and tests).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import Config
from .llm import get_client
from .serialize import serialize

JUDGE_SYSTEM = """You are an expert evaluator for a catalogue search engine. \
Assess how relevant a retrieved record is to a user's search query.

Scoring rubric:
  3 = Highly relevant: the record directly and precisely addresses the query
  2 = Relevant: the record clearly covers the query's theme or concept
  1 = Marginal: loosely related, might help but misses the main intent
  0 = Not relevant: no meaningful connection to the query

Respond with a JSON object only: {"score": <0|1|2|3>, "reasoning": "<one sentence>"}"""

JUDGE_USER = """Query: {query}

Record:
{record}

How relevant is this record to the query? Respond with JSON only."""


def _key(query: str, record_id: str, model: str) -> str:
    h = hashlib.sha1(f"{query}\x00{record_id}\x00{model}".encode()).hexdigest()
    return h


class Judge:
    def __init__(self, config: Config, provider: str = "anthropic",
                 model: str = "claude-haiku-4-5",
                 cache_path: str = "data/judge_cache.json"):
        self.config = config
        self.provider = provider
        self.model = model
        self.client = get_client(provider, model)
        self.cache_path = Path(cache_path)
        self.cache: dict[str, float] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self._dirty = 0

    def _heuristic(self, query: str, record_text: str) -> float:
        q = set(re.findall(r"\w+", query.lower()))
        d = set(re.findall(r"\w+", record_text.lower()))
        if not q:
            return 0.0
        overlap = len(q & d) / len(q)
        return float(min(3, round(overlap * 4)))  # 0..3

    def score(self, query: str, record_id: str, record: dict) -> float:
        k = _key(query, record_id, self.model)
        if k in self.cache:
            return self.cache[k]
        record_text = serialize(record, self.config)[:1500]
        if self.provider in ("heuristic", "none", "offline"):
            s = self._heuristic(query, record_text)
        else:
            raw = self.client.complete(
                JUDGE_SYSTEM, JUDGE_USER.format(query=query, record=record_text))
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                s = float(json.loads(m.group(0)).get("score", 0)) if m else 0.0
            except (ValueError, json.JSONDecodeError):
                s = 0.0
        self.cache[k] = s
        self._dirty += 1
        if self._dirty >= 50:
            self.flush()
        return s

    def flush(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
        self._dirty = 0


def judged_at_k(judge: Judge, rankings, records: dict, k: int = 10) -> float:
    """Mean over queries of the average judged score of the top-k retrieved.

    ``rankings`` is a list of (retrieved_ids, positive_id) as produced by
    evaluation; ``positive_id`` is unused here (the judge scores actual results).
    Returns a value in [0, 3].
    """
    if not rankings:
        return 0.0
    total = 0.0
    n = 0
    # group by query is implicit: rankings already carry the query via closure;
    # we instead pass (retrieved_ids, query) tuples from the caller.
    for retrieved_ids, query in rankings:
        if not retrieved_ids:
            continue
        scores = [judge.score(query, rid, records[rid])
                  for rid in retrieved_ids[:k] if rid in records]
        if scores:
            total += sum(scores) / len(scores)
            n += 1
    judge.flush()
    return total / n if n else 0.0
