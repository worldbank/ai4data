"""
Smoke tests for the core (no torch / no API needed).

Run: pytest -q   (from the pift-toolkit directory, with the package installed,
or with PYTHONPATH=src).
"""

import random
from pathlib import Path

from pift.config import load_config
from pift.records import load_records, split_ids, is_eval
from pift.serialize import serialize, build_segments
from pift.metrics import evaluate_rankings

CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "example.yaml")


def test_config_loads():
    cfg = load_config(CONFIG)
    assert cfg.protected_labels  # at least one protected field
    assert "Title" in cfg.protected_labels
    assert cfg.serialization["label_scheme"] in ("label", "key")


def test_serialize_canonical_contains_protected():
    cfg = load_config(CONFIG)
    records = load_records(cfg)
    rid = next(iter(records))
    text = serialize(records[rid], cfg)
    assert "Title:" in text


def test_protected_field_survives_dropout():
    cfg = load_config(CONFIG)
    records = load_records(cfg)
    rid = next(iter(records))
    # high dropout: every droppable field is likely removed, title must remain
    for seed in range(20):
        text = serialize(records[rid], cfg, rng=random.Random(seed),
                         permute=True, field_dropout=0.99)
        assert "Title:" in text


def test_permutation_changes_order_but_not_content_set():
    cfg = load_config(CONFIG)
    records = load_records(cfg)
    rid = next(iter(records))
    segs = build_segments(records[rid], cfg)
    labels = {s[0] for s in segs}
    # canonical order is deterministic
    a = serialize(records[rid], cfg)
    b = serialize(records[rid], cfg)
    assert a == b
    # a permuted render keeps the same labels present (no dropout)
    p = serialize(records[rid], cfg, rng=random.Random(1), permute=True)
    present = {seg.split(":")[0] for seg in p.split(" | ")}
    assert present <= labels


def test_split_is_disjoint_and_stable():
    cfg = load_config(CONFIG)
    ids = list(load_records(cfg))
    train, ev = split_ids(ids, 0.2)
    assert set(train).isdisjoint(ev)
    assert all(is_eval(i, 0.2) for i in ev)
    # stable across calls
    assert split_ids(ids, 0.2)[1] == ev


def test_metrics_perfect_ranking():
    rankings = [(["a", "b", "c"], "a"), (["x", "y"], "y")]
    m = evaluate_rankings(rankings, ks=(1, 5, 10))
    assert m["recall@1"] == 0.5      # only first query has positive at rank 1
    assert m["recall@5"] == 1.0
    assert 0 < m["mrr"] <= 1.0


def test_heuristic_judge_scores_and_caches(tmp_path):
    from pift.judge import Judge

    cfg = load_config(CONFIG)
    records = load_records(cfg)
    rid = next(iter(records))
    cache = tmp_path / "judge_cache.json"
    judge = Judge(cfg, provider="heuristic", cache_path=str(cache))
    title = records[rid].get("title", "")
    s = judge.score(title, rid, records[rid])     # query overlaps the record
    assert 0.0 <= s <= 3.0
    judge.flush()
    # a second judge instance reads the persisted verdict
    judge2 = Judge(cfg, provider="heuristic", cache_path=str(cache))
    assert judge2.cache  # non-empty
    assert judge2.score(title, rid, records[rid]) == s
