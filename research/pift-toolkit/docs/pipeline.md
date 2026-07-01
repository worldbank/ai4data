# Pipeline guide

The five stages, in order, with the commands and what each produces. All read the
catalogue and schema from your config. Run from the `pift-toolkit` directory with
the package installed (`pip install -e ".[train]"`).

A device flag (`--device cpu|mps|cuda`) is available on the stages that encode;
omit it to auto-detect (CUDA, then Apple MPS, then CPU).

## 0. Preview (sanity check)

Before generating anything, confirm your config serializes records the way you
expect.

```bash
pift preview --config configs/example.yaml --n 3
```

For each record it prints the canonical serialization and one augmented
(permuted + dropped) serialization. Check that the title survives, long fields are
sensibly truncated, and the labels read correctly. Fix the config until this
looks right; everything downstream depends on it.

## 1. Generate supervision

```bash
pift generate --config configs/example.yaml --split train
pift generate --config configs/example.yaml --split eval
```

Produces `data/train_queries.jsonl` and `data/eval_queries.jsonl`, each a
JSON-Lines file of `{query_id, query, facet, lang, record_id}`. The two splits
cover disjoint records (by id hash) and use different generator models.

Notes:
- Needs an API key for `anthropic`/`openai` providers (`.env`). The `heuristic`
  provider runs offline.
- `--limit N` caps the number of records, useful for a costed pilot before a full
  run.
- Cost scales with records times languages times `queries_per_record`. Start with
  one language and a small `--limit` to estimate.

## 2. Mine hard negatives

```bash
pift mine --config configs/example.yaml \
  --queries data/train_queries.jsonl --out data/triplets.jsonl \
  --miner intfloat/multilingual-e5-small --n-negatives 3
```

Embeds the canonical corpus and each query with the miner model, retrieves the
top ranks, and writes one triplet per query:
`{query_id, query, facet, lang, positive_id, negative_ids}`.

- `--miner` defaults to the config base model. A reasonable choice is the same
  family you will fine-tune.
- The near-duplicate guard (cosine to the positive above 0.95) drops false
  negatives; the count is reported.
- Set `--n-negatives` to match `training.n_negatives` in the config.

## 3. Fine-tune

```bash
pift finetune --config configs/example.yaml \
  --triplets data/triplets.jsonl --out models/my-encoder
```

Contrastive fine-tuning with on-the-fly field-order permutation and dropout.
Saves a SentenceTransformer directory plus `pift_config.json` (records the
prefixes). Override config defaults with `--base`, `--loss`, `--epochs`,
`--batch-size`.

- `--loss cgist` requires `guide_model` in the config.
- On CUDA out-of-memory, lower `training.mini_batch_size` in the config rather
  than the batch size.
- Training is the only stage that benefits substantially from a GPU.

## 4. Evaluate

```bash
pift evaluate --config configs/example.yaml \
  --queries data/eval_queries.jsonl --model models/my-encoder
```

Reports held-out Recall@k, MRR, and nDCG@k against the full corpus, then the
**order-robustness test**: it rebuilds the index under a different fixed field
order and re-evaluates. The printed `order-change delta (nDCG@10)` should be near
zero for a permutation-invariant model. Run the same command with
`--model <base_hf_id>` (or omit `--model` to use the config base) to see the
fragility of the un-fine-tuned or non-permutation-trained baseline for contrast.

Use `--no-robustness` to skip the second pass.

**Graded LLM judge (optional).** Add `--judge` to also score the top-k retrieved
records on a 0-3 relevance rubric. This captures usefulness that the single
labeled positive misses, which matters when the corpus has near-duplicates.

```bash
pift evaluate --config configs/example.yaml --queries data/eval_queries.jsonl \
  --model models/my-encoder --judge --judge-provider anthropic --judge-model claude-haiku-4-5
```

Judge providers: `anthropic`, `openai`, or `heuristic` (offline, no key, a
lexical proxy for demos). Verdicts are cached at `--judge-cache`
(`data/judge_cache.json` by default) and keyed by (query, record, judge model),
so re-running or adding a model reuses prior judgements.

## 4b. Benchmark several models

```bash
pift benchmark --config configs/example.yaml --queries data/eval_queries.jsonl \
  --models base models/my-encoder some-org/another-encoder \
  --judge --judge-provider anthropic
```

Evaluates every listed model on the same queries and corpus and prints a
leaderboard sorted by nDCG@10, with Recall@k, MRR, the order-robustness delta,
and (with `--judge`) the graded score. The best value in each column is bolded in
the Markdown output. Results are written to `--out` (default `data/benchmark/`)
as `leaderboard.md` and `results.json`.

- Use `base` in the model list as shorthand for the config's base model, so you
  can compare a fine-tune against its own starting point.
- The corpus is serialized once and the judge cache is shared across all models,
  so comparing N models costs roughly one model's worth of judge calls plus N
  encodings.

## 5. Search / serve

```bash
pift search --config configs/example.yaml --model models/my-encoder
```

Builds an in-memory index and answers free-text queries interactively. For
embedding this in an application or scaling past a laptop-sized corpus, see
[deployment.md](deployment.md).

## End-to-end, offline

`pift demo` chains generation (heuristic provider) so you can validate the whole
flow without an API key, then prints the mine/finetune/evaluate commands to
continue. This is also the basis of the smoke test in `tests/`.
