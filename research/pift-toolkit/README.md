# PI-FT Toolkit

A reusable, configuration-driven pipeline for **permutation-invariant fine-tuning
(PI-FT)** of embedding models for **structured-metadata retrieval**.

It packages the method from the paper [*Field Order Should Not Matter:
Permutation-Invariant Embedding Model Fine-Tuning for Structured Metadata
Retrieval*](https://arxiv.org/abs/2606.30473) (Solatorio, Dupriez, Macalaba) into a tool any organization can point
at its own catalogue. You bring a set of structured records (each a small schema
of labeled fields) and a YAML config; the toolkit generates training
supervision, mines hard negatives, fine-tunes a small open encoder, evaluates it
(including an order-robustness test), and serves search.

The whole thing is driven by **one config file**. Adapting it to a new catalogue
means editing `configs/example.yaml`, not the code.

## Why this exists

When an AI assistant answers a question over a data catalogue, it first
*retrieves* a record and then reasons over it. If retrieval returns the wrong
record, the answer is wrong no matter how good the model is. Two problems make
that retrieval step fragile:

1. **Field order.** To embed a structured record you must flatten its fields
   into one string, which forces a field order. Once a model is fine-tuned it
   can learn to rely on a field's *position* instead of its *label*, so rebuilding
   the index under a different order silently degrades quality. PI-FT removes this
   by serializing each record under a freshly shuffled field order during
   training.

2. **No usage logs for the long tail.** Click logs only cover what users already
   find. The toolkit instead *generates* grounded, facet-targeted queries with an
   LLM, giving full coverage of every record and facet, in any language you ask
   for.

The result, in the paper's setting, is a 118M-parameter encoder that runs on a
CPU, beats much larger zero-shot baselines, and stays invariant to field order.
See [docs/method.md](docs/method.md).

## Install

```bash
cd pift-toolkit
python -m venv .venv && source .venv/bin/activate

# Core only (config, serialization, metrics, offline demo):
pip install -e .

# Full pipeline (training, mining, evaluation, serving):
pip install -e ".[train]"

# Plus hosted-LLM query generation:
pip install -e ".[all]"
cp .env.example .env        # add ANTHROPIC_API_KEY or OPENAI_API_KEY
```

## Quickstart (offline, no API key, no GPU)

The sample catalogue in `examples/` lets you exercise the wiring immediately. The
`heuristic` generator builds queries from the record fields, so no LLM is needed.

```bash
# 1. See how records serialize, canonical vs. augmented (permuted + dropped):
pift preview --config configs/example.yaml --n 2

# 2. Generate queries offline (heuristic provider), then run the chain:
pift demo     --config configs/example.yaml --out data/demo
pift mine     --config configs/example.yaml --queries data/demo/train_queries.jsonl \
              --out data/demo/triplets.jsonl --n-negatives 2 --device cpu
pift finetune --config configs/example.yaml --triplets data/demo/triplets.jsonl \
              --out data/demo/model --epochs 2 --batch-size 8 --device cpu
pift evaluate --config configs/example.yaml --queries data/demo/train_queries.jsonl \
              --model data/demo/model --device cpu
pift search   --config configs/example.yaml --model data/demo/model
```

(The sample corpus is tiny, so its metrics saturate; the point is to confirm the
pipeline runs before you point it at real data and a real LLM.)

## The pipeline

```
 records.jsonl ─┐
   (+ config)   │
                ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ generate │ LLM writes grounded, facet-targeted queries        │  pift generate
   │          │  per record, per language  →  *_queries.jsonl       │
   ├──────────┼──────────────────────────────────────────────────┤
   │ mine     │ embed corpus, retrieve hard negatives,             │  pift mine
   │          │  drop near-duplicates of the positive → triplets   │
   ├──────────┼──────────────────────────────────────────────────┤
   │ finetune │ contrastive training with on-the-fly field-order   │  pift finetune
   │          │  permutation + dropout (cmnrl / cgist) → model dir │
   ├──────────┼──────────────────────────────────────────────────┤
   │ evaluate │ held-out Recall@k / MRR / nDCG@k, order-           │  pift evaluate
   │          │  robustness, + optional graded LLM-judge score     │
   ├──────────┼──────────────────────────────────────────────────┤
   │ benchmark│ compare several models side by side (leaderboard)  │  pift benchmark
   ├──────────┼──────────────────────────────────────────────────┤
   │ search   │ build an index and answer free-text queries        │  pift search
   └──────────┴──────────────────────────────────────────────────┘
```

### Grading and comparison

`evaluate` and `benchmark` can score the top-k retrieved records with an
**LLM-as-a-judge** on a 0-3 relevance rubric (`--judge`). The single labeled
positive per query understates usefulness in a near-duplicate-rich catalogue, so
the graded score distinguishes systems that the binary metrics call tied.
Relevance is a property of the (query, record) pair, so judgements are cached on
disk and shared across every compared model, which keeps multi-model comparison
affordable.

`benchmark` evaluates several models on the same held-out set and writes a
leaderboard (Recall@k, MRR, nDCG@10, order-robustness delta, optional judged
score) to the console, `leaderboard.md`, and `results.json`:

```bash
pift benchmark --config configs/example.yaml --queries data/eval_queries.jsonl \
  --models base models/my-encoder another-org/some-encoder \
  --judge --judge-provider anthropic
```

(`base` is shorthand for the config's base model, so you can compare a
fine-tune against its own starting point.)

## Adapting to your catalogue

1. Put your records in a JSON-Lines file (one record per line) or a directory of
   per-record JSON files. Each record needs a unique id.
2. Copy `configs/example.yaml`, set `catalogue.id_field` and the records path,
   and list your `fields` with a `role` each (`protected` / `fixed` / `elastic`).
3. Set `base_model.hf_id` to the encoder you want to fine-tune (and its prefixes
   if it uses them, e.g. E5).
4. Run `pift preview` to sanity-check serialization, then `generate → mine →
   finetune → evaluate`.

Full field reference: [docs/configuration.md](docs/configuration.md).
Stage-by-stage guide: [docs/pipeline.md](docs/pipeline.md).
Production serving: [docs/deployment.md](docs/deployment.md).

## What's in here

| Path | Purpose |
|---|---|
| `src/pift/serialize.py` | Schema-aware serialization, field permutation, dropout (the method core) |
| `src/pift/config.py` | Loads and validates the catalogue YAML |
| `src/pift/generate.py` | LLM query generation (Anthropic / OpenAI / offline heuristic) |
| `src/pift/mine.py` | Hard-negative mining with the near-duplicate guard |
| `src/pift/finetune.py` | Permutation-invariant contrastive fine-tuning (cmnrl / cgist) |
| `src/pift/evaluate.py` | Retrieval metrics + the order-robustness test |
| `src/pift/judge.py` | LLM-as-a-judge: graded 0-3 relevance with a shared on-disk cache |
| `src/pift/benchmark.py` | Multi-model comparison / leaderboard |
| `src/pift/search.py` | In-memory index and query (and `pift search`) |
| `src/pift/cli.py` | The `pift` command |
| `configs/example.yaml` | A complete worked config for the sample catalogue |
| `examples/sample_catalogue/` | A small synthetic catalogue to run end to end |
| `docs/` | Method, configuration, pipeline, and deployment guides |

## Citation

If you use this toolkit or the PI-FT method, please cite:

```bibtex
@article{solatorio2026fieldorder,
  title={Field Order Should Not Matter: Permutation-Invariant Embedding Model Fine-Tuning for Structured Metadata Retrieval},
  author={Solatorio, Aivin V. and Dupriez, Olivier and Macalaba, Rafael},
  journal={arXiv preprint arXiv:2606.30473},
  year={2026},
  url={https://arxiv.org/abs/2606.30473},
  eprint={2606.30473},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

Paper: [https://arxiv.org/abs/2606.30473](https://arxiv.org/abs/2606.30473)

