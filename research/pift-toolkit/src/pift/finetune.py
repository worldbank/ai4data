"""
Permutation-invariant fine-tuning.

Training examples are (query, positive, negative...) tuples. The documents are
serialized on the fly with a fresh field-order permutation and field dropout on
every access, so every epoch presents new orderings and the encoder learns to
read the field labels rather than their positions. This is the "two lines in the
data loader" that the method comes down to (see ``serialize.render_segments``);
everything else here is standard contrastive fine-tuning.

Losses:
  - ``cmnrl`` (default): CachedMultipleNegativesRankingLoss, unguided. The
    cached (GradCache) form decouples the negative pool (the batch) from memory
    (the mini-batch), so models of different sizes train against the same
    negatives.
  - ``cgist``: CachedGISTEmbedLoss, guided. A guide model masks in-batch
    false negatives, which matters when the corpus has many near-duplicates.

At the end the model is saved together with ``pift_config.json``, which records
the query/document prefixes so evaluation and serving apply them automatically.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .config import Config
from .encoder import pick_device, resolve_model
from .records import load_records
from .serialize import build_segments, render_segments


def _make_transform(config: Config, segs_by_id: dict, field_dropout: float,
                    query_prefix: str, doc_prefix: str):
    """On-the-fly serialization: a fresh permutation per access.

    Cells are payloads: ``anchor`` is the raw query; ``positive`` is
    ``"facet||record_id"``; ``negative_k`` is ``"||record_id"``. The transform
    decodes whatever columns the trainer requests.
    """
    def render_doc(cell: str, rng: random.Random) -> str:
        facet, _, rid = cell.partition("||")
        return doc_prefix + render_segments(
            segs_by_id[rid], config, rng=rng, permute=True,
            field_dropout=field_dropout, protect=config.protect_for_facet(facet or None),
        )

    def transform(batch):
        rng = random.Random()
        out = {}
        for key, values in batch.items():
            if key == "anchor":
                out[key] = [query_prefix + v for v in values]
            else:
                out[key] = [render_doc(v, rng) for v in values]
        return out

    return transform


def finetune(config: Config, triplets_path: str, output: str,
             base_model: str | None = None, **overrides) -> str:
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer import losses

    t = dict(config.training)
    t.update({k: v for k, v in overrides.items() if v is not None})

    model_id, qp, dp = resolve_model(base_model, config)
    records = load_records(config)
    segs_by_id = {rid: build_segments(r, config) for rid, r in records.items()}

    triplets = [json.loads(l) for l in Path(triplets_path).read_text().splitlines() if l.strip()]
    n_neg = t["n_negatives"]
    available = min((len(x["negative_ids"]) for x in triplets), default=0)
    if available < n_neg:
        print(f"[finetune] triplets carry {available} negatives but training expects "
              f"{n_neg}; using {available}. Re-mine with more to match the config.")
        n_neg = available
    if n_neg < 1:
        raise ValueError("triplets have no negatives; re-run `pift mine`")
    usable = [x for x in triplets if len(x["negative_ids"]) >= n_neg]
    cols = {
        "anchor": [x["query"] for x in usable],
        "positive": [f"{x.get('facet') or ''}||{x['positive_id']}" for x in usable],
    }
    for k in range(n_neg):
        cols[f"negative_{k+1}"] = [f"||{x['negative_ids'][k]}" for x in usable]
    dataset = Dataset.from_dict(cols).shuffle(seed=42)
    dataset.set_transform(_make_transform(
        config, segs_by_id, t["field_dropout"], qp, dp))
    print(f"[finetune] {len(dataset):,} rows, dynamic per-access permutation; "
          f"base={model_id} loss={t['loss']}")

    device = pick_device(overrides.get("device"))
    model = SentenceTransformer(model_id, device=device)
    if t["max_seq_length"]:
        model.max_seq_length = t["max_seq_length"]

    loss_name = t["loss"]
    if loss_name == "cgist":
        guide_id = t["guide_model"]
        if not guide_id:
            raise ValueError("loss 'cgist' requires training.guide_model in the config")
        guide = SentenceTransformer(guide_id, device=device)
        guide.eval()
        loss = losses.CachedGISTEmbedLoss(model, guide, mini_batch_size=t["mini_batch_size"])
        print(f"[finetune] guided (CachedGIST) with {guide_id}")
    elif loss_name == "cmnrl":
        loss = losses.CachedMultipleNegativesRankingLoss(
            model, mini_batch_size=t["mini_batch_size"])
    else:
        raise ValueError(f"unsupported loss {loss_name!r} (use cmnrl or cgist)")

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    workers = 8 if device == "cuda" else 0
    dl_extra = {"dataloader_persistent_workers": True,
                "dataloader_prefetch_factor": 4} if workers else {}

    args = SentenceTransformerTrainingArguments(
        output_dir=output + "/checkpoints",
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        learning_rate=t["lr"],
        warmup_steps=0.1,
        use_cpu=(device == "cpu"),
        bf16=use_bf16,
        dataloader_num_workers=workers,
        logging_steps=25,
        report_to=[],
        seed=42,
        save_strategy="no",
        **dl_extra,
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=dataset, loss=loss)
    trainer.train()

    Path(output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    (Path(output) / "pift_config.json").write_text(json.dumps({
        "base_model": model_id,
        "query_prefix": qp or None,
        "doc_prefix": dp or None,
        "trained_with": {k: t[k] for k in
                         ("loss", "guide_model", "epochs", "batch_size",
                          "lr", "n_negatives", "max_seq_length", "field_dropout")},
    }, indent=2))
    print(f"[finetune] saved -> {output}")
    return output
