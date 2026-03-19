"""
02_generate_embeddings.py
=========================
Encode documents using a sentence-transformers model and export:
  - raw_embeddings.npy  (float32 [N, D]) — intermediate, used by 03_build_index.py
  - flat/embeddings.int8.json            — final flat-search format (for collections <= flat_threshold)

Supports optional Matryoshka dimension truncation (--matryoshka_dim) for models
explicitly trained with Matryoshka Representation Learning. Do NOT set this flag
for models that do not support it (e.g. GIST-small-Embedding-v0).

Usage:
  # From existing metadata.json produced by 01_fetch_and_prepare.py
  python 02_generate_embeddings.py \
    --metadata_path=../../data/prwp/metadata.json \
    --output_dir=../../data/prwp \
    --model=avsolatorio/GIST-small-Embedding-v0

  # From existing float32 embeddings (skip re-encoding)
  python 02_generate_embeddings.py \\
    --embeddings_path=data/avsolatorio__GIST-small-Embedding-v0__004__doc_embeddings.json \\
    --output_dir=data/prwp \\
    --id_field=idno \\
    --content_field=abstract \\
    --title_field=title \\
    --preview_fields=idno,title,abstract,type,doi

  # With Matryoshka truncation (only for MRL-trained models)
  python 02_generate_embeddings.py \\
    --metadata_path=data/prwp/metadata.json \\
    --output_dir=data/prwp \\
    --model=nomic-ai/nomic-embed-text-v1.5 \\
    --matryoshka_dim=128
"""

import json
import os
import numpy as np
import fire
from pathlib import Path


def quantize_sq8(vec: np.ndarray) -> dict:
    """Apply per-vector symmetric int8 scalar quantization (SQ8).

    Compresses a float32 vector to int8 by dividing by the max absolute value
    scaled to the int8 range [-127, 127]. Achieves 4× size reduction with
    cosine similarity error typically < 0.001.

    Args:
        vec: 1-D float32 numpy array to quantize.

    Returns:
        Dict with keys:
            ``scale``: float multiplier to recover approximate float32 values.
            ``qv``: list of int8 integers.
    """
    max_abs = float(np.max(np.abs(vec)))
    if max_abs < 1e-9:
        return {"scale": 1.0, "qv": [0] * len(vec)}
    scale = max_abs / 127.0
    qv = np.round(vec / scale).clip(-127, 127).astype(np.int8).tolist()
    return {"scale": scale, "qv": qv}


def dequantize_sq8(scale: float, qv: list[int]) -> np.ndarray:
    """Reconstruct an approximate float32 vector from SQ8 quantized data.

    Args:
        scale: The scale factor stored alongside the quantized vector.
        qv: List of int8 quantized values.

    Returns:
        Approximate float32 numpy array.
    """
    return np.array(qv, dtype=np.float32) * scale


def l2_normalize(embs: np.ndarray) -> np.ndarray:
    """L2-normalize a batch of embedding vectors row-wise.

    Vectors with near-zero norm are left unchanged (norm replaced by 1.0 to
    avoid division by zero).

    Args:
        embs: 2-D float32 array of shape [N, D].

    Returns:
        Row-normalized float32 array of the same shape.
    """
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return embs / norms


def validate_quantization(
    original: np.ndarray,
    quantized_items: list[dict],
    sample_size: int = 500,
) -> float:
    """Validate that int8 round-trip preserves cosine similarity.

    Samples up to ``sample_size`` vectors, reconstructs them from their
    quantized representations, and measures cosine similarity against the
    original normalized vectors.

    Args:
        original: L2-normalized float32 array of shape [N, D].
        quantized_items: List of SQ8 dicts (``scale``, ``qv``) parallel to
            ``original``.
        sample_size: Number of random vectors to sample for validation.

    Returns:
        Minimum cosine similarity across the sample (should be > 0.999).

    Raises:
        AssertionError: If the minimum cosine similarity falls below 0.98.
    """
    n = min(sample_size, len(original))
    idx = np.random.choice(len(original), n, replace=False)

    reconstructed = np.array(
        [
            dequantize_sq8(quantized_items[i]["scale"], quantized_items[i]["qv"])
            for i in idx
        ],
        dtype=np.float32,
    )
    reconstructed = l2_normalize(reconstructed)

    orig_sample = original[idx]

    sims = np.sum(
        orig_sample * reconstructed, axis=1
    )  # dot product of normalized = cosine sim
    min_sim = float(np.min(sims))
    mean_sim = float(np.mean(sims))
    print(
        f"  Quantization quality — min cosine sim: {min_sim:.6f}, mean: {mean_sim:.6f}"
    )
    assert min_sim > 0.98, (
        f"Quantization quality too low: min cosine sim = {min_sim:.6f} (threshold: 0.98)"
    )
    return min_sim


def main(
    # Input: either metadata_path (raw docs) or embeddings_path (pre-computed)
    metadata_path: str | None = None,
    embeddings_path: str | None = None,
    # Output
    output_dir: str = "data/collection",
    # Model (only needed if generating embeddings from metadata_path)
    model: str = "avsolatorio/GIST-small-Embedding-v0",
    batch_size: int = 64,
    # Field names (for embeddings_path or metadata_path)
    id_field: str = "idno",
    content_field: str = "abstract",
    title_field: str = "title",
    embedding_field: str = "embedding",
    preview_fields: str = "idno,title,abstract,type,doi",  # comma-separated
    # Optional Matryoshka truncation — ONLY for models trained with MRL
    matryoshka_dim: int | None = None,
    # Flat format options
    bm25_text_field: str = "abstract",  # field to inline in flat format for BM25
    bm25_title_field: str = "title",
    seed: int = 42,
) -> None:
    """Generate or load embeddings, quantize, and export the flat search index.

    Produces two outputs in ``output_dir``:
      - ``raw_embeddings.npy`` — L2-normalized float32 array for 03_build_index.py
      - ``flat/embeddings.int8.json`` — compact SQ8-quantized flat search index

    Args:
        metadata_path: Path to ``metadata.json`` from step 01. Triggers
            on-the-fly encoding with ``model``. Mutually exclusive with
            ``embeddings_path``.
        embeddings_path: Path to a pre-computed JSON embeddings file. Skips
            the sentence-transformers encoding step.
        output_dir: Directory where outputs are written.
        model: HuggingFace model ID used when ``metadata_path`` is provided.
        batch_size: Encoding batch size for sentence-transformers.
        id_field: Field name for the document identifier.
        content_field: Field name for the main text content used for encoding.
        title_field: Field name for the document title.
        embedding_field: Key holding the raw embedding list when loading from
            ``embeddings_path``.
        preview_fields: Comma-separated field names to include in the flat
            index for display.
        matryoshka_dim: If set, truncates embeddings to this dimensionality
            after L2 normalization. Only valid for MRL-trained models.
        bm25_text_field: Field to store as the ``text`` value in flat index
            entries (used for BM25 search in the browser).
        bm25_title_field: Field to store as the ``title`` value in flat index
            entries.
        seed: Random seed for reproducible quantization validation sampling.

    Raises:
        ValueError: If neither ``metadata_path`` nor ``embeddings_path`` is
            provided.
        AssertionError: If quantization quality falls below the 0.98 threshold.
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    (output_path / "flat").mkdir(parents=True, exist_ok=True)

    # fire may parse comma-separated strings as tuples
    if isinstance(preview_fields, (list, tuple)):
        preview_field_list = [str(f).strip() for f in preview_fields]
    else:
        preview_field_list = [f.strip() for f in str(preview_fields).split(",")]

    # ── 1. Load or generate embeddings ──────────────────────────────────────
    if embeddings_path:
        print(f"Loading pre-computed embeddings from: {embeddings_path}")
        with open(embeddings_path, "r") as f:
            raw_data = json.load(f)

        metadata: list[dict] = []
        embeddings_list: list[list[float]] = []
        for item in raw_data:
            embeddings_list.append(item[embedding_field])
            meta = {
                field: item.get(field) for field in preview_field_list if field in item
            }
            meta["id"] = str(item.get(id_field, item.get("id", "")))
            if "idno" in meta:
                meta["idno"] = str(meta["idno"])
            meta["title"] = item.get(title_field, "")
            meta["text"] = item.get(bm25_text_field, item.get(content_field, ""))
            metadata.append(meta)

        embeddings = np.array(embeddings_list, dtype=np.float32)
        print(f"  Loaded {len(metadata)} items, dim={embeddings.shape[1]}")

    elif metadata_path:
        print(f"Generating embeddings from: {metadata_path}")
        from sentence_transformers import SentenceTransformer

        with open(metadata_path, "r") as f:
            docs = json.load(f)

        model_obj = SentenceTransformer(model)
        texts = [
            (doc.get(title_field, "") or "")
            + "\n\n"
            + (doc.get(content_field, "") or "")
            for doc in docs
        ]
        print(f"  Encoding {len(texts)} documents with {model}...")
        emb_array = model_obj.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
        embeddings = np.array(emb_array, dtype=np.float32)

        metadata = []
        for doc in docs:
            raw_id = doc.get(id_field, doc.get("id", ""))
            meta = {
                field: doc.get(field) for field in preview_field_list if field in doc
            }
            meta["id"] = str(raw_id)
            if "idno" in meta:
                meta["idno"] = str(
                    meta["idno"]
                )  # normalize to string (match semantic-search shape)
            meta["title"] = doc.get(title_field, "")
            meta["text"] = doc.get(bm25_text_field, doc.get(content_field, ""))
            metadata.append(meta)

        print(f"  Generated embeddings: shape={embeddings.shape}")
    else:
        raise ValueError("Must provide either --metadata_path or --embeddings_path")

    n_items, full_dim = embeddings.shape

    # ── 2. L2-normalize (for cosine similarity via dot product) ─────────────
    embeddings_norm = l2_normalize(embeddings)

    # ── 3. Optional Matryoshka truncation ───────────────────────────────────
    active_dim = full_dim
    if matryoshka_dim is not None:
        if matryoshka_dim >= full_dim:
            print(
                f"  matryoshka_dim={matryoshka_dim} >= full_dim={full_dim}, skipping truncation"
            )
        else:
            print(f"  Applying Matryoshka truncation: {full_dim}D → {matryoshka_dim}D")
            # Validate truncation quality on a sample before committing
            sample_idx = np.random.choice(n_items, min(200, n_items), replace=False)
            full_sample = embeddings_norm[sample_idx]
            trunc_sample = l2_normalize(embeddings_norm[sample_idx, :matryoshka_dim])
            sims = np.sum(full_sample[:, :matryoshka_dim] * trunc_sample, axis=1)
            # For MRL models, truncated cosine sim should be high (>0.90)
            min_trunc_sim = float(np.min(sims))
            print(
                f"  Truncation quality — min cosine sim (full vs truncated): {min_trunc_sim:.4f}"
            )
            if min_trunc_sim < 0.85:
                print(
                    f"  WARNING: Low truncation similarity ({min_trunc_sim:.4f}). "
                    f"This model may not support Matryoshka truncation well."
                )

            embeddings_norm = l2_normalize(embeddings_norm[:, :matryoshka_dim])
            active_dim = matryoshka_dim

    print(f"  Active embedding dim: {active_dim}")

    # ── 4. SQ8 quantization ─────────────────────────────────────────────────
    print("Applying SQ8 int8 quantization...")
    quantized = [quantize_sq8(embeddings_norm[i]) for i in range(n_items)]
    print(f"  Quantized {n_items} vectors")

    # ── 5. Validate quantization quality ────────────────────────────────────
    print("Validating quantization quality...")
    validate_quantization(embeddings_norm, quantized)

    # ── 6. Save raw normalized embeddings (for 03_build_index.py) ───────────
    npy_path = output_path / "raw_embeddings.npy"
    np.save(str(npy_path), embeddings_norm)
    print(f"Saved raw normalized embeddings → {npy_path}")

    # Save metadata for index building
    meta_path = output_path / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"Saved metadata ({n_items} items) → {meta_path}")

    # ── 7. Export flat/embeddings.int8.json ─────────────────────────────────
    flat_items: list[dict] = []
    for i, (meta, q) in enumerate(zip(metadata, quantized)):
        item: dict = {
            "id": meta["id"],
            "scale": q["scale"],
            "qv": q["qv"],
            "title": meta.get("title", ""),
            "text": meta.get("text", ""),
        }
        # Include any extra preview fields
        for field in preview_field_list:
            if field not in ("id", "title", "text") and field in meta:
                item[field] = meta[field]
        flat_items.append(item)

    flat_data = {
        "format": "int8_flat",
        "n_items": n_items,
        "dim": active_dim,
        "full_dim": full_dim,
        "matryoshka_dim": matryoshka_dim,
        "model_id": model if not embeddings_path else "unknown",
        "items": flat_items,
    }
    flat_path = output_path / "flat" / "embeddings.int8.json"
    with open(flat_path, "w") as f:
        json.dump(flat_data, f, ensure_ascii=False, separators=(",", ":"))

    flat_size_mb = flat_path.stat().st_size / 1e6
    print(
        f"Saved flat index ({n_items} items, {active_dim}D int8) → {flat_path} ({flat_size_mb:.1f} MB)"
    )
    print("\nDone! Next: run 03_build_index.py to build the sharded HNSW index.")


if __name__ == "__main__":
    fire.Fire(main)
