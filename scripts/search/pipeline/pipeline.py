"""
pipeline.py — End-to-end CLI orchestrator
==========================================
Runs all three pipeline steps in order:
  01_fetch_and_prepare.py  → metadata.json
  02_generate_embeddings.py → raw_embeddings.npy + flat/embeddings.int8.json
  03_build_index.py         → manifest.json + index/

Usage:
  # PRWP (full run, fetch from API)
  python pipeline.py prwp \
    --source=worldbank_api \
    --doctype="Policy Research Working Paper" \
    --model=avsolatorio/GIST-small-Embedding-v0 \
    --output_dir=../../data/prwp

  # WDI from local Excel
  python pipeline.py wdi \\
    --source=excel \\
    --input_file=../../data/WDIEXCEL.xlsx \\
    --sheet_name=Series \\
    --id_field="Series Code" \\
    --content_fields="Indicator Name,Long definition,Short definition" \\
    --preview_fields="Series Code,Indicator Name,Long definition,Short definition,Source" \\
    --model=avsolatorio/GIST-small-Embedding-v0 \\
    --output_dir=../../data/wdi \\
    --flat_threshold=2000

  # Skip fetch (use existing metadata.json)
  python pipeline.py prwp --skip_fetch \\
    --model=avsolatorio/GIST-small-Embedding-v0 \\
    --output_dir=../../data/prwp

  # Skip fetch + embedding (use existing raw_embeddings.npy)
  python pipeline.py prwp --skip_fetch --skip_embed \\
    --output_dir=../../data/prwp

  # With Matryoshka truncation (MRL-trained models only)
  python pipeline.py my_collection \\
    --source=json \\
    --input_file=my_docs.json \\
    --model=nomic-ai/nomic-embed-text-v1.5 \\
    --matryoshka_dim=128 \\
    --output_dir=../../data/my_collection
"""

import sys

import fire
import importlib.util
from pathlib import Path


def _load_module(module_name: str, file_path: Path) -> object:
    """Load a Python source file as a module using importlib.

    Uses the modern ``importlib.util`` pattern instead of the deprecated
    ``SourceFileLoader.load_module()`` API.

    Args:
        module_name: Logical module name used for the module's ``__name__``
            attribute (does not need to correspond to a package).
        file_path: Absolute or relative path to the ``.py`` source file.

    Returns:
        The loaded module object with all its top-level names available as
        attributes.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        Exception: Any exception raised during execution of the module.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(
    collection_id: str,
    # Step 1: fetch / prepare
    source: str = "worldbank_api",
    doctype: str = "Policy Research Working Paper",
    input_file: str | None = None,
    sheet_name: str | None = None,
    id_field: str = "idno",
    content_fields: str = "title,abstract",
    preview_fields: str = "idno,title,abstract,type,doi,url,date_published",
    bm25_fields: str = "title,text",
    max_docs: int = 0,
    # Step 2: embedding
    model: str = "avsolatorio/GIST-small-Embedding-v0",
    batch_size: int = 64,
    matryoshka_dim: int | None = None,
    title_field: str = "title",
    bm25_text_field: str = "abstract",
    # Step 3: index building
    hnsw_M: int = 16,
    ef_construction: int = 200,
    flat_threshold: int = 2000,
    n_clusters: int | None = None,
    kmeans_niter: int = 30,
    compress: str = "gzip",
    # Control
    output_dir: str | None = None,
    skip_fetch: bool = False,
    skip_embed: bool = False,
    seed: int = 42,
) -> None:
    """Run the full search index build pipeline for a named collection.

    Orchestrates the three pipeline steps in order:
      1. **Fetch & prepare** — download or load documents → ``metadata.json``
      2. **Generate embeddings** — encode and quantize → ``raw_embeddings.npy``
         and ``flat/embeddings.int8.json``
      3. **Build index** — cluster, HNSW, shard export → ``index/`` and
         ``manifest.json``

    Individual steps can be skipped with ``--skip_fetch`` or ``--skip_embed``
    when intermediate outputs already exist on disk.

    Args:
        collection_id: Short identifier for the collection (e.g. ``"prwp"``).
            Used as the default output directory suffix and stored in the
            manifest.
        source: Data source for step 1. One of ``worldbank_api``, ``excel``,
            or ``json``.
        doctype: Document type for World Bank API fetching.
        input_file: Input file path for ``excel`` or ``json`` sources.
        sheet_name: Sheet name for Excel inputs.
        id_field: Field name used as the document identifier.
        content_fields: Comma-separated field names merged into the embedding
            content text.
        preview_fields: Comma-separated field names included in display records.
        bm25_fields: Comma-separated field names stored in the manifest for
            browser BM25 search.
        max_docs: Maximum documents to fetch from the API (0 = all).
        model: HuggingFace sentence-transformers model ID for encoding.
        batch_size: Encoding batch size.
        matryoshka_dim: Optional Matryoshka truncation dimension (MRL models
            only).
        title_field: Field name for document title.
        bm25_text_field: Field name for BM25 text content (typically the
            abstract).
        hnsw_M: FAISS HNSW M parameter.
        ef_construction: FAISS HNSW efConstruction parameter.
        flat_threshold: Collections at or below this size use flat search mode.
        n_clusters: Number of K-Means clusters for sharding. None = auto.
        kmeans_niter: Number of K-Means training iterations.
        compress: ``"gzip"`` (default) or ``"none"`` for GitHub Pages.
        output_dir: Override the default output path (``../../data/{collection_id}``).
        skip_fetch: If True, skip step 1 and use an existing ``metadata.json``.
        skip_embed: If True, skip step 2 and use an existing
            ``raw_embeddings.npy``.
        seed: Random seed passed to all steps.
    """
    if output_dir is None:
        # Default: repo root / data / {collection_id} (stable regardless of cwd)
        repo_root = Path(__file__).resolve().parent.parent.parent
        output_path = repo_root / "data" / collection_id
    else:
        output_path = Path(output_dir)
    print(f"\n{'=' * 60}")
    print(f"Pipeline: {collection_id}")
    print(f"Output:   {output_path.resolve()}")
    print(f"{'=' * 60}\n")

    pipeline_dir = Path(__file__).parent

    # ── Step 1: Fetch & prepare ──────────────────────────────────────────────
    if not skip_fetch:
        print("Step 1: Fetching and preparing documents...")
        fetch_mod = _load_module("fetch", pipeline_dir / "01_fetch_and_prepare.py")
        fetch_mod.main(
            source=source,
            output_dir=str(output_path),
            doctype=doctype,
            max_docs=max_docs,
            input_file=input_file,
            sheet_name=sheet_name,
            id_field=id_field,
            content_fields=content_fields,
            preview_fields=preview_fields,
        )
    else:
        print("Step 1: Skipped (--skip_fetch)")

    # ── Step 2: Generate embeddings ──────────────────────────────────────────
    if not skip_embed:
        print("\nStep 2: Generating embeddings and quantizing...")
        embed_mod = _load_module("embed", pipeline_dir / "02_generate_embeddings.py")
        meta_path = str(output_path / "metadata.json")
        embed_mod.main(
            metadata_path=meta_path,
            output_dir=str(output_path),
            model=model,
            batch_size=batch_size,
            id_field=id_field,
            content_field=content_fields.split(",")[1]
            if "," in content_fields
            else content_fields,
            title_field=title_field,
            preview_fields=preview_fields,
            matryoshka_dim=matryoshka_dim,
            bm25_text_field=bm25_text_field,
            seed=seed,
        )
    else:
        print("Step 2: Skipped (--skip_embed)")

    # ── Step 3: Build index ──────────────────────────────────────────────────
    print("\nStep 3: Building search index...")
    index_mod = _load_module("index", pipeline_dir / "03_build_index.py")
    try:
        index_mod.main(
            output_dir=str(output_path),
            collection_id=collection_id,
            model_id=model,
            hnsw_M=hnsw_M,
            ef_construction=ef_construction,
            flat_threshold=flat_threshold,
            n_clusters=n_clusters,
            kmeans_niter=kmeans_niter,
            preview_fields=preview_fields,
            bm25_fields=bm25_fields,
            compress=compress,
            seed=seed,
        )
    except Exception as e:
        print(f"\nStep 3 failed: {e}", file=sys.stderr)
        raise

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete! Index ready at: {output_path.resolve()}")
    print(f"Load manifest at: {(output_path / 'manifest.json').resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    fire.Fire(main)
