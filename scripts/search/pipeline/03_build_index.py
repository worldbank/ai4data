"""
03_build_index.py
=================
Build a browser-compatible search index from normalized embeddings.

For collections with n_items <= flat_threshold: emits only a manifest pointing
to the flat index already built by 02_generate_embeddings.py.

For collections with n_items > flat_threshold: builds a sharded HNSW index:

  1. K-Means clustering (FAISS) — assigns each node to a shard.
     Nodes in the same cluster are semantically related, so HNSW neighbors
     tend to fall in the same shard → 2-4 shard fetches per query instead of 10-15.

  2. FAISS IndexHNSWFlat — built on L2-normalized embeddings (inner product = cosine sim).

  3. Shard export — for each K-Means cluster:
       index/layer0/shard_NNN.json  {shard_id, nodes:[{id, scale, qv, neighbors}]}

  4. Upper-layer export — all nodes appearing in layers 1+ (tiny, ~5% of N):
       index/upper_layers.json  {max_layer, entry_node_id, nodes:{id:{max_layer,scale,qv,layers}}}

  5. Lookup tables:
       index/node_to_shard.json    {node_id: shard_id}
       index/cluster_centroids.json  [{shard_id, scale, qv}]
       index/config.json

  6. bm25_corpus.json (new) — text-only corpus for browser-side BM25 search:
       index/bm25_corpus.json  [{id, title, text}, ...]
     Roughly 6 MB uncompressed / 2 MB gzip. Enables BM25 without the 44 MB flat index.

  7. manifest.json (top-level, consumed by search.worker.js)

Output format is designed for lazy browser loading:
  - Upper layers + lookup: fetch once at init (~200-300 KB for PRWP)
  - Layer0 shards: fetch on demand during beam search (~35 KB/shard x 3-4 shards)
  - bm25_corpus.json: fetched once when BM25 mode is activated (~2 MB gzip)

Usage:
  python 03_build_index.py \\
    --output_dir=../../data/prwp \\
    --collection_id=prwp \\
    --model_id=avsolatorio/GIST-small-Embedding-v0

  # Custom HNSW params
  python 03_build_index.py \\
    --output_dir=../../data/prwp \\
    --collection_id=prwp \\
    --hnsw_M=16 \\
    --ef_construction=200 \\
    --n_clusters=110 \\
    --flat_threshold=2000

  # Disable compression (keep uncompressed .json — required for GitHub Pages)
  python 03_build_index.py --output_dir=../../data/prwp --compress=none
"""

import gzip
import json
import sys
import warnings
import numpy as np
import fire
from pathlib import Path

# Suppress resource_tracker "leaked semaphore" at shutdown (FAISS/OpenMP leave one open)
warnings.filterwarnings(
    "ignore",
    message="resource_tracker: There appear to be.*leaked semaphore",
    category=UserWarning,
    module="multiprocessing.resource_tracker",
)


def _compress_json(path: Path, remove_original: bool = True) -> Path:
    """Compress a JSON file to gzip format in-place.

    Args:
        path: Path to the uncompressed .json file.
        remove_original: If True, delete the original .json after compression.

    Returns:
        Path to the resulting .json.gz file.
    """
    gz_path = path.with_suffix(path.suffix + ".gz")
    with open(path, "rb") as f_in:
        with gzip.open(gz_path, "wb", compresslevel=6) as f_out:
            f_out.writelines(f_in)
    if remove_original:
        path.unlink()
    return gz_path


def quantize_sq8(vec: np.ndarray) -> dict:
    """Apply per-vector symmetric int8 scalar quantization (SQ8).

    Args:
        vec: 1-D float32 numpy array to quantize.

    Returns:
        Dict with ``scale`` (float) and ``qv`` (list of int8).
    """
    max_abs = float(np.max(np.abs(vec)))
    if max_abs < 1e-9:
        return {"scale": 1.0, "qv": [0] * len(vec)}
    scale = max_abs / 127.0
    qv = np.round(vec / scale).clip(-127, 127).astype(np.int8).tolist()
    return {"scale": scale, "qv": qv}


def get_hnsw_neighbors(
    hnsw: object,
    offsets: np.ndarray,
    flat_neighbors: np.ndarray,
    node_id: int,
    layer: int,
) -> list[int]:
    """Extract the neighbor list of a FAISS HNSW node at a given layer.

    FAISS stores HNSW neighbors in a flat int array. The layout per node is::

        [layer0_neighbors (nb_neighbors(0) slots),
         layer1_neighbors (nb_neighbors(1) slots),
         layer2_neighbors (nb_neighbors(2) slots), ...]

    ``offsets[node_id]`` gives the start index of this node's block in
    ``flat_neighbors``. ``nb_neighbors(k)`` gives the capacity at layer k.

    Args:
        hnsw: FAISS HNSW object (``index.hnsw``).
        offsets: Flat int64 array of per-node offsets into ``flat_neighbors``.
        flat_neighbors: Flat int32 array of all neighbor IDs; -1 = unused slot.
        node_id: Integer node ID whose neighbors are requested.
        layer: Layer index (0 = base layer).

    Returns:
        List of valid (non-negative) neighbor node IDs at the requested layer.
    """
    # Cumulative slot offset for layers 0..layer-1
    cum = sum(hnsw.nb_neighbors(l) for l in range(layer))
    start = int(offsets[node_id]) + cum
    end = start + hnsw.nb_neighbors(layer)
    nbrs = flat_neighbors[start:end]
    return [int(n) for n in nbrs if n >= 0]  # -1 = unused slot


def validate_recall(
    embeddings_norm: np.ndarray,
    hnsw_index: object,
    n_queries: int = 100,
    topk: int = 10,
) -> float:
    """Compare HNSW approximate top-k results against brute-force exact search.

    Args:
        embeddings_norm: L2-normalized float32 array of shape [N, D].
        hnsw_index: Built FAISS IndexHNSWFlat.
        n_queries: Number of random query vectors to sample.
        topk: Number of nearest neighbors to retrieve per query.

    Returns:
        Mean recall@topk across sampled queries (1.0 = perfect).

    Raises:
        AssertionError: If recall falls below 0.85.
    """
    import faiss

    n = len(embeddings_norm)
    idx = np.random.choice(n, min(n_queries, n), replace=False)
    queries = embeddings_norm[idx].astype(np.float32)

    # HNSW search
    hnsw_index.hnsw.efSearch = 50
    _, hnsw_ids = hnsw_index.search(queries, topk)

    # Brute-force search (flat IP index)
    flat_index = faiss.IndexFlatIP(embeddings_norm.shape[1])
    flat_index.add(embeddings_norm.astype(np.float32))
    _, bf_ids = flat_index.search(queries, topk)

    recalls: list[float] = []
    for h, b in zip(hnsw_ids, bf_ids):
        h_set = set(h[h >= 0])
        b_set = set(b[b >= 0])
        recalls.append(len(h_set & b_set) / len(b_set) if b_set else 1.0)

    recall = float(np.mean(recalls))
    print(f"  HNSW recall@{topk} (ef=50, {len(recalls)} queries): {recall:.4f}")
    assert recall >= 0.85, (
        f"HNSW recall too low: {recall:.4f} (threshold: 0.85). "
        f"Try increasing hnsw_M or ef_construction."
    )
    return recall


def main(
    output_dir: str = "data/collection",
    collection_id: str = "collection",
    model_id: str = "avsolatorio/GIST-small-Embedding-v0",
    # Index params
    hnsw_M: int = 16,
    ef_construction: int = 200,
    flat_threshold: int = 2000,
    # Shard params
    n_clusters: int | None = None,  # None = auto: max(10, sqrt(n_items))
    kmeans_niter: int = 30,
    # Preview fields to include in flat format (already written by 02_generate_embeddings.py)
    preview_fields: str = "idno,title,abstract,type,doi",
    # BM25 fields for manifest
    bm25_fields: str = "title,text",
    # Compression: "gzip" writes .json.gz and removes .json (smaller transfer).
    # Use "none" for GitHub Pages or other static hosts that cannot serve pre-compressed files.
    compress: str = "gzip",
    seed: int = 42,
) -> None:
    """Build and serialize the complete browser search index.

    Reads ``raw_embeddings.npy`` and ``metadata.json`` from ``output_dir``
    (written by 02_generate_embeddings.py) and produces:

    - ``index/upper_layers.json[.gz]`` — HNSW upper-layer graph
    - ``index/layer0/shard_NNN.json[.gz]`` — base-layer shards (one per cluster)
    - ``index/node_to_shard.json[.gz]`` — node-to-shard lookup table
    - ``index/cluster_centroids.json[.gz]`` — quantized cluster centroid vectors
    - ``index/config.json[.gz]`` — index configuration
    - ``index/titles.json[.gz]`` — lightweight display metadata (no text field)
    - ``index/bm25_corpus.json[.gz]`` — text corpus for browser-side BM25 search
    - ``manifest.json`` — top-level manifest consumed by search.worker.js

    For small collections (<= ``flat_threshold`` items), only the manifest is
    written and the flat index from step 02 is used directly.

    Args:
        output_dir: Directory containing ``raw_embeddings.npy`` and
            ``metadata.json``; all outputs are written here.
        collection_id: Short identifier for this collection, stored in manifest.
        model_id: HuggingFace model ID, stored in manifest for reference.
        hnsw_M: FAISS HNSW M parameter (number of bidirectional links per node).
        ef_construction: FAISS HNSW efConstruction parameter.
        flat_threshold: Collections with this many items or fewer use flat
            search mode; larger collections use HNSW.
        n_clusters: Number of K-Means clusters (shards). Defaults to
            ``max(10, sqrt(n_items))``.
        kmeans_niter: Number of K-Means iterations.
        preview_fields: Comma-separated fields to include in manifest.
        bm25_fields: Comma-separated field names for BM25 search, stored in
            the manifest for the browser worker.
        compress: ``"gzip"`` to write ``.json.gz`` files (default, recommended
            for Node.js/nginx servers), or ``"none"`` to keep plain ``.json``
            (required for GitHub Pages).
        seed: Random seed for K-Means and recall validation.

    Raises:
        FileNotFoundError: If ``raw_embeddings.npy`` does not exist.
        AssertionError: If HNSW recall or shard integrity checks fail.
    """
    import faiss

    # Limit FAISS/OpenMP threads to avoid semaphore leak at shutdown
    try:
        faiss.omp_set_num_threads(1)
    except AttributeError:
        pass

    np.random.seed(seed)
    output_path = Path(output_dir)
    index_dir = output_path / "index"
    layer0_dir = index_dir / "layer0"

    # ── 1. Load normalized embeddings ───────────────────────────────────────
    npy_path = output_path / "raw_embeddings.npy"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"{npy_path} not found. Run 02_generate_embeddings.py first."
        )

    print(f"Loading embeddings from {npy_path}...")
    embeddings_norm = np.load(str(npy_path)).astype(np.float32)
    n_items, dim = embeddings_norm.shape
    print(f"  {n_items} items, {dim}D")

    # Load metadata
    meta_path = output_path / "metadata.json"
    with open(meta_path) as f:
        metadata: list[dict] = json.load(f)
    assert len(metadata) == n_items, "metadata length mismatch"

    # Load flat embeddings config to get matryoshka_dim etc.
    flat_path = output_path / "flat" / "embeddings.int8.json"
    flat_config: dict = {}
    if flat_path.exists():
        with open(flat_path) as f:
            # Only read header (first 512 bytes is enough for config fields)
            content = f.read(512)
        # Quick key extraction — flat file may be large, just parse keys
        try:
            with open(flat_path) as f2:
                tmp = json.loads(f2.read().split('"items"')[0] + '"items":[]}')
                flat_config = {k: v for k, v in tmp.items() if k != "items"}
        except Exception:
            pass

    matryoshka_dim = flat_config.get("matryoshka_dim")

    # Build manifest skeleton (write at the end)
    manifest: dict = {
        "version": "1.0",
        "collection_id": collection_id,
        "n_items": n_items,
        "embedding_dim": dim,
        "matryoshka_dim": matryoshka_dim,
        "quant": "int8",
        "model_id": flat_config.get("model_id", model_id),
        "flat": {"path": "flat/embeddings.int8.json"},
        "index": {"path": "index/", "config": "index/config.json"},
        "thresholds": {"flat_max": flat_threshold},
        "preview_fields": [f.strip() for f in preview_fields.split(",")],
        "bm25_fields": [f.strip() for f in bm25_fields.split(",")],
    }

    # ── 2. Decide flat vs HNSW mode ─────────────────────────────────────────
    use_compress = str(compress).lower() == "gzip"
    if n_items <= flat_threshold:
        print(
            f"n_items={n_items} <= flat_threshold={flat_threshold} → using flat search mode"
        )
        manifest["search_mode"] = "flat"
        if use_compress:
            flat_path = output_path / "flat" / "embeddings.int8.json"
            if flat_path.exists():
                _compress_json(flat_path)
                manifest["flat"]["path"] = "flat/embeddings.int8.json.gz"
            manifest["compressed"] = True
        with open(output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest (flat mode) → {output_path / 'manifest.json'}")
        print("Done! Flat index already built by 02_generate_embeddings.py.")
        return

    use_compress = str(compress).lower() == "gzip"

    # ── 3. K-Means clustering (shard assignment) ────────────────────────────
    n_auto = max(10, int(np.sqrt(n_items)))
    n_clusters_actual = n_clusters if n_clusters is not None else n_auto
    print(
        f"\nK-Means clustering: {n_items} items → {n_clusters_actual} clusters (shards)..."
    )

    kmeans = faiss.Kmeans(
        dim, n_clusters_actual, niter=kmeans_niter, seed=seed, gpu=False, verbose=False
    )
    kmeans.train(embeddings_norm)
    _, cluster_ids = kmeans.index.search(embeddings_norm, 1)
    cluster_ids = cluster_ids.flatten().astype(int)

    cluster_sizes = np.bincount(cluster_ids, minlength=n_clusters_actual)
    print(
        f"  Cluster sizes — min: {cluster_sizes.min()}, max: {cluster_sizes.max()}, mean: {cluster_sizes.mean():.1f}"
    )
    sys.stdout.flush()

    # ── 4. SQ8 quantization of all vectors ──────────────────────────────────
    print("Quantizing all vectors (SQ8)...")
    sys.stdout.flush()
    quantized = [quantize_sq8(embeddings_norm[i]) for i in range(n_items)]
    print(f"  Quantized {len(quantized)} vectors.")
    sys.stdout.flush()

    # ── 5. Build HNSW index ──────────────────────────────────────────────────
    print(f"\nBuilding HNSW index (M={hnsw_M}, ef_construction={ef_construction})...")
    hnsw_index = faiss.IndexHNSWFlat(dim, hnsw_M, faiss.METRIC_INNER_PRODUCT)
    hnsw_index.hnsw.efConstruction = ef_construction
    hnsw_index.add(embeddings_norm)
    print("  HNSW index built.")

    # Validate recall
    print("Validating HNSW recall@10...")
    recall = validate_recall(embeddings_norm, hnsw_index)

    # ── 6. Extract HNSW graph structure ─────────────────────────────────────
    hnsw = hnsw_index.hnsw
    levels = faiss.vector_to_array(hnsw.levels).astype(int)
    flat_neighbors = faiss.vector_to_array(hnsw.neighbors)
    offsets = faiss.vector_to_array(hnsw.offsets)
    max_layer = int(hnsw.max_level)
    entry_node_id = int(hnsw.entry_point)

    # In FAISS, levels[i] = number of levels the node appears in (>= 1).
    # levels[i] = 1 → node is ONLY in layer 0 (not an upper-layer node).
    # levels[i] = k → node is in layers 0 through k-1; its max layer index = k-1.
    print(f"\nHNSW structure: max_layer={max_layer}, entry_node={entry_node_id}")
    for l in range(max_layer + 1):
        # Node is in layer l iff levels[i] >= l+1, i.e., levels[i] > l
        count = int(np.sum(levels > l))
        print(f"  Layer {l}: {count} nodes")

    # ── 7. Export upper_layers.json (layers 1+, always fetched at init) ─────
    print("\nExporting upper_layers.json...")
    index_dir.mkdir(parents=True, exist_ok=True)

    upper_nodes: dict = {}
    for node_id in range(n_items):
        num_levels = int(levels[node_id])
        # Skip nodes that only appear in layer 0 (num_levels == 1)
        if num_levels < 2:
            continue
        actual_max_layer = num_levels - 1  # 0-indexed maximum layer for this node
        layers_neighbors: dict = {}
        for l in range(1, num_levels):  # layers 1 .. actual_max_layer
            nbrs = get_hnsw_neighbors(hnsw, offsets, flat_neighbors, node_id, l)
            if nbrs:
                layers_neighbors[str(l)] = nbrs
        q = quantized[node_id]
        upper_nodes[str(node_id)] = {
            "max_layer": actual_max_layer,
            "scale": q["scale"],
            "qv": q["qv"],
            "layers": layers_neighbors,
        }

    upper_layers_data = {
        "max_layer": max_layer,
        "entry_node_id": entry_node_id,
        "nodes": upper_nodes,
    }
    upper_path = index_dir / "upper_layers.json"
    with open(upper_path, "w") as f:
        json.dump(upper_layers_data, f, separators=(",", ":"))
    size_kb = upper_path.stat().st_size / 1024
    print(f"  upper_layers.json: {len(upper_nodes)} nodes, {size_kb:.1f} KB")

    # ── 8. Export layer0 shards by K-Means cluster ───────────────────────────
    print(f"\nExporting {n_clusters_actual} layer0 shards...")
    layer0_dir.mkdir(parents=True, exist_ok=True)

    node_to_shard: dict = {}
    for cluster_id in range(n_clusters_actual):
        node_ids = np.where(cluster_ids == cluster_id)[0]
        shard_nodes: list[dict] = []
        for node_id in node_ids:
            nid = int(node_id)
            node_to_shard[str(nid)] = cluster_id
            nbrs = get_hnsw_neighbors(hnsw, offsets, flat_neighbors, nid, 0)
            q = quantized[nid]
            shard_nodes.append(
                {
                    "id": nid,
                    "scale": q["scale"],
                    "qv": q["qv"],
                    "neighbors": nbrs,
                }
            )

        shard_data = {"shard_id": cluster_id, "nodes": shard_nodes}
        shard_path = layer0_dir / f"shard_{cluster_id:03d}.json"
        with open(shard_path, "w") as f:
            json.dump(shard_data, f, separators=(",", ":"))

    shard_sizes_kb = [
        (layer0_dir / f"shard_{c:03d}.json").stat().st_size / 1024
        for c in range(n_clusters_actual)
    ]
    print(
        f"  Shard sizes (KB) — min: {min(shard_sizes_kb):.1f}, "
        f"max: {max(shard_sizes_kb):.1f}, mean: {np.mean(shard_sizes_kb):.1f}"
    )

    # ── 9. node_to_shard.json ────────────────────────────────────────────────
    nts_path = index_dir / "node_to_shard.json"
    with open(nts_path, "w") as f:
        json.dump(node_to_shard, f, separators=(",", ":"))
    print(
        f"\nSaved node_to_shard.json ({n_items} entries, {nts_path.stat().st_size / 1024:.1f} KB)"
    )

    # ── 10. cluster_centroids.json ───────────────────────────────────────────
    centroids_data: list[dict] = []
    for c in range(n_clusters_actual):
        centroid = kmeans.centroids[c]
        # Centroids from FAISS Kmeans may not be unit-norm; normalize for dot product
        norm = np.linalg.norm(centroid)
        if norm > 1e-9:
            centroid = centroid / norm
        q = quantize_sq8(centroid.astype(np.float32))
        centroids_data.append({"shard_id": c, "scale": q["scale"], "qv": q["qv"]})

    centroids_path = index_dir / "cluster_centroids.json"
    with open(centroids_path, "w") as f:
        json.dump(centroids_data, f, separators=(",", ":"))
    print(
        f"Saved cluster_centroids.json ({n_clusters_actual} centroids, "
        f"{centroids_path.stat().st_size / 1024:.1f} KB)"
    )

    # ── 11. config.json ──────────────────────────────────────────────────────
    config = {
        "n_items": n_items,
        "dim": dim,
        "matryoshka_dim": matryoshka_dim,
        "quant": "int8",
        "hnsw_M": hnsw_M,
        "hnsw_ef_construction": ef_construction,
        "n_layers": max_layer + 1,
        "n_clusters": n_clusters_actual,
        "entry_node_id": entry_node_id,
        "entry_layer": max_layer,
        "recall_at_10": round(recall, 4),
    }
    config_path = index_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json → {config_path}")

    # ── 12. Validate shard integrity ─────────────────────────────────────────
    print("\nValidating shard integrity (checking all neighbor IDs are valid)...")
    valid_ids = set(range(n_items))
    all_ok = True
    for c in range(n_clusters_actual):
        shard_path = layer0_dir / f"shard_{c:03d}.json"
        with open(shard_path) as f:
            shard = json.load(f)
        for node in shard["nodes"]:
            for nbr in node["neighbors"]:
                if nbr not in valid_ids:
                    print(f"  ERROR: Invalid neighbor {nbr} in shard {c}")
                    all_ok = False
    if all_ok:
        print("  All neighbor IDs valid.")

    # ── 13. titles.json (lightweight metadata for result display) ───────────
    # Omits vectors and long text — only display fields needed by the browser.
    # Indexed by integer node_id (same as HNSW node ID, which is insertion order).
    # Format matches semantic-search: idno, title, type (always "document").
    # NOTE: The ``text`` field is intentionally excluded here to keep this file
    # small. Use bm25_corpus.json if you need the text for BM25 search.
    preview_field_list = [f.strip() for f in preview_fields.split(",")]
    titles_data: dict = {}
    display_fields = [f for f in preview_field_list if f not in ("abstract", "text")]
    for i, meta in enumerate(metadata):
        entry: dict = {}
        for k in display_fields:
            v = meta.get(k)
            if v:
                entry[k] = v
        if "title" in meta:
            entry["title"] = meta["title"]
        # Normalize type to "document" so titles.json matches semantic-search format
        entry["type"] = "document"
        titles_data[str(i)] = entry

    titles_path = index_dir / "titles.json"
    with open(titles_path, "w") as f:
        json.dump(titles_data, f, ensure_ascii=False, separators=(",", ":"))
    print(
        f"Saved titles.json ({n_items} entries, {titles_path.stat().st_size / 1024:.1f} KB)"
    )

    # Update manifest to reference titles file
    manifest["index"]["titles"] = "index/titles.json"

    # ── 13b. bm25_corpus.json (text corpus for browser-side BM25) ───────────
    # Contains id + title + text for every document. The ``text`` field is the
    # abstract/body text from metadata.json — the same content used for
    # embedding. This file is fetched once when BM25 mode is activated and
    # enables keyword search without downloading the 44 MB flat index.
    # Size: ~6 MB uncompressed, ~2 MB gzip.
    bm25_corpus: list[dict] = []
    for i, meta in enumerate(metadata):
        bm25_corpus.append(
            {
                "id": meta.get("id", str(i)),
                "title": meta.get("title", ""),
                "text": meta.get("text", ""),
            }
        )

    bm25_path = index_dir / "bm25_corpus.json"
    with open(bm25_path, "w", encoding="utf-8") as f:
        json.dump(bm25_corpus, f, ensure_ascii=False, separators=(",", ":"))
    print(
        f"Saved bm25_corpus.json ({n_items} entries, {bm25_path.stat().st_size / 1024:.1f} KB)"
    )

    # Update manifest to reference bm25_corpus file
    manifest["index"]["bm25_corpus"] = "index/bm25_corpus.json"

    # ── 13c. Optional gzip compression ──────────────────────────────────────
    if use_compress:
        print("\nCompressing JSON files (gzip)...")
        # Compress flat index (from 02)
        flat_path_uncompressed = output_path / "flat" / "embeddings.int8.json"
        if flat_path_uncompressed.exists():
            _compress_json(flat_path_uncompressed)
            manifest["flat"]["path"] = "flat/embeddings.int8.json.gz"
        # Compress index files
        for name in (
            "upper_layers.json",
            "node_to_shard.json",
            "cluster_centroids.json",
            "config.json",
            "titles.json",
            "bm25_corpus.json",
        ):
            p = index_dir / name
            if p.exists():
                _compress_json(p)
        if manifest["index"].get("titles") == "index/titles.json":
            manifest["index"]["titles"] = "index/titles.json.gz"
        manifest["index"]["config"] = "index/config.json.gz"
        manifest["index"]["upper_layers"] = "index/upper_layers.json.gz"
        manifest["index"]["node_to_shard"] = "index/node_to_shard.json.gz"
        manifest["index"]["bm25_corpus"] = "index/bm25_corpus.json.gz"
        # Compress layer0 shards
        for c in range(n_clusters_actual):
            sp = layer0_dir / f"shard_{c:03d}.json"
            if sp.exists():
                _compress_json(sp)
        manifest["compressed"] = True
        print("  Compressed all index files to .json.gz")
        # Recompute shard sizes for bandwidth estimate
        shard_sizes_kb = [
            (layer0_dir / f"shard_{c:03d}.json.gz").stat().st_size / 1024
            for c in range(n_clusters_actual)
        ]

    # ── 14. manifest.json ────────────────────────────────────────────────────
    manifest["search_mode"] = "hnsw"
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest (hnsw mode) → {manifest_path}")

    # ── 15. Summary ──────────────────────────────────────────────────────────
    total_mb = sum(
        p.stat().st_size
        for p in output_path.rglob("*")
        if p.suffix in (".json", ".gz")
    ) / 1e6
    print(f"\n{'=' * 60}")
    print(f"Index built successfully!")
    print(f"  Collection: {collection_id} ({n_items} items, {dim}D int8)")
    print(
        f"  Mode: HNSW (M={hnsw_M}, ef={ef_construction}, {n_clusters_actual} shards)"
    )
    print(f"  Recall@10: {recall:.4f}")
    fmt = "compressed gzip" if use_compress else "uncompressed JSON"
    print(f"  Total index size: {total_mb:.1f} MB ({fmt})")
    print(
        f"  Per-query bandwidth (cold, 3-4 shards): "
        f"~{3 * np.mean(shard_sizes_kb):.0f}–{4 * np.mean(shard_sizes_kb):.0f} KB"
    )
    print(f"{'=' * 60}")
    print(
        "\nDone! Next: serve the output_dir as static files and load manifest.json in search.worker.js"
    )


if __name__ == "__main__":
    fire.Fire(main)
