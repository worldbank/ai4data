"""Clustering utilities for grouping data dictionary variables by semantic similarity.

Pipeline:
1. Optionally reduce dimensionality with TruncatedSVD (for large dictionaries).
2. Estimate optimal number of clusters via silhouette score (or use heuristic).
3. Run AgglomerativeClustering with Ward linkage.
4. Post-hoc token-budget split: subdivide clusters that would exceed the LLM
   context limit when rendered as a variable list.

All sklearn/numpy imports are lazy to avoid import-time overhead.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .schemas import DictionaryVariable

if TYPE_CHECKING:
    import numpy as np

# ----- Defaults ----- #

DEFAULT_SVD_COMPONENTS = 64
DEFAULT_SVD_THRESHOLD = 150        # apply SVD only when N > this
DEFAULT_N_CLUSTERS_RANGE = (3, 30)
DEFAULT_MAX_CLUSTER_TOKENS = 2048  # max tokens per cluster for LLM prompt
DEFAULT_APPROX_TOKENS_PER_LABEL = 10  # conservative estimate per "name: label" line


# ----- Dimensionality reduction ----- #


def reduce_dimensions(
    embeddings: "np.ndarray",
    n_components: int = DEFAULT_SVD_COMPONENTS,
    *,
    threshold: int = DEFAULT_SVD_THRESHOLD,
    random_state: int = 42,
) -> "np.ndarray":
    """Apply TruncatedSVD when N > threshold, otherwise return embeddings unchanged.

    For small dictionaries (N ≤ threshold), full-dimensional embeddings work well
    and SVD may overfit. For large dictionaries, SVD improves clustering quality
    by removing noise dimensions.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Shape (N, D) float32 embedding matrix.
    n_components : int
        Target dimensionality after reduction.
    threshold : int
        Minimum N to apply SVD.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Shape (N, min(n_components, D)) reduced embeddings.
    """
    import numpy as np

    n, d = embeddings.shape
    if n <= threshold or d <= n_components:
        return embeddings
    try:
        from sklearn.decomposition import TruncatedSVD
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for dimensionality reduction. "
            "Install with: uv pip install ai4data[metadata]"
        ) from exc
    n_comp = min(n_components, d, n - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    reduced = svd.fit_transform(embeddings)
    # Re-normalize so cosine similarity remains valid
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (reduced / norms).astype(np.float32)


# ----- Cluster count estimation ----- #


def _silhouette_estimate(
    embeddings: "np.ndarray",
    n_range: Tuple[int, int],
    random_state: int,
) -> int:
    """Estimate optimal k by maximising silhouette score over n_range."""
    import numpy as np

    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required. Install with: uv pip install ai4data[metadata]"
        ) from exc

    n = embeddings.shape[0]
    lo, hi = n_range
    hi = min(hi, n - 1)
    lo = min(lo, hi)
    if lo >= hi:
        return lo

    # Use a random subsample for speed when N is large
    max_sample = 500
    if n > max_sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, max_sample, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    best_k, best_score = lo, -1.0
    for k in range(lo, hi + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, linkage="ward"
        ).fit_predict(sample)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(sample, labels, metric="euclidean")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def estimate_n_clusters(
    embeddings: "np.ndarray",
    n_range: Tuple[int, int] = DEFAULT_N_CLUSTERS_RANGE,
    *,
    random_state: int = 42,
) -> int:
    """Estimate the optimal number of clusters.

    Uses silhouette score over the candidate range. Falls back to the
    square-root heuristic ``max(3, int(sqrt(N / 2)))`` when the range is
    degenerate or N is very small.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Shape (N, D) embedding matrix.
    n_range : tuple of (int, int)
        (min_clusters, max_clusters) search range.
    random_state : int
        Random seed.

    Returns
    -------
    int
        Estimated number of clusters.
    """
    n = embeddings.shape[0]
    lo, hi = n_range
    hi = min(hi, n - 1)
    lo = max(2, min(lo, hi))

    # Fall back to heuristic for very small N
    if n < 6 or lo >= hi:
        return max(lo, min(hi, max(3, int(math.sqrt(n / 2)))))

    try:
        return _silhouette_estimate(embeddings, (lo, hi), random_state)
    except Exception:
        return max(lo, min(hi, max(3, int(math.sqrt(n / 2)))))


# ----- Clustering ----- #


def cluster_variables(
    embeddings: "np.ndarray",
    n_clusters: Optional[int] = None,
    *,
    n_range: Tuple[int, int] = DEFAULT_N_CLUSTERS_RANGE,
    linkage: str = "ward",
    random_state: int = 42,
) -> "np.ndarray":
    """Cluster embedding vectors using AgglomerativeClustering.

    Ward linkage minimises within-cluster variance, which is well-suited to
    semantic embedding spaces. Unlike k-means, it is deterministic and does
    not require careful initialisation.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Shape (N, D) embedding matrix.
    n_clusters : int, optional
        Number of clusters. If not given, estimated via ``estimate_n_clusters``.
    n_range : tuple of (int, int)
        Search range for automatic cluster count estimation.
    linkage : str
        Linkage criterion for AgglomerativeClustering.
    random_state : int
        Seed for silhouette estimation sub-sampling.

    Returns
    -------
    numpy.ndarray
        Integer label array of length N.
    """
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required. Install with: uv pip install ai4data[metadata]"
        ) from exc

    k = n_clusters or estimate_n_clusters(
        embeddings, n_range=n_range, random_state=random_state
    )
    k = max(1, min(k, embeddings.shape[0]))
    return AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(
        embeddings
    )


# ----- Token budget enforcement ----- #


def _cluster_token_count(
    variables: List[DictionaryVariable],
    approx_tokens_per_label: int = DEFAULT_APPROX_TOKENS_PER_LABEL,
) -> int:
    """Estimate the token count for a cluster's variable list."""
    return sum(
        approx_tokens_per_label + len(v.label.split())
        for v in variables
    )


def split_clusters_for_token_budget(
    cluster_labels: "np.ndarray",
    variables: List[DictionaryVariable],
    *,
    max_tokens_per_cluster: int = DEFAULT_MAX_CLUSTER_TOKENS,
    approx_tokens_per_label: int = DEFAULT_APPROX_TOKENS_PER_LABEL,
) -> "np.ndarray":
    """Split clusters that exceed the token budget into smaller clusters.

    This post-hoc step ensures that each LLM call receives a variable list that
    fits within the context window. When a cluster is oversized, the minimum
    number of trailing variables are moved into a new cluster until the
    remainder fits the budget. The process repeats until all clusters comply,
    or a single variable alone exceeds the limit (left unchanged).

    Parameters
    ----------
    cluster_labels : numpy.ndarray
        Integer label array of length N from ``cluster_variables``.
    variables : list of DictionaryVariable
        Variables corresponding to each position in ``cluster_labels``.
    max_tokens_per_cluster : int
        Maximum token budget per cluster prompt.
    approx_tokens_per_label : int
        Per-variable token overhead (variable name + formatting).

    Returns
    -------
    numpy.ndarray
        Updated label array (same length, possibly more unique cluster IDs).
    """
    labels = cluster_labels.copy()
    cluster_map = build_cluster_map(labels, variables)
    name_to_idx = {v.variable_name: i for i, v in enumerate(variables)}
    next_id = int(labels.max()) + 1

    changed = True
    while changed:
        changed = False
        oversized = sorted(
            (
                (cid, _cluster_token_count(vars_, approx_tokens_per_label))
                for cid, vars_ in cluster_map.items()
            ),
            key=lambda x: -x[1],
        )
        for cid, tokens in oversized:
            if tokens <= max_tokens_per_cluster:
                continue

            remaining = list(cluster_map[cid])
            if len(remaining) <= 1:
                # A single variable exceeds the budget; cannot split further.
                continue

            move_vars: List[DictionaryVariable] = []
            while (
                len(remaining) > 1
                and _cluster_token_count(remaining, approx_tokens_per_label)
                > max_tokens_per_cluster
            ):
                move_vars.insert(0, remaining.pop())

            if not move_vars:
                continue

            for var in move_vars:
                labels[name_to_idx[var.variable_name]] = next_id

            cluster_map[cid] = remaining
            cluster_map[next_id] = move_vars
            next_id += 1
            changed = True
            break  # restart loop after any change

    return labels


def merge_clusters_for_token_budget(
    cluster_labels: "np.ndarray",
    variables: List[DictionaryVariable],
    *,
    max_tokens_per_cluster: int = DEFAULT_MAX_CLUSTER_TOKENS,
    approx_tokens_per_label: int = DEFAULT_APPROX_TOKENS_PER_LABEL,
) -> "np.ndarray":
    """Deprecated alias for :func:`split_clusters_for_token_budget`."""
    warnings.warn(
        "merge_clusters_for_token_budget is deprecated; "
        "use split_clusters_for_token_budget instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return split_clusters_for_token_budget(
        cluster_labels,
        variables,
        max_tokens_per_cluster=max_tokens_per_cluster,
        approx_tokens_per_label=approx_tokens_per_label,
    )


# ----- Utility ----- #


def build_cluster_map(
    cluster_labels: "np.ndarray",
    variables: List[DictionaryVariable],
) -> Dict[int, List[DictionaryVariable]]:
    """Build a mapping from cluster ID to list of DictionaryVariable objects.

    Parameters
    ----------
    cluster_labels : numpy.ndarray
        Integer label array of length N.
    variables : list of DictionaryVariable
        Variables corresponding to each position.

    Returns
    -------
    dict
        ``{cluster_id: [DictionaryVariable, ...]}``, ordered by cluster size
        (descending).
    """
    from collections import defaultdict

    cluster_map: Dict[int, List[DictionaryVariable]] = defaultdict(list)
    for label, var in zip(cluster_labels.tolist(), variables):
        cluster_map[int(label)].append(var)
    # Sort by descending cluster size for deterministic processing order
    return dict(
        sorted(cluster_map.items(), key=lambda x: -len(x[1]))
    )
