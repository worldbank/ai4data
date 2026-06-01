from typing import NamedTuple

import numpy as np
from sklearn.cluster import KMeans


class ClusteringResult(NamedTuple):
    cluster_centers: np.ndarray
    labels: np.ndarray


def semantic_cluster_embedding(data: np.ndarray, n_clusters: int = 10, **kmeans_kwargs) -> ClusteringResult:
    """
    Perform KMeans clustering on the embedding data to obtain semantic clusters.

    Args:
        data (np.ndarray): The embedding data, expected to be of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to create. Defaults to 10.
        **kmeans_kwargs: Additional keyword arguments to pass to the KMeans constructor.

    Returns:
        ClusteringResult: A named tuple containing the cluster centers and labels.

    Raises:
        ValueError: If `n_clusters` is greater than the number of data points.
    """

    # Ensure that the number of clusters does not exceed the number of data points
    if n_clusters > len(data):
        Warning(
            f"n_clusters ({n_clusters}) cannot be greater than the number of data points ({len(data)}). Setting n_clusters to {len(data)}."
        )

        n_clusters = len(data)

    n_clusters = min(n_clusters, len(data))

    kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
    kmeans.fit(data)

    # Return the cluster centers and labels as a named tuple
    return ClusteringResult(cluster_centers=kmeans.cluster_centers_, labels=kmeans.labels_)
