"""Evaluate clustering algorithms using a range of common metrics."""

from time import time
from sklearn import metrics
from numpy.typing import ArrayLike
from typing import Dict
from sklearn import metrics

def evaluate_clusters(name: str, data: ArrayLike, true_labels: ArrayLike, estimated_labels: ArrayLike) -> Dict[str, float]:
    """Benchmark a clustering algorithm using a range of common metrics.
    
    Parameters:
    -----------
    name:
        Name of the clustering algorithm.
    data:
        The data to cluster.
    true_labels:
        The true labels of the data.
    estimated_labels:
        The labels estimated by the clustering algorithm.
    
    Returns:
    --------
    metrics:
        A dictionary of the metric names as keys and their values.
    """
    metrics = {}
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]

    for m in clustering_metrics: 
        result = m(true_labels, estimated_labels)
        metrics[name] = result

    # The silhouette score requires the full dataset
    metrics["silhuette"] = metrics.silhouette_score(data, true_labels, metric="euclidean", sample_size=300)
    return metrics