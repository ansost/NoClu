"""
Functions for using different clustering algorithms.

Exports the functions:
    kmeans - k-means clustering
    dbscan - DBSCAN clustering
"""
from numpy.typing import ArrayLike
from sklearn.cluster import KMeans, DBSCAN


def kmeans(vectors: ArrayLike, n_clusters: int) -> ArrayLike:
    """Cluster vectors using k-means.

    Parameters
    ----------
    vectors:
        Vectors to cluster.
    n_clusters:
        Number of clusters to create.

    Returns
    -------
    labels:
        Cluster labels of each point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(vectors)
    labels = kmeans.labels_
    return labels


def dbscan(vectors: ArrayLike, epsilon: float) -> ArrayLike:
    """Cluster vectors using DBSCAN.

    Parameters
    ----------
    vectors:
        Vectors to cluster.
    epsilon:
        Maximum distance between two samples for one to be considered as in the neighborhood of the other.

    Returns
    -------
    labels:
        Cluster labels of each point.
    """
    db = DBSCAN(eps=epsilon).fit(vectors)
    labels = db.labels_
    return labels
