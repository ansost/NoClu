"""Functions for using different clustering algorithms."""
from numpy.typing import ArrayLike
from typing import List
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering


def kmeans(vectors: ArrayLike, n_clusters: int) -> List[ArrayLike]:
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


def dbscan(vectors: ArrayLike, epsilon: float) -> List[ArrayLike]:
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


def wards(vectors: ArrayLike, n_clusters: int) -> List[ArrayLike]:
    """Cluster vectors using Ward's method.

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
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(vectors)
    labels = ward.labels_
    return labels


# TODO: Add BICO (from github repo?, is there another source?)
