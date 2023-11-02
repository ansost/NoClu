"""Plot clusters from kMeans and DBSCAN.

Usage:
    python3 plot_clusters.py
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_DBSCAN(labels, input_, coreSampleIndices_, filename):
    """Plot clusters from DBSCAN.
    Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html.

    Parameters
    ----------
    labels:
        Cluster labels of each point.
    input_:
        Vectors to cluster.
    coreSampleIndices_:
        Indices of core samples.
    filename:
        Name of saved plot.
    """
    nClusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    nNoise_ = list(labels).count(-1)

    uniqueLabels = set(labels)
    coreSamplesMask = np.zeros_like(labels, dtype=bool)
    coreSamplesMask[coreSampleIndices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(uniqueLabels))]
    for k, col in zip(uniqueLabels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        classMemberMask = labels == k

        xy = input_[classMemberMask & coreSamplesMask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = input_[classMemberMask & ~coreSamplesMask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {nClusters_}")
    plt.show()
    plt.savefig(f"../figures/clusters/{filename}.png")
