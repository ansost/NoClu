"""Evaluate clustering algorithms using a range of common metrics."""

from time import time
from sklearn import metrics
from numpy.typing import ArrayLike
from typing import List, Dict
import string
from collections import Counter, defaultdict
import networkx as nx
from networkx.algorithms import bipartite


# def evaluate_clusters(true_labels: ArrayLike, estimated_labels: ArrayLike) -> float:
#     """Benchmark a clustering algorithm using a range of common metrics.

#     Parameters:
#     -----------
#     name:
#         Name of the clustering algorithm.
#     data:
#         The data to cluster.
#     true_labels:
#         The true labels of the data.
#     estimated_labels:
#         The labels estimated by the clustering algorithm.

#     Returns:
#     --------
#     metrics:
#         A dictionary of the metric names as keys and their values.
#     """
#     # The silhouette score requires the full dataset
#     silhuette = metrics.silhouette_score(X=true_labels, labels=estimated_labels)
#     return silhuette


def translate_labels(labels: ArrayLike, gold_labels=ArrayLike) -> Dict[str, List[int]]:
    """Translate predicted labels back to original labels.
    For each input word, add it to a dictionary with clusters as keys and the original labels as values.

    Parameters:
    -----------
    labels:
        The labels estimated by the clustering algorithm.
    gold_labels:
        The true labels of the data.

    Returns:
    --------
    clusters:
        Dictionary of lists keyed by clusters such that clusters and their members' gold labels as values in a list such that: ['C0'] = [0, 1, 0, 3, 13, 13, 6, 7, 7, 7]
    """
    clusters = defaultdict(list)
    for index, label in enumerate(labels):
        actual_label = gold_labels[index]
        clusters["C" + str(label)].append(actual_label)
    return clusters


def overlap(cluster: List[int], label: int) -> int:
    """Compute overlap between a predicted cluster and a label form the gold labels.
    Returns negative overlap so smaller numbers are better.

    Parameters:
    -----------
    cluster:
        List of predicted labels.
    label:
        Label from the gold labels.

    Returns:
    --------
    overlap:
        Negative overlap between the predicted cluster and the gold label."""
    c = Counter(cluster)

    if c[label] == 0:
        return None
    if c[label] != 0:
        return -c[label]


def mincostflow(predicted_labels: ArrayLike) -> Dict:
    """Compute a min cost flow between the predicted and the true labels.
    Creates a directed digraph with edges between predicted and true labels.

    Parameters:
    -----------
    predicted_labels:
        The labels estimated by the clustering algorithm.

    Returns:
    --------
    cost:
        The cost of the min cost flow.
    flowDict:
        Dictionary of dictionaries keyed by nodes such that flowDict[u][v] is the flow edge (u, v).
    """

    predictionclusters_as_nodes = [
        string.ascii_lowercase[i] for i in range(len(predicted_labels.keys()))
    ]

    # Initialize all gold labels as nodes represented by numbers. Predicted clusters are letters.
    # Bipartite = 0 is the gold labels and bipartite = 1 is the predicted labels.
    B = nx.DiGraph()
    B.add_nodes_from(list(range(0, 14)), bipartite=0)
    B.add_nodes_from(predictionclusters_as_nodes, bipartite=1)

    # Compute overlap between each of the gold clusters.
    for index, cluster in enumerate(predicted_labels.keys()):
        for label in range(0, 14):
            cost = overlap(predicted_labels[cluster], label)
            if cost:
                B.add_edge(
                    label, predictionclusters_as_nodes[index], weight=cost, capacity=1
                )  # TODO: Stimmt capacity=1 here?

    B.add_node("super_source")
    for node in range(0, 14):
        B.add_edge("super_source", node)

    B.add_node("super_sink")
    for node in predictionclusters_as_nodes:
        B.add_edge(
            node, "super_sink", capacity=1
        )  # If capacity is not set, it is assumed to be infinite.

    # Compute min cost flow.
    flowDict = nx.max_flow_min_cost(B, s="super_source", t="super_sink")
    cost = nx.cost_of_flow(B, flowDict)
    return cost, flowDict
