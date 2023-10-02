"""
Evaluate clustering algorithms using a maximum flow minimum cost algorithm.

Exports the functions:
    - evaluate_clustering(): evaluate clustering algorithms using a range of common metrics.
    - translate_labels(): translate predicted labels back to original labels.
    - overlap(): compute overlap between a predicted cluster and a label form the gold labels.
    - mincostflow(): compute a min cost flow between the predicted and the true labels.
"""

from time import time
from sklearn import metrics
from numpy.typing import ArrayLike
from typing import List, Dict
import string
import json
from collections import Counter, defaultdict
import networkx as nx
from networkx.algorithms import bipartite


def translate_labels(
    labels: ArrayLike, gold_labels=ArrayLike, from_dbscan: bool = None
) -> Dict[str, List[int]]:
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

    if from_dbscan:
        for index, label in enumerate(labels):
            if label != -1:
                actual_label = gold_labels[index]
                clusters["C" + str(label)].append(actual_label)
    else:
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
    """Compute a minimum cost maximum flow between the predicted and the true labels.
    Creates a directed graph (Digraph) with edges between predicted and true labels. a super source and sink
    between which to compute the flow. The cost of each edge is the negative overlap between the predicted
    and the true label. The capacity of each edge is 1 or infinite if not set.

    Predicted clusters are represented by numbers and gold labels are represented by numbers preceeded by a 'c'.
    For the graph, bipartite = 0 is the gold labels and bipartite = 1 is the predicted labels.

    Parameters:
    -----------
    predicted_labels:
        The labels estimated by the clustering algorithm and translated to the original labels.

    Returns:
    --------
    cost:
        The cost of the min cost flow.
    flowDict:
        Dictionary of dictionaries keyed by nodes such that flowDict[u][v] is the flow edge (u, v).
    """

    predictionclusters_as_nodes = [
        "C" + str(i) for i in range(len(predicted_labels.keys()))
    ]

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
                )

    B.add_node("super_source")
    for node in range(0, 14):
        B.add_edge("super_source", node)

    B.add_node("super_sink")
    for node in predictionclusters_as_nodes:
        B.add_edge(node, "super_sink", capacity=1)

    # Compute min cost flow.
    flowDict = nx.max_flow_min_cost(B, s="super_source", t="super_sink")
    cost = nx.cost_of_flow(B, flowDict)
    return cost, flowDict
