"""
Evaluate clustering algorithms using a maximum flow minimum cost algorithm.

Exports the functions:
    - evaluate_clustering: evaluate clustering algorithms using a range of common metrics.
    - translate_labels: translate predicted labels back to original labels.
    - overlap: compute overlap between a predicted cluster and a label form the gold labels.
    - mincostflow: compute a min cost flow between the predicted and the true labels.
"""
from collections import Counter, defaultdict

import networkx as nx
from typing import List, Dict, Union
from numpy.typing import ArrayLike

from src.constants import *


def translate_labels(
    labels: ArrayLike, goldLabels: ArrayLike, dbscanLabels: Union[bool, None] = None
) -> Dict[str, List[int]]:
    """Translate predicted labels back to original labels.
    For each input word, add it to a dictionary with clusters as keys and the original labels as values.

    Parameters:
    -----------
    labels:
        The labels estimated by the clustering algorithm.
    goldLabels:
        The true labels of the data.
    dbscanLabels:
        Whether the clustering algorithm is dbscan or not. If dbscan, the labels wiht -1 are filtered out since they denote noise.

    Returns:
    --------
    clusters:
        Dictionary of lists keyed by clusters such that clusters and their members' gold labels as values in a list such that: ['C0'] = [0, 1, 0, 3, 13, 13, 6, 7, 7, 7]
    """
    clusters = defaultdict(list)

    if dbscanLabels:
        for index, label in enumerate(labels):
            if label != -1:
                actualLabel = goldLabels[index]
                clusters["C" + str(label)].append(actualLabel)
    else:
        for index, label in enumerate(labels):
            actualLabel = goldLabels[index]
            clusters["C" + str(label)].append(actualLabel)
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


def mincostflow(predictedLabels: ArrayLike) -> (int, Dict[str, Dict[str, int]]):
    """Compute a minimum cost maximum flow between the predicted and the true labels.
    Creates a directed graph (Digraph) with edges between predicted and true labels.
    The cost of each edge is the negative overlap between the predicted and the true label.
    The capacity of each edge is 1 or infinite if not set. IN addition there is a supoer source
    connected to all gold labels and a super sink connected to all predicted labels. A min cost max flow is
    computed between the super source and the super sink.

    Predicted clusters are represented by numbers and gold labels are represented by numbers preceeded by a 'c'.
    For the graph, bipartite = 0 is the gold labels and bipartite = 1 is the predicted labels.

    Parameters:
    -----------
    predictedLabels:
        The labels estimated by the clustering algorithm and translated to the original labels.

    Returns:
    --------
    cost:
        The cost of the min cost flow.
    flowDict:
        Dictionary of dictionaries keyed by nodes such that flowDict[u][v] is the flow edge (u, v).
    """
    predictionLabelKeys = list(predictedLabels.keys())
    predictionClustersAsNodes = ["C" + str(i) for i in range(len(predictionLabelKeys))]

    B = nx.DiGraph()
    B.add_nodes_from(list(range(0, 14)), bipartite=0)
    B.add_nodes_from(predictionClustersAsNodes, bipartite=1)

    # Compute overlap between each of the gold clusters.
    for index, cluster in enumerate(predictionLabelKeys):
        for label in range(0, 14):
            cost = overlap(predictedLabels[cluster], label)
            if cost:  # If overlap is not zero, aka cost is not zero add edge.
                B.add_edge(
                    label, predictionClustersAsNodes[index], weight=cost, capacity=1
                )

    B.add_node("superSource")
    for node in range(0, 14):
        B.add_edge("superSource", node)

    B.add_node("superSink")
    for node in predictionClustersAsNodes:
        B.add_edge(node, "superSink", capacity=1)

    # Compute min cost flow.
    flowDict = nx.max_flow_min_cost(B, s="superSource", t="superSink")
    cost = nx.cost_of_flow(B, flowDict)
    return cost, flowDict
