"""
Plot results from clustering evaluationusing a scatterplot.
Optionally save the plot.

Exports the functions:
    make_plot_title - Make a title for the plot.
    plot_clusters - Plot the data in a scatterplot.
"""
import numpy as np
import matplotlib.pyplot as plt


def make_plot_title(
    algorithm: str, algoParameter: str, inputName: str, cost: str
) -> str:
    """Make a title for the plot.

    Parameters
    ----------
    algorithm:
        Clustering algorithm used.
    algoParameter:
        Parameter used for the clustering algorithm.
    inputName:
        Name of the input file.
    cost:
        Cost of the clustering.

    Returns
    -------
    title:
        Title for the plot.
    """
    if algorithm == "kmeans":
        return f"K-means with {algoParameter} clusters on {inputName} (Cost {cost})."
    if algorithm == "dbscan":
        return f"DBSCAN with {algoParameter} epsilon on {inputName} (Cost {cost})."
    if algorithm == "BICO":
        return f"BICO with {algoParameter} projections on {inputName} (Cost {cost})."
    return "No title"


def plot_clusters(
    inputData, predictions, title: str, saveTo: str | None = None
) -> None:
    """Plot the data in a scatterplot.

    Parameters
    ----------
    inputData:
        Data to plot.
    predictions:
        Predicted clusters.
    title:
        Title of the plot.
    saveTo:
        Path to save the plot to.
    """
    plt.scatter(inputData[:, 0], inputData[:, 1], c=predictions)
    plt.title(title)
    plt.show()
    if saveTo:
        plt.savefig(saveTo)
