"""
Plot results from clustering evaluation.

Usage:
    python3 plot_results.py

This script produces barplots with a lineplotoverleyed at the top of the bars.
The x axis describes the number of cluster for k-means and the epsilon value for DBSCAN.
The y-axis describes the cost of the minimum cost flow.
Figures are saved in the figure folder by default.
"""
import sys

sys.path.append("..")  # FIXME: this is a hack

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.constants import RESULTCSV


def plot_data(data: pd.core.frame.DataFrame, saveTo: str, xLabel: str, algorithm: str):
    """Plot code and number of clusters in a bar and line plot.
    Saves output plot to saveTo.

    Parameters:
    ----------
    data:
        Dataframe with the data to plot.
    saveTo:
        Path to save the plot.
    xLabel:
        Label for the x axis.
    algorithm:
        Name of the algorithm to plot.
    """
    plt.clf()
    if algorithm == "kmeans":
        plt.bar(data["nClusters"], data["cost"], color="lightsteelblue")
        plt.xticks(data["nClusters"].astype({"nClusters": "int32"}))
        plt.plot(data["nClusters"], data["cost"], color="slategrey")

    if algorithm == "dbscan":
        plt.bar(data["nClusters"].astype("str"), data["cost"], color="lightsteelblue")
        plt.plot(data["nClusters"].astype("str"), data["cost"], color="slategrey")
    plt.xlabel(xLabel)
    plt.ylabel("Cost")
    plt.title(
        "Cost for different number of clusters for input "
        + data["input"].iloc[0][:-4].replace("_", " ")
    )

    plt.savefig(saveTo + data["input"].iloc[0][:-4] + ".png")


if __name__ == "__main__":
    d = pd.read_csv(RESULTCSV)
    interestingColumns = d[["cost", "nClusters", "algorithm", "input"]]
    kmeans = interestingColumns[interestingColumns["algorithm"] == "kmeans"]
    kmeans.loc[:, "nClusters"] = kmeans.loc[:, "nClusters"].astype(int)
    dbscan = interestingColumns[interestingColumns["algorithm"] == "dbscan"]

    kmeans2152 = kmeans[kmeans["input"] == "nonsynchr_pcatsne_215_2.npy"]
    kmeans2153 = kmeans[kmeans["input"] == "nonsynchr_pcatsne_215_3.npy"]
    kmeans2772 = kmeans[kmeans["input"] == "nonsynchr_pcatsne_277_2.npy"]
    kmeans2773 = kmeans[kmeans["input"] == "nonsynchr_pcatsne_277_3.npy"]

    kmeansDatasets = [kmeans2152, kmeans2153, kmeans2772, kmeans2773]
    for kmeansSet in kmeansDatasets:
        kmeansSet.loc[:, "cost"] = np.absolute(kmeansSet["cost"])

    dbscan2152 = dbscan[dbscan["input"] == "nonsynchr_pcatsne_215_2.npy"]
    dbscan2153 = dbscan[dbscan["input"] == "nonsynchr_pcatsne_215_3.npy"]
    dbscan2772 = dbscan[dbscan["input"] == "nonsynchr_pcatsne_277_2.npy"]
    dbscan2773 = dbscan[dbscan["input"] == "nonsynchr_pcatsne_277_3.npy"]
    dbscanDatasets = [dbscan2152, dbscan2153, dbscan2772, dbscan2773]

    for dbscanSet in dbscanDatasets:
        dbscanSet.loc[:, "cost"] = np.absolute(dbscanSet["cost"])

    for data in kmeansDatasets:
        plot_data(data, "../figures/cost_kmeans_", "Number of clusters", "kmeans")

    for data in dbscanDatasets:
        plot_data(data, "../figures/cost_dbscan_", "Epsilon value", "dbscan")
