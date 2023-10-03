"""
Run and evaluate clustering algorithms.

Usage:
    python3 main.py

Input for the script is a config file, which contains the following parameters:
    - input: path to input file (word embeddings)
    - combinations: list of combinations of clustering algorithms and number of clusters
    - saveTo: path to save output files

More information on the parameters can be found in the config file ('/data/config_files/noclu.config').

Raises exception if input file is not a .npy file.
"""
import time
import json
from typing import Tuple, Union, Dict
import sys

sys.path.append("..")  # FIXME: this is a hack

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.typing import ArrayLike

from src.evaluate import mincostflow, translate_labels
from src.clustering import kmeans, dbscan
from src.constants import *


def valid_algorithm(algorithm: str) -> bool:
    """Check whether clustering algorithm is implemented.
    Current valid algorithms are 'kmeans' and 'dbscan'.
    """
    if str(algorithm) not in ["kmeans", "dbscan"]:
        print(f"Invalid algorithm: {algorithm}. Must be 'kmeans' or 'dbscan'.")
        return False
    return True


def get_params(combination: str) -> Tuple[str, Union[int, float]]:
    """Get parameter combinations from config string."""
    params = combination.split("_")
    algorithm = params[0]
    if algorithm == "dbscan":
        nClusters = float(params[1])
        return algorithm, nClusters
    nClusters = int(params[1])
    return algorithm, nClusters


def new_row(
    input: ArrayLike,
    cost: int,
    algorithm: str,
    nClusters: Union[int, float],
    goldLabelPath: str,
    saveTo: str,
) -> Dict[str, str]:
    """Create new row for result.csv.

    Parameters:
    -----------
    input:
        Path to input file from current iteration.
    algorithm:
        Clustering algorithm.
    nClusters:
        Number of clusters.
    goldLabelPath:
        Path to gold labels.
    saveTo:
        Path to save output files.
    cost:
        Cost of min cost flow.

    Returns:
    --------
    newRow:
        Dictionary of parameters.
    """
    newRow = {
        "input": input.split("/")[-1],
        "cost": cost,
        "algorithm": algorithm,
        "nClusters": nClusters,
        "goldLabelPath": goldLabelPath,
        "saveTo": saveTo,
        "date": pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M-%S"),
    }
    return newRow


if __name__ == "__main__":
    startTime = time.time()
    ALGODICT = {"kmeans": kmeans, "dbscan": dbscan}

    # Load gold labels from npy.
    with open(NOCLUCONFIG, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    input = config["input"]
    combinations = config["combinations"]
    saveTo = config["saveTo"]

    if not input.endswith("npy"):
        raise Exception("Input file must be .npy file.")

    wordEmbeddings = np.load(input)
    goldLabels = np.load(GOLDLABELS1D)

    df = pd.read_csv(RESULTCSV)

    for combination in tqdm(combinations):
        algorithm, nClusters = get_params(combination)

        print(f"Running {algorithm} with {nClusters} clusters.")
        if valid_algorithm(algorithm):
            if algorithm == "dbscan":
                dbscanLabels = True
            else:
                dbscanLabels = False

            # Cluster.
            labels = ALGODICT[algorithm](wordEmbeddings, nClusters)
            filename = f"{saveTo}{combination}{input.split('/')[-1]}"
            np.save(filename, labels)

            # Compute min cost flow.
            translatedLabels = translate_labels(labels, goldLabels, dbscanLabels)
            cost, flowDict = mincostflow(predictedLabels=translatedLabels)

            # Write to json.
            with open(f"{FLOWDICTOUT}{combination}.json", "w") as f:
                json.dump(flowDict, f)

            # Save results in logs.
            goldLabelPath = GOLDLABELS1D.split("/")[-1]
            inputFile = input.split("/")[-1]
            addRow = new_row(
                inputFile, cost, algorithm, nClusters, goldLabelPath, saveTo
            )
            df.loc[len(df.index)] = addRow

    df.to_csv(RESULTCSV, index=False)
    print(df[["algorithm", "nClusters", "cost", "input"]].tail(10))
    print(f"Finished in {time.time() - startTime} seconds.")
