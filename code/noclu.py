"""Run and evaluate clustering algorithms.
 
	Usage:
		noclu.py <input> <combinations> <saveto> 
  
	Arguments:
		input   path to input data - str
		combinations   List of clustering algorithms and number of clusters/epsilon. Valid algos are 'kmeans', 'dbscan', 'ward'. Example: ['kmeans_2', 'dbscan_0.5', 'ward_2'] - list
        saveto  directory to save output to - str
"""
import time
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from docopt import docopt
from typing import Tuple, Dict
from tqdm import tqdm
from evaluation import mincostflow, translate_labels
from clustering import *


def valid_algorithm(algorithm: str) -> bool:
    """Check whether clustering algorithm is implemented.
    Current valid algorithms are 'kmeans', 'dbscan' and 'ward'.
    """
    if str(algorithm) not in ["kmeans", "dbscan", "ward"]:
        print(f"Invalid algorithm: {algorithm}. Must be 'kmeans', 'dbscan' or 'ward'.")
        return False
    return True


def get_params(combination: str) -> Tuple[str, str]:
    """Get parameter combinations from string."""
    params = combination.split("_")
    algorithm = params[0]
    n_clusters = int(params[1])
    return algorithm, n_clusters


def new_row(
    input: ArrayLike, algorithm: str, n_clusters: int, gold_label_path: str, saveto: str
) -> Dict[str, str]:
    new_row = {
        "input": input,
        "algorithm": algorithm,
        "n_clusters": n_clusters,
        "gold_labels": gold_label_path,
        "output": saveto,
        "date": pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M-%S"),
    }
    return new_row


if __name__ == "__main__":
    args = docopt(__doc__)
    input = args["<input>"]
    combinations = args["<combinations>"]
    saveto = args["<saveto>"]
    start_time = time.time()
    algo_dict = {"kmeans": kmeans, "dbscan": dbscan, "ward": wards}

    word_embeddings = np.load(input)
    df = pd.read_csv("../data/result.csv")
    gold_label_path = "../data/gold_labels_1d.npy"
    gold_labels = np.load(gold_label_path)

    combinations = combinations.split()
    for combination in tqdm(combinations):
        algorithm, n_clusters = get_params(combination)

        print(f"Running {algorithm} with {n_clusters} clusters.")
        if valid_algorithm(algorithm):
            # Cluster.
            labels = algo_dict[algorithm](word_embeddings, n_clusters)
            filename = f"{saveto}{combination}{input.split('/')[-1]}"
            np.save(filename, labels)

            # Compute min cost flow.
            translated_labels = translate_labels(labels, gold_labels)
            cost, flowDict = mincostflow(predicted_labels=translated_labels)

            # Save results in logs.
            input_file = input.split("/")[-1]
            new_row = {
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "cost": cost,
                "flowDict": flowDict,
                "gold_label_path": gold_label_path,
                "input": input_file,
                "saveto": saveto,
                "date": pd.to_datetime("today"),
            }
            df.loc[len(df.index)] = new_row

    df.to_csv("../data/result.csv", index=False)
    print(df[["algorithm", "n_clusters", "cost", "input"]].tail(10))
    print(f"Finished in {time.time() - start_time} seconds.")
