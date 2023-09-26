"""Run and evaluate clustering algorithms.
 
	Usage:
		noclu.py <input> <combinations> <saveto> 
  
	Arguments:
		input   path to input data - str
		combinations   List of clustering algorithms and number of clusters/epsilon. Valid algos are 'kmeans', 'dbscan', 'ward'. Example: ['kmeans_2', 'dbscan_0.5', 'ward_2'] - list
        saveto  directory to save output to - str
        gold_labels path to true labels of the input - str
"""
import time
import pickle
import numpy as np
import pandas as pd
from docopt import docopt
from typing import Tuple
from tqdm import tqdm
from evaluation import evaluate_clusters
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


if __name__ == "__main__":
    args = docopt(__doc__)
    input = args["<input>"]
    combinations = args["<combinations>"]
    saveto = args["<saveto>"]

    start_time = time.time()

    algo_dict = {"kmeans": kmeans, "dbscan": dbscan, "ward": wards}

    word_embeddings = np.load(input)

    df = pd.read_csv("../data/tests/test_results.csv")

    gold_label_path = "../data/nonsynchr_labels_1dnumbers.npy"
    gold_labels = np.load(gold_label_path)
    # with open("../data/nonsynchr_labels.pkl", "rb") as f:
    #    gold_labels = pickle.load(f)

    combinations = combinations.split()
    for combination in tqdm(combinations):
        algorithm, n_clusters = get_params(combination)
        print(f"Running {algorithm} with {n_clusters} clusters.")

        if valid_algorithm(algorithm):
            labels = algo_dict[algorithm](word_embeddings, n_clusters)
            evaluate_clusters(true_labels=gold_labels, estimated_labels=labels)
            # np.save(saveto + combination + ".npy", labels)

            new_row = {
                "input": input,
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "gold_labels": gold_label_path,
                "output": saveto,
                "date": pd.to_datetime("today").strftime("%Y-%m-%d-%H-%M-%S"),
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("../data/tests/test_results.csv", index=False)

    print(f"Finished in {time.time() - start_time} seconds.")
