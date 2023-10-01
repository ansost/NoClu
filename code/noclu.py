"""Run and evaluate clustering algorithms.
Input for the script is a config file, which contains the following parameters:
- input: path to input file (word embeddings)
- gold_labels: path to gold labels
- combinations: list of combinations of clustering algorithms and number of clusters
- saveto: path to save output files

More information on the parameters can be found in the config file ('/data/config_files/noclu.config').
"""
import time
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import yaml
import json
from typing import Tuple, Dict
from tqdm import tqdm
from evaluation import mincostflow, translate_labels
from clustering import kmeans, dbscan, wards


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
    if algorithm == "dbscan":
        n_clusters = float(params[1])
    else:
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
    start_time = time.time()

    print("Loading config file...")
    with open("../data/config_files/noclu.config", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    input = config["input"]
    gold_label_path = config["gold_labels"]
    combinations = config["combinations"]
    saveto = config["saveto"]

    assert input[-3:] == "npy", "Input file must be .npy file."
    word_embeddings = np.load(input)
    gold_labels = np.load(gold_label_path)

    df = pd.read_csv("../data/result.csv")
    algo_dict = {"kmeans": kmeans, "dbscan": dbscan, "ward": wards}

    for combination in tqdm(combinations):
        algorithm, n_clusters = get_params(combination)

        print(f"Running {algorithm} with {n_clusters} clusters.")
        if valid_algorithm(algorithm):
            if algorithm == "dbscan":
                from_dbscan = True
            else:
                from_dbscan = False

            # Cluster.
            labels = algo_dict[algorithm](word_embeddings, n_clusters)
            filename = f"{saveto}{combination}{input.split('/')[-1]}"
            np.save(filename, labels)

            # Compute min cost flow.
            translated_labels = translate_labels(labels, gold_labels, from_dbscan)
            cost, flowDict = mincostflow(predicted_labels=translated_labels)

            # write to json
            with open(f"../data/flow_dictionaries/{combination}.json", "w") as f:
                json.dump(flowDict, f)

            # Save results in logs.
            input_file = input.split("/")[-1]
            new_row = {
                "algorithm": algorithm,
                "n_clusters": n_clusters,
                "cost": cost,
                "gold_label_path": gold_label_path,
                "input": input_file,
                "saveto": saveto,
                "date": pd.to_datetime("today"),
            }
            df.loc[len(df.index)] = new_row

    df.to_csv("../data/result.csv", index=False)
    print(df[["algorithm", "n_clusters", "cost", "input"]].tail(10))
    print(f"Finished in {time.time() - start_time} seconds.")
