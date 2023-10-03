"""
Reduce dimensions for word embeddings using PCA and t-SNE.

Usage:
    python3 pca_tsne.py

Input for the script is a config file, which contains the following parameters:
- conditions:
    specifies the reduction algorithm (PCA or PCA + t-SNE) and the number of
    dimensions/components. The filename also indicates the algorithm and the number of
    dimensions/components used.
- input:
    path to input file (word embeddings)
For more information on the input see the configuration file ('data/config_files/pca_tsne.config').

This script saved the lower dimensions vectors as .npy files in the folder '/data/dim_reduced_input' with the
combination as filename.
"""
from typing import Tuple, Union

import yaml
import numpy as np
from tqdm import tqdm

from src.constants import *
from src.pca_tsne import run_pca, run_tsne


def get_params(combination: str) -> Tuple[str, int, Union[str, None]]:
    """Get parameter combinations from str filename.
    See Module docstring for more information on parameter combinations."""
    params = combination.split("_")
    algo = params[0]
    components = int(params[1])
    dims = None
    if len(params) == 3:
        dims = int(params[2])
    return algo, components, dims


def valid_algorithm(algo: str) -> bool:
    """Check whether str filename contains valid algorithm (combination).
    Current valid algorithms are either 'pca' or 'pcatsne'.
    """
    if str(algo) != "pca" and str(algo) != "pcatsne":
        print(f"Invalid algorithm: {algo}. Must be 'pca' or 'pcatsne'.")
        return False
    return True


if __name__ == "__main__":
    print("Loading config file...")
    with open(PCATSNECONFIG, "r") as f:
        config = yaml.safe_load(f)
        conditions = config["conditions"]
        input = config["input"]

    if not input.endswith("npy"):
        raise Exception("Input file must be .npy file.")
    wordEmbeddings = np.load(input)

    for combination in tqdm(conditions):
        algo, components, dims = get_params(combination)
        if valid_algorithm(algo):
            results = run_pca(vectors=wordEmbeddings, components=components)

            if algo == "pcatsne":
                if dims:
                    results = run_tsne(vectors=results, dims=int(dims))
            np.save(DIMREDOUT + combination + ".npy", results)
