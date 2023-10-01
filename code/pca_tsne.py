"""Reduce dimensions for word embeddings using PCA and t-SNE.
The 'conditions' variable specifies the reduction algorithm
(PCA or PCA + t-SNE) and the number of dimensions/components.
The filename also indicates the algorithm and the number of
dimensions/components used.

Example:
'pca_300' - PCA with 300 components
'pcatsne_300_2' - PCA with 300 components followed by t-SNE with 2 dimensions
"""
from typing import Tuple
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_pca(vectors: np.array, components: int) -> np.array:
    """Run dimensionality reduction using Principal Component Analysis.

    Parameters
    ----------
    vectors:
        Vectors to reduce.
    components:
        Number of dimensions to reduce to.

    Returns
    -------
    results:
        Reduced vectors.
    """
    pca = PCA(n_components=components)
    results = pca.fit_transform(vectors)
    return results


def run_tsne(vectors: np.array, dims: int) -> np.array:
    """Run dimensionality reduction using Principal Component Analysis and t-SNE in that order.

    Parameters
    ----------
    vectors:
        Vectors to reduce.
    dims:
        Number of dimensions to reduce to using t-SNE.

    Returns
    -------
    tsne_result:
        Reduced vectors.
    """
    tsne = TSNE(n_components=dims, init="pca", random_state=0)
    result = tsne.fit_transform(vectors)
    return result


def get_params(combination: str) -> Tuple[str, str, str]:
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
    with open("../data/config_files/pca_tsne.config", "r") as f:
        config = yaml.safe_load(f)
        conditions = config["conditions"]
        input = config["input"]

    assert input[-3:] == "npy", "Input file must be .npy file."
    word_embeddings = np.load(input)

    print("Reducing data for combination...")
    for combination in tqdm(conditions):
        algo, components, dims = get_params(combination)
        print(combination)
        if valid_algorithm(algo):
            results = run_pca(vectors=word_embeddings, components=components)

            if algo == "pcatsne":
                if dims:
                    results = run_tsne(vectors=results, dims=dims)
            np.save(
                "../data/dim_reduced_input/nonsynchr_" + combination + ".npy", results
            )
