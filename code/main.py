"""Run and evaluate clustering algorithms.
 
	Usage:
		main.py <input> <combinations> <saveto> <gold_labels>
  
	Arguments:
		input   path to input data - str
		combinations   List of clustering algorithms and number of clusters/epsilon. Valid algos are 'kmeans', 'dbscan', 'ward'. Example: Example: ['kmeans_2', 'dbscan_0.5', 'ward_2'] - list
        saveto  path to save output to - str
        gold_labels path to true labels of the input - str
""" 
import time
import numpy as np
from docopt import docopt
from typing import Tuple
from tqdm import tqdm
from evaluation import evaluate_clusters

def valid_algorithm(algorithm: str) -> bool:
    """Check whether clustering algorithm is implemented.
    Current valid algorithms are 'kmeans', 'dbscan' and 'ward'.
    """
    if str(algorithm) not in ["kmeans", "dbscan", "ward"]:
        print(f"Invalid algorithm: {algorithm}. Must be 'kmeans', 'dbscan' or 'ward'.")
        return False
    return True

def get_params(combination: str) -> Tuple[str, str, str]:
    """Get parameter combinations from string."""
    params = combination.split("_")
    algorithm = params[0]
    n_clusters = int(params[1])
    return algorithm, n_clusters

if __name__ == '__main__':
    args = docopt(__doc__)
    input = args['<input>']
    combinations = args['<combinations>']
    saveto = args['<saveto>']
    gold_labels = args['<gold_labels>']

    start_time = time.time()
    word_embeddings = np.load(input)
    gold_labels = np.load(gold_labels)

    for combination in tqdm(combinations):
        algorithm, n_clusters = get_params(combination)

        if valid_algorithm(algorithm): 
            print(f"Running {algorithm} with {n_clusters} clusters.")
            from clustering import algorithm # TODO: Is this inefficient for e.g. running kmeans trice?

            labels = algorithm(input, n_clusters)
            result = evaluate_clusters(name = algorithm, estimated_labels = labels, true_labels= gold_labels, saveto)

            # TODO: save eval results to saveto
            print(f"Done. Saving results to {saveto}.")
    print(f"Finished in {time.time() - start_time} seconds.")