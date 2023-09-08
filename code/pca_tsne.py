"""Reduce dimensions for word embeddings using PCA and t-SNE.
The 'conditions' variable specifies the number of dimensions/components 
and the reduction algorithm (PCA or PCA + t-SNE). The filename also
indicates the algorithm and the number of dimensions/components used. 

Example:
'pca_300' - PCA with 300 components
'pcatsne_300_2' - PCA with 300 components followed by t-SNE with 2 dimensions
""" 
from typing import List, Tuple
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
    pca = PCA(n_components = components)
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
    tsne = TSNE(n_components=dims)
    result = tsne.fit_transform(vectors)
    return result

def get_params(combination: str) -> Tuple(str, str, str):
    params = combination.split('_')
    algo = params[0]
    components = params[1]
    dims = None
    if len(params) == 3:
        dims = params[2]
    return algo, components, dims

def valid_algo(algo):
    if algo != 'pca' or algo != 'pcatsne': 
        print(f"Invalid algorithm: {combination}. Must be 'pca' or 'pcatsne'.") 
        return False
    return True

if __name__ == '__main__':
    conditions = [
    'pca_300', 
    'pca_200', 
    'pca_80',
    'pcatsne_300_2',
    'pcatsne_300_3',
    'pcatsne_200_2',
    'pcatsne_200_3',
    'pcatsne_80_2',
    'pcatsne_80_3'
    ]

    word_embeddings = np.load("../../data/nonsynchr_casevectors.npy")

    print('Reducing data for combination...')
    for combination in tqdm(conditions): 
        algo, components, dims = get_params(combination)
        if valid_algo(algo):
            results = run_pca(vectors = word_embeddings, components = components)
        
            if algo == 'pcatsne':
                if dims:
                    results = run_tsne(vectors = results, tsne_dims = dims)
            np.save("../../data/dim_reduced/nonsynchr_" + combination + ".npy", results)