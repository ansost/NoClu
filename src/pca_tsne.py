"""
Exports functions for running PCA and t-SNE:
    - run_pca: run dimensionality reduction using Principal Component Analysis.
    - run_tsne: run dimensionality reduction by t-SNE on data produced from PCA.
"""
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_pca(vectors: ArrayLike, components: int) -> ArrayLike:
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


def run_tsne(vectors: ArrayLike, dims: int) -> ArrayLike:
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
