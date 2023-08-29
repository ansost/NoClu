

def cluster_kmeans(n_clusters, input):
    """Cluster the input into n_clusters using kmeans.
    
    Parameters
    ----------
    n_clusters: int
        Number of clusters to cluster the input into.
    input: array-like
        Input to cluster.
        
    Returns
    -------
    labels: array-like
        Cluster labels for each element in the input.
    centers: array-like
        Cluster centers.
    """
    x = np.ascontiguousarray(input) # Kmeans function copies input if input is not c-contigous.
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(x)
    return kmeans.labels_, kmeans.cluster_centers_