#!/bin/bash

# Usage:
#     noclu.py <input> <combinations> <saveto>

# Arguments:
#     input   path to input data - str
#     combinations   List of clustering algorithms and number of clusters/epsilon. Valid algos are 'kmeans', 'dbscan', 'ward'. Example: ['kmeans_2', 'dbscan_0.5', 'ward_2'] - list
#     saveto  path to save output to - str
#     gold_labels path to true labels of the input - str
#../data/dim_reduced_input/nonsynchr_pca_80.npy \

python3 noclu.py  \
    ../data/tests/test_data.npy \
    'kmeans_9' \
    ../data/clustering_output/