"""Future complete preprocessing script for the project. For now only filters the synchretic cases."""
from sklearn.cluster import KMeans
import fasttext as ft
import numpy as np
import time
import json

if __name__ == '__main__':
    start_time = time.time()

    print('Loading data...')
    list_of_all_forms = np.loadtxt(
        "../data/list_of_noun_forms_full.csv", dtype=object, delimiter=",")
    list_forms = list_of_all_forms[:, 4:] # List of cases without lemma and tags.

    # Binary coding of synchretisms. A form at index i,j is not syncretic if matrix_of_syncr[i][j]==2**j.
    matrix_of_syncr = np.loadtxt(
        "../data/matrix_of_syncr.csv", dtype=float, delimiter=","
    ).astype(int)

    # Load fasttext model.
    ft = ft.load_model("../data/final-model.bin")

    print('Retrieving non-synchratic case vectors from model...')
    # Get case vectors for all non-synchretic cases. 
    nonsynchr_casevectors = []
    for form in range(0,len(list_forms)):
        for case in range(0, 14):
            if matrix_of_syncr[form][case]==2**case: # Get word vector if case is not synchretic.
                vector = ft.get_word_vector(list_forms[form][case])
                nonsynchr_casevectors.append(vector)

    # Save the case vectors.
    np.save('../data/nonsynchr_casevectors', nonsynchr_casevectors)
    print(f"Preprocessing done! \nCompleted in {time.time() - start_time}.")
