"""Filter synchratic labels and retrieve nonsynchratic case vectors from fasttext model."""
import time
import pickle
import fasttext as ft
import numpy as np


if __name__ == "__main__":
    start_time = time.time()

    print("Loading data...")
    list_of_all_forms = np.loadtxt(
        "../data/list_of_noun_forms_full.csv", dtype=object, delimiter=","
    )

    # List of cases without lemma and tags.
    list_forms = list_of_all_forms[:, 4:]

    # Binary coding of synchretisms. A form at index i,j is not syncretic if matrix_of_syncr[i][j]==2**j.
    matrix_of_syncr = np.loadtxt(
        "../data/matrix_of_syncr.csv", dtype=float, delimiter=","
    ).astype(int)

    # Load fasttext model.
    model = ft.load_model("../data/final-model.bin")

    print("Filtering synchretic cases from labels...")
    # Get case vectors for all non-synchretic cases.
    nonsynchr_caselabels = []
    for form in range(0, len(list_forms)):
        sublist = []
        for case in range(0, 14):
            if (
                matrix_of_syncr[form][case] == 2**case
            ):  # Get label if label is not synchretic.
                label = list_forms[form][case]
                sublist.append(label)
        nonsynchr_caselabels.append(sublist)

    with open(
        "../data/nonsynchr_labels.pkl", "wb"
    ) as f:  # Can't save a jagged array with numpy.
        pickle.dump(nonsynchr_caselabels, f)

    print("Retrieving non-synchratic vectors from model...")
    vectors = []
    for sublist in nonsynchr_caselabels:
        for label in sublist:
            vector = model.get_word_vector(label)
            vectors.append(vector)

    np.save("../data/nonsynchr_casevectors_test.npy", vectors)
    print(f"Preprocessing done! \nCompleted in {time.time() - start_time}.")
