"""
Get gold_labels, forms and vectors from original data.

Usage: python3 preprocess.py

This script preprocesses the data for usage with the 'pca_tsne.py' script.
Throughout the script the following terminology is used:
- label = numbers that describes the case of the noun realization.
- case = grammatical description of the noun realization (e.g. singular nominative).
- form = noun realization (e.g. "дом").

'list_of_noun_forms_full.csv'
is a list of all noun forms in the fasttext model that contains the columns
lemma, unkown number, grammatical gender, animacy, and 14 columns for each case (see further below for list of cases).
Example: человек,2723.0,masc,anim,человек,человека,человеку,человека,человеком,человеке,человека,люди,людей,людям,людей,людьми,людях,людей
The first four colums are filtered out to leave only the columns with the cases (variable: 'list_forms').
The variable 'list_forms' has a shape of (11515, 14).

'matrix_of_syncr.csv'
is a matrix with binary coding that describes which forms are syncretic out of the 14 forms for each noun.
Each form corresponds to a power of 2 from 2^0 to 2^13. When a form is synchretic, they share the same power of 2.

The script filters out the forms in 'list_forms' that are synchretic using 'matrix_of_syncr'.
As a by-product, the gold labels are saved as numbers ('goldLabels') and the gold forms are saved as strings ('goldForms').
This is done using two dicitonaries that map the cases to numbers and vice versa: 'case2label' and 'index2case'.
The two gold lists are saved as:
'gold_labels_2d.pkl' and 'gold_labels_1d.npy' & 'gold_forms_2d.pkl' and 'gold_forms_1d.npy'.
These are 1 dimensional and 2 dimensional lists that only differ in that the 1d file is the flat version of the 2d file.
The code for this project only uses the 1 dimensional list, but the 2 dimensional list is kept for future use.
The use of two different files is due to the fact that numpy can not save 'jagged arrays' (i.e. arrays with different lengths)
see this link for more info: https://stackoverflow.com/questions/65165451/how-to-make-2d-jagged-array-using-numpy.

'data/final-model.bin'.
is a pre-trained fasttext model that contains the vectors for all the forms in 'list_of_noun_forms_full.csv'.
It is used to extract the vectors of the forms in 'goldForms' from the fasttext model, in other words the vectors of the non-syncretic forms.
The vectors are saved as 'nonsynchr_casevectors.npy'.
"""
import json

import pickle
import numpy as np
import fasttext as ft

from src.constants import *

if __name__ == "__main__":
    # Load data.
    listOfAllForms = np.loadtxt(LISTOFALLFORMS, dtype=object, delimiter=",")
    listForms = listOfAllForms[:, 4:]
    matrixOfSyncr = np.loadtxt(BINARYMATRIX, dtype=float, delimiter=",").astype(int)

    with open(CASE2LABEL, "r") as f:
        case2label = json.load(f)

    with open(INDEX2CASE, "r") as f:
        index2case = json.load(f)

    # Make gold labels.
    goldLabels = []
    goldForms = []

    for formIdx in range(0, len(listForms)):
        caseLabels = []
        caseForms = []

        for caseIdx in range(0, 14):
            # Get label  and form if they are not synchretic.
            if matrixOfSyncr[formIdx][caseIdx] == 2**caseIdx:
                form = listForms[formIdx][caseIdx]

                case = index2case[caseIdx]
                label = case2label[case]

                caseLabels.append(label)
                caseForms.append(form)

        goldLabels.append(caseLabels)
        goldForms.append(caseForms)

    with open(GOLDLABELS2D, "wb") as f:
        pickle.dump(goldLabels, f)
    with open(GOLDFORMS2D, "wb") as f:
        pickle.dump(goldForms, f)

    flatLabels = [item for sublist in goldLabels for item in sublist]
    flatForms = [item for sublist in goldForms for item in sublist]
    np.save(GOLDLABELS1D, flatLabels)
    np.save(FOLDFORMS1D, flatForms)

    # Retrieve vectors for gold forms.
    model = ft.load_model(FASTTEXTMODEL)
    vectors = []
    for sublist in flatForms:
        for label in sublist:
            vector = model.get_word_vector(label)
            vectors.append(vector)
    np.save(VECTORSOUT, vectors)
