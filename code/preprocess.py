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

'../data/matrix_of_syncr.csv' 
is a matrix with binary coding that describes which forms are syncretic out of the 14 forms for each noun. 
Each form corresponds to a power of 2 from 2^0 to 2^13. When a form is synchretic, they share the same power of 2.

The script filters out the forms in 'list_forms' that are synchretic using 'matrix_of_syncr'. 
As a by-product, the gold labels are saved as numbers ('gold_labels') and the gold forms are saved as strings ('gold_forms').
This is done using two dicitonaries that map the cases to numbers and vice versa: 'case2label' and 'index2case'.
The two gold lists are saved as:
'gold_labels_2d.pkl' and 'gold_labels_1d.npy' & 'gold_forms_2d.pkl' and 'gold_forms_1d.npy'.
These are 1 dimensional and 2 dimensional lists that only differ in that the 1d file is the flat version of the 2d file.
The code for this project only uses the 1 dimensional list, but the 2 dimensional list is kept for future use.
The use of two different files is due to the fact that numpy can not save 'jagged arrays' (i.e. arrays with different lengths)
see this link for more info: https://stackoverflow.com/questions/65165451/how-to-make-2d-jagged-array-using-numpy.

'../data/final-model.bin'.
is a pre-trained fasttext model that contains the vectors for all the forms in 'list_of_noun_forms_full.csv'.
It is used to extract the vectors of the forms in 'gold_forms' from the fasttext model, in other words the vectors of the non-syncretic forms.
The vectors are saved as 'nonsynchr_casevectors.npy'.
"""
import numpy as np
import fasttext as ft
import pickle
import time
import json

start_time = time.time()

# Load data.
list_of_all_forms = np.loadtxt(
    "../data/list_of_noun_forms_full.csv", dtype=object, delimiter=","
)
list_forms = list_of_all_forms[:, 4:]
matrix_of_syncr = np.loadtxt(
    "../data/matrix_of_syncr.csv", dtype=float, delimiter=","
).astype(int)

with open("../data/case2label.json", "r") as f:
    case2label = json.load(f)

with open("../data/index2case.json", "r") as f:
    index2case = json.load(f)

# Make gold labels.
gold_labels = []
gold_forms = []

for form_idx in range(0, len(list_forms)):
    case_labels = []
    case_forms = []

    for case_idx in range(0, 14):
        # Get label  and form if they are not synchretic.
        if matrix_of_syncr[form_idx][case_idx] == 2**case_idx:
            form = list_forms[form_idx][case_idx]

            case = index2case[case_idx]
            label = case2label[case]

            case_labels.append(label)
            case_forms.append(form)

    gold_labels.append(case_labels)
    gold_forms.append(case_forms)

with open("../data/gold_labels_2d.pkl", "wb") as f:
    pickle.dump(gold_labels, f)
with open("../data/gold_forms_2d.pkl", "wb") as f:
    pickle.dump(gold_forms, f)

flat_labels = [item for sublist in gold_labels for item in sublist]
flat_forms = [item for sublist in gold_forms for item in sublist]
np.save("../data/gold_labels_1d.npy", flat_labels)
np.save("../data/gold_forms_1d.npy", flat_forms)

# Retrieve vectors for gold forms.
model = ft.load_model("../data/final-model.bin")
vectors = []
for sublist in flat_forms:
    for label in sublist:
        vector = model.get_word_vector(label)
        vectors.append(vector)
np.save("../data/nonsynchr_casevectors.npy", vectors)
print(f"Preprocessing done! \nCompleted in {time.time() - start_time}.")
