"""Get list of gold labels and corresponding forms from full list of forms."""
import numpy as np
import pickle

# Load data.
list_of_all_forms = np.loadtxt(
    "../data/list_of_noun_forms_full.csv", dtype=object, delimiter=","
)
list_forms = list_of_all_forms[:, 4:]
matrix_of_syncr = np.loadtxt(
    "../data/matrix_of_syncr.csv", dtype=float, delimiter=","
).astype(int)

case2label = {
    "sing_nomn": 0,
    "sing_gent": 1,
    "sing_datv": 2,
    "sing_accs": 3,
    "sing_ablt": 4,
    "sing_loct": 5,
    "sing_gen2": 6,
    "plur_nomn": 7,
    "plur_gent": 8,
    "plur_datv": 9,
    "plur_accs": 10,
    "plur_ablt": 11,
    "plur_loct": 12,
    "plur_gen2": 13,
}

index2case = {
    0: "sing_nomn",
    1: "sing_gent",
    2: "sing_datv",
    3: "sing_accs",
    4: "sing_ablt",
    5: "sing_loct",
    6: "sing_gen2",
    7: "plur_nomn",
    8: "plur_gent",
    9: "plur_datv",
    10: "plur_accs",
    11: "plur_ablt",
    12: "plur_loct",
    13: "plur_gen2",
}

# Make gold labels.
# label = numbers that describes the case of the noun realization.
# case = grammatical description of the noun realization (e.g. singular nominative).
# form = noun realization (e.g. "дом").
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

# save with pickle
with open("../data/gold_labels_2d.pkl", "wb") as f:
    pickle.dump(gold_labels, f)
with open("../data/gold_forms_2d.pkl", "wb") as f:
    pickle.dump(gold_forms, f)

flat_labels = [item for sublist in gold_labels for item in sublist]
flat_forms = [item for sublist in gold_forms for item in sublist]
np.save("../data/gold_labels_1d.npy", flat_labels)
np.save("../data/gold_forms_1d.npy", flat_forms)
