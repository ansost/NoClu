# Dimensionality reduction on Russian noun embedding clusters

by [**Anna Stein**](https://ansost.github.io)

Evaluating clusters formed by dimensionality reduction algorithms (PCA, t-SNE) on the output of clustering algorithms (k-means, ...) by using a min cost flow approach.

## Software implementation

All source code used to generate the results and figures in this paper are in the `code/` folder.
The calculations and figure generation are run in both [Jupyter notebooks](http://jupyter.org/) and [Python](https://www.python.org/) scripts.
```bash
.
├── LICENSE
├── README.md
├── REQUIREMENTS.txt
├── code 
│   ├── clustering.py
│   ├── evaluation.py
│   ├── noclu.py
│   ├── preprocessing
│   │   ├── gold_labels.py
│   │   ├── oldpreprocess.py
│   │   └── pca_tsne.py
│   └── run_noclu.sh
├── data
│   ├── case2label.json
│   ├── clustering_output
│   ├── dim_reduced_input
│   │   ├── nonsynchr_pca_200.npy
│   │   ├── nonsynchr_pca_300.npy
│   │   ├── nonsynchr_pca_80.npy
│   │   ├── nonsynchr_pcatsne_200_2.npy
│   │   ├── nonsynchr_pcatsne_200_3.npy
│   │   ├── nonsynchr_pcatsne_300_2.npy
│   │   ├── nonsynchr_pcatsne_300_3.npy
│   │   ├── nonsynchr_pcatsne_80_2.npy
│   │   └── nonsynchr_pcatsne_80_3.npy
│   ├── gold_forms_2d.pkl
│   ├── gold_labels_2d.pkl
│   ├── index2case.json
│   ├── list_of_noun_forms_full.csv
│   ├── matrix_of_syncr.csv
│   ├── nonsynchr_casevectors.npy
│   ├── result.csv
├── figures
│   └── PCA_variance.png
.
```

### Data

DESCRIPTION OF DATA HERE

## Getting the code

Either clone the [git](https://git-scm.com/) repository:
```sh
git clone git@github.com:ansost/NoClu.git
```
or [download a zip archive](https://github.com/ansost/NoClu/archive/refs/heads/main.zip).

## Install dependencies
In the root folder of the repository, install the Python requirements using [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing):
```sh
pip install -r requirements.txt
```

## Reproducing the results

DESCRIPTION HERE

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The paper text is not open source. The author reserves the rights to the
paper content.

## Author

**Anna Stein**

Website: [https://ansost.github.io](https://ansost.github.io)

If you are having problems with anything regarding this repository, please write me email: [anna.stein@hhu.de](mailto:anna.stein@hhu.de)
