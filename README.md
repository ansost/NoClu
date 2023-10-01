# Dimensionality reduction on Russian noun embedding clusters

by [**Anna Stein**](https://ansost.github.io)

TODO: Add description and slides link.

## Getting the code

Either clone the [git](https://git-scm.com/) repository:

```sh
git clone git@github.com:ansost/NoClu.git
```

or [download a zip archive](https://github.com/ansost/NoClu/archive/refs/heads/main.zip).

### Requirements

See `requirements.txt` for a full list of requirements.
The fastest way to install the requirements is using [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing) and a [virtual enviornment](https://docs.python.org/3/tutorial/venv.html) (like [venv](https://docs.python.org/3/library/venv.html)).
> Make sure to substitue <name_of_vev> with an actual name for your environment.

```sh
python3 -m venv <name_of_venv>
source <name_of_venv>/bin/activate
pip install -r requirements.txt
```

## Software implementation

All source code used to generate the results and figures in this paper are in the `code/` folder.
The calculations and figure generation are run in both [Jupyter notebooks](http://jupyter.org/) and [Python](https://www.python.org/) scripts.

### Data

TODO: Add data description.

#### Preprocessing

Filter synchretic forms from the word embeddings and extract the vectors for the nonsynchretic forms. Also gather the gold labels.

```sh
python3 preprocessing.py
```

#### Dimensionality Reduction

Use just PCA or PCA followed by t-SNE to reduce the dimensions of the vectors. See the docstring of the script and the top of the config file (`data/config_files/npclu.py`) for more information on the input parameters.
> Note that all computations involving t-SNE take a long time to run (1h+).

```sh
python3 pca_tsne.py
```

#### Clustering and Evaluation

Cluster the low-dimensional data using `kmeans`, `DBSCAN` or `Ward's` (agglomerative clustering). Evalaute the results using a maximum flow minimum cost algorithm implemented in `networkx`.
> Note that since the sklearn implementation of [Ward's clustering needs O(n²) memory](https://stackoverflow.com/questions/55316093/memory-error-while-doing-hierarchical-clustering) and you most likely need to use a HPC system to run the computation.

```sh
python3 noclu.py
```

Results are saved in `data/clustering_output/` and `data/result.csv`. An overview is printed in the command line.

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

### License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The paper text is not open source. The author reserves the rights to the
paper content.

#### Author: **Anna Stein**

#### Adviser: Yulia Zinova

Website: [https://ansost.github.io](https://ansost.github.io)

If you are having problems with anything regarding this repository, please write me email: [anna.stein@hhu.de](mailto:anna.stein@hhu.de)
