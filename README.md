# Dimensionality reduction on Russian noun embedding clusters

by [**Anna Stein**](https://ansost.github.io)

<details>
<summary>See repository structure</summary>

```bash
.
├── LICENSE
├── README.md
├── REQUIREMENTS.txt
├── src
├── scripts
├── data
├── figures
.
```

</details>

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

All source code used to generate the results and figures in this paper are in the `src/` and `scripts` folder.
The calculations and figure generation are run in [Python](https://www.python.org/) scripts.

### Data

The main data source is a pre-trained fasttext model with word embeddings for Russian noun cases.
More information on the data that was used can be found in the preprocessing script (`scripts/preprocessing.py`).

#### Preprocessing

Filter synchretic forms from the word embeddings and extract the vectors for the nonsynchretic forms. Also gather the gold labels.

```sh
python3 preprocessing.py
```

#### Dimensionality Reduction

Use just PCA or PCA followed by t-SNE to reduce the dimensions of the vectors. See the docstring of the script and the top of the config file (`data/config_files/npclu.py`) for more information on the input parameters.
> Note that all computations involving t-SNE may take a time to run (1h+).

```sh
python3 pca_tsne.py
```

#### Clustering and Evaluation

Cluster the low-dimensional data using `kmeans` and `DBSCAN`. Evaluate the results using a maximum flow minimum cost algorithm implemented in `networkx`.

```sh
python3 noclu.py
```

Results are saved in `data/clustering_output/` and `data/result.csv`. An overview is printed in the command line.

### License

All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See `LICENSE.md` for the full license text.
The project report and slide presentation content is not open source. The author reserves the rights to the content.

#### Author: **Anna Stein**

#### Adviser: Yulia Zinova

Website: [https://ansost.github.io](https://ansost.github.io)

If you are having problems with anything regarding this repository, please write me email: [anna.stein@hhu.de](mailto:anna.stein@hhu.de)
