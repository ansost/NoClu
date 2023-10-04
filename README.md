# Dimensionality reduction on Russian noun embedding clusters

by [**Anna Stein**](https://ansost.github.io)

**Project description:** 

Dimensionality reduction techniques are commonly used to reduce the dimensions of data before
clustering. High dimensional data usually leads to worse clustering results. However, in the process
of reducing the data, some data may get lost which can lead to poor performance of the clustering
algorithms. This project aims to investigate the effect of Principal Component Analysis and t-distributed
stochastic neighbor embedding on performance of clustering algorithms k-means and DBSCAN. Results
show that DBSCAN may be more susceptible to different numbers of dimensions of the input data when
its produced by t-SNE. No such effect us found for k-means. Overall, the number of PCA components
(of 95% and 99% variance) do not affect the performance of the clustering algorithms as strongly as the
dimension reduction by t-SNEdoes. Further, more detailed research is needed to confirm these findings.

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
├── notebooks
├── figures
.
```

</details>

## Getting the code

Either clone the [git](https://git-scm.com/) repository:

```sh
git clone git@github.com:ansost/NoClu.git
```

Or [download a zip archive](https://github.com/ansost/NoClu/archive/refs/heads/main.zip).

### Requirements

See `requirements.txt` for a full list of requirements.
The fastest way to install the requirements is using [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/#use-pip-for-installing) and a [virtual environment](https://docs.python.org/3/tutorial/venv.html) (like [venv](https://docs.python.org/3/library/venv.html)).
> Make sure to substitute <name_of_vev> with an actual name for your environment.

```sh
python3 -m venv <name_of_venv>
source <name_of_venv>/bin/activate
pip install -r requirements.txt
```

## Software implementation

All source code used to generate the results and figures in this paper are in the `src/` and `scripts` folder.
The calculations and figure generation are run in [Python](https://www.python.org/) scripts with [Python 3.8.10](https://www.python.org/downloads/release/python-3810/).

This repository uses pre-commit hooks. Learn more about them and how to install/use them here: [https://pre-commit.com/](https://pre-commit.com/).

Two optional scripts for producing plots are run in [Jupyter notebooks](https://jupyter.org/).

### Data

The primary data source is a pre-trained fastText model with word embeddings for Russian noun cases.
More information on the data used can be found in the preprocessing script (`scripts/preprocess.py`).

Since the model is very large and currently stored in git large file storage, please contact the author if you would like to use it. 

#### Preprocessing

Filter syncretic forms from the word embeddings and extract the vectors for the nonsynchretic forms. Also, gather the gold labels.

```sh
python3 preprocess.py
```

#### Dimensionality Reduction

> Note that you must navigate to the `scripts/` folder to run this script and the ones in the following sections.

Use just PCA or PCA followed by t-SNE to reduce the dimensions of the vectors. See the docstring of the script and the top of the config file (`data/config_files/npclu.py`) for more information on the input parameters.
> Note that all computations involving t-SNE may take time to run (1h+).

```sh
python3 reduce.py
```

#### Clustering and Evaluation

Cluster the low-dimensional data using `kmeans` and `DBSCAN`. Evaluate the results using a maximum flow minimum cost algorithm implemented in `networkx`.

```sh
python3 execute.py
```

Results are saved in `data/clustering_output/` and `data/result.csv`. An overview is printed in the command line.

#### Plotting

The thesescripts are **optional** scripts and notebooks that can be used to reproduce the plots in the report. They are optional to run the clustering and evaluation.

Run the following script to plot the results of the clustering and evaluation as a bar and line plot. The absolute value of the cost is displayed on the y-axis, while the number of clusters (for k-means) or the epsilon value (for DBSCAN) is displayed on the x-axis. The script output is saved in the 'figures/' folder.

```sh
python3 plot_results.py
```

The two notebooks in the `notebooks/` folder can be used to plot elbow plots to find the optimal number of clusters for k-means and the optimal epsilon value for DBSCAN. Additionally, there is a notebook for finding the optimal number of components for PCA.

### License

All source code is made available under a BSD 3-clause license. You can freely use and modify the code without warranty if you provide attribution to the authors. See `LICENSE.md` for the full license text.
The project report and slide presentation content are not open source. The author reserves the rights to the content.

#### Author: **Anna Stein**

#### Adviser: Yulia Zinova

Website: [https://ansost.github.io](https://ansost.github.io)

If you are having problems with anything regarding this repository, please write me email: [anna.stein@hhu.de](mailto:anna.stein@hhu.de)
