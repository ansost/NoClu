# Dimensionality reduction on Russian noun embedding clusters

by [**Anna Stein**](https://ansost.github.io)

Evaluating clusters formed by dimensionality reduction algorithms (PCA, t-SNE) on the output of clustering algorithms (k-means, ...) by using a min cost flow approach.

Approximate workflow:
1. Remove synchretic word embeddings from the data
  -  I dont get the coding of the synchretic matrix a 100% but I can hopefully just use your function
2. Implement min cost flow
  - I will probably use the library NetworkX using their function for that since Lemon is only available in c++ and I could not find a port to
    Python
3. Use a clustering algorithm (k-means, DBSCAN, Ward's Method, BICO algorithm) on the non-synchretic word embeddings
4. Use PCA (optionally with tsne) with some n_dimensions
5. Input true labels from embeddings and (reduced) clusters from clustering algorithm as input for the min cost flow implementation
6. Repeat for different clustering algorithms and n_dimensions for PCA & tsne

## Software implementation

All source code used to generate the results and figures in this paper are in the `code/` folder.
The calculations and figure generation are run in both [Jupyter notebooks](http://jupyter.org/) and [Python](https://www.python.org/) scripts.

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

## Workflow

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
