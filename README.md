# CoqGym-GNN
GNN extension and improvements on [CoqGym](https://github.com/princeton-vl/CoqGym) repository.

![Example proof](images/example_proof.jpg)

Note, this repo includes the codebase of Coq, SerAPI, CoqHammer, and the Coq projects in coq_projects.


## Table of Contents

0. [Notation](#0-notation)
1. [Goals](#1-goals)
2. [Main Contributions](#2-main-contributions)
    - [Pipeline Modifications](#21-pipeline-modifications)
    - [Design Modifications](#22-design-modifications)
4. [Setup and Installation](#3-setup-and-installation)
3. [FAQ and Known Bugs](#4-faq-and-known-bugs)
5. [Resources](#5-resources)
6. [Reproducing Our Work](#6-reproducing-our-work)
7. [Future Work](#7-future-work)


## 0. Notation

Here are some important notation to understand the below explanations:

- $G = (V, E)$: Graph notation, where $|V|$ is the number of vertices (nodes) and $|E|$ is the number of edges.
- `x`: $\mathbb{R} ^{|V|}$ node list of node types, referenced by index into the non-terminal node information. Child-first ordering enforced by `traverse_postorder`
- `edge_index`: $\mathbb{R} ^ 2 \times \mathbb{R}^{|E|}$ edge list referenced by index into `x`.

## 1. Goals

With the advent of recent progress in graph neural networks (GNNs), we hope to improve on the original CoqGym results by replacing their TreeLSTM module with various GNN implementations.

## 2. Main Contributions

The main contributions of this repository are some pipeline modifications to allow for the GNNs, and the actual GNN implementation. In the following sections, these modifications will be presented in detail with comparisons to the original when relevant.

These implementations are split between many branches to provide easier management of different tests:
- [`master`](https://github.com/danjenson/CoqGym-GNN/tree/master): Contains the original CoqGym paper at the time of the fork.
- [`bofb`](https://github.com/danjenson/CoqGym-GNN/tree/bofb): Contains first implementation of [graph batches](#21-pipeline-modifications)
- [`rl-mods`](https://github.com/danjenson/CoqGym-GNN/tree/rl-mods): Modified decoder with more expressive attention mechanism
- [`int-emb`](https://github.com/danjenson/CoqGym-GNN/tree/int-emb): Contains implementation of `IntegerFeatureEncoder` in the encoder and the modifications from `rl-mods`.

### 2.1 Pipeline Modifications

In order to learn using GNNs efficiently, `x` and `edge_index` information need to be extracted from the `lark.tree.Tree` representations of ASTs. This computation is very costly, so it is delegated to the proof extraction stage. Here are the pipeline modifications that facilitate this change:

- Modified proof step data representation from `dict` to `torch_geometric.data.Batch` objects
- Modified merge operation to facilitate `Batch` merging
- Used `torch_geometric.data.Dataset` object over `torch.data.Dataset` object
- Changed saving protocol from `.pickle` to PyTorch-optimized `.pt` using `torch.save()`

More explicitly, a comparison of the proof step structures is outlined below:

**CoqGym proof step**:
```python
{
    file : str,
    proof_name : str,
    n_step : int,
    env : [
        {
            qualid: str,
            ast : lark.tree.Tree
        },
        ...
    ],
    local_context : [
        {
            ident : str
            ast : lark.tree.Tree
        },
        ...
    ],
    goal : lark.tree.Tree,
    tactic_actions : int | str,
    tactic_str : str,
}
```

**CoqGym-GNN proof step**:
```python
torch_geometric.data.Batch (
    x : tensor.Tensor,
    edge_index : tensor.Tensor,
    batch : tensor.Tensor,
    # Some modifications to original CoqGym attributes
    file : str,
    proof_name : str,
    n_step : int,
    env : [
        {
            qualid: str,
            ast : lark.tree.Tree
        },
        ...
    ],
    local_context : [
        {
            ident : str
            ast : lark.tree.Tree
        },
        ...
    ],
    goal : lark.tree.Tree,
    tactic_actions : int | str,
    tactic_str : str,

)
```

Along with these changes, optimizations were made to the data generation process which facilitates easier updates to the extracted dataset and lighter computational resource requirements. Specifically:

- Added `filter_file` option in `iter_proofs` and its derivatives to skip loading of proof if not being considered
- Save files as data is generated, so a monolithic list is not needed to keep track of data.
- Added multi-processing script for both `extract_proof.py` and `evaluate.py` for more efficient use of computational resources.

### 2.2 Design Modifications


TODO: process below modifications
- Implement 2-layer GNN with modular convolutions
- Integer embeddings of nodes
- Added layer-norms and Prelu activations to attention mechanism

## 3. Setup and Installation

CoqGym has many dependencies and is nontrivial to set up correctly. The following instruction detail how to obtain the CoqGym dataset and build the interaction environment natively.

### 3.1 Required dependencies

- [OCaml Package Manager](https://opam.ocaml.org/) (OPAM) is used to install OCaml and the corresponding packages.
- [Anaconda Distribution](https://www.anaconda.com/products/distribution) is used to install python and manage python-specific packages. 
- [Lightning Memory-Mapped Database](https://www.symas.com/lmdb) (LMDB) is used to store S-expressions in `*.json` files.
- [Ruby](https://www.symas.com/lmdb) is used for xyz

### 3.2 Building Coq, SerAPI, CoqHammer, and the Coq Projects

1. Create an OPAM switch for OCaml 4.07.1+flambda: `opam switch create 4.07.1+flambda && eval $(opam env)`
1. Clone the repository: `git clone https://github.com/danjenson/CoqGym-GNN.git`
1. Install Coq, SerAPI and CoqHammer: `cd CoqGym && source install.sh`
1. Build the Coq projects (can take a while): `cd coq_projects && make && cd ..`
1. Create and activate the conda environment: `conda env create -f coq_gym.yml && conda activate coq_gym`

*Note*: [Coq](https://github.com/coq/coq), [SerAPI](https://github.com/ejgallego/coq-serapi), [CoqHammer](https://github.com/lukaszcz/coqhammer), and the Coq projects in [coq_projects](./coq_projects) directory are indendent software projects with their own code repositories, but please follow the instructions above to build the specific versions we need.

### 3.3 Downloading the pre-extracted proofs (our data)

1. Download the CoqGym dataset [here](https://drive.google.com/drive/folders/149m_17VkYYkl0kdSB4AI8zodCuTmPaA6?usp=sharing)

2. Unzip the data and set the paths: `python unzip_data.py`

*Caveat*: The second step sets the absolute paths in the data. You have to re-do it whenever the absolote path of the `data/` directory changes (e.g. after moving the entire repo to another directory).

### 3.4 Check whether everything is setup correctly

Run `python eval_env.py` to check if it terminates normally without raising an error.

Now you are ready to interact with CoqGym! 

## 4 Running train and testing pipelines

### 4.1 Extracting proof steps

Our encoder-decoder models are trained on individual proof stps
### 4.2 Training the encoder-decoder models

Here we describe how to train our models on CoqGym.
xx 










## 4. FAQ and Known Bugs



## 5. Resources

<!-- Add resources on pre-built data and  -->

## 6. Reproducing Our Work


## 7. Future Work
