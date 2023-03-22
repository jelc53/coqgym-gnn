# CoqGym-GNN
GNN extension and improvements on [CoqGym](https://github.com/princeton-vl/CoqGym) repository.

![Example proof](images/example_proof.jpg)

Note, this repo includes the codebase of Coq, SerAPI, CoqHammer, and the Coq projects in coq_projects.


## Table of Contents


0. [Notation](#0-notation)
1. [Project Goals](#1-goals)
2. [Main Contributions](#2-main-contributions)
    - [Pipeline Modifications](#21-pipeline-modifications)
    - [Design Modifications](#22-design-modifications)
3. [Setup and Installation](#3-setup-and-installation)
4. [Running train and test pipelines](#4-running-train-and-testing-pipelines)
5. [FAQ and Known Bugs](#5-faq-and-known-bugs)
6. [Resources](#6-resources)


## 0. Notation

Here are some important notation to understand the below explanations:

- $G = (V, E)$: Graph notation, where $|V|$ is the number of vertices (nodes) and $|E|$ is the number of edges.
- `x`: $\mathbb{R} ^{|V|}$ node list of node types, referenced by index into the non-terminal node information. Child-first ordering enforced by `traverse_postorder`
- `edge_index`: $\mathbb{R} ^ 2 \times \mathbb{R}^{|E|}$ edge list referenced by index into `x`.


## 1. Project Goals

With the advent of recent progress in graph neural networks (GNNs), we hope to improve on the original CoqGym results by replacing their TreeLSTM encoder module with various GNN implementations.


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
    tactic_actions : list[int | str],
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
            ident : str,
            text : str,
            ast : lark.tree.Tree
        },
        ...
    ],
    goal : {
        id : int,
        text : str,
        ast : lark.tree.Tree
    }
    tactic_actions : int | str,
    tactic_str : str,
)
```

Along with these changes, optimizations were made to the data generation process which facilitates easier updates to the extracted dataset and lighter computational resource requirements. Specifically:

- Added `filter_file` option in `iter_proofs` and its derivatives to skip loading of proof if not being considered
- Save files as data is generated, so a monolithic list is not needed to keep track of data.
- Added multi-processing script for both `extract_proof.py` and `evaluate.py` for more efficient use of computational resources.

### 2.2 Design Modifications

Some notable design modifications in both the new GNN encoder and the existing RL pipeline were made for testing purposes, which are listed below

- 2-layer GNN with modular convolutions
  - Multi-headed Graph Attention (GAT) or GraphSage convolutions
- Used `torch_geometric.graphgym.models.encoder.IntegerFeatureEncoder` over one-hot encodings of node types
- Increased expressiveness of attention module in the decoder
  - Added an extra layer
  - Used PreLu activations between each layer
  - Used batch normalization within each layer


## 3. Setup and Installation

CoqGym has many dependencies and is nontrivial to set up correctly. The following instruction detail how to obtain the CoqGym dataset and build the interaction environment natively.

### 3.1 Required dependencies

- [OCaml Package Manager](https://opam.ocaml.org/) (OPAM) is used to install OCaml and the corresponding packages.
<!-- - [Anaconda Distribution](https://www.anaconda.com/products/distribution) is used to install python and manage python-specific packages.  -->
- [Lightning Memory-Mapped Database](https://www.symas.com/lmdb) (LMDB) is used to store S-expressions in `*.json` files.
<!-- - [Ruby](https://www.symas.com/lmdb) is used for xyz -->

### 3.2 Building Coq, SerAPI, CoqHammer, and the Coq Projects

1. Create an OPAM switch for OCaml 4.07.1+flambda: `opam switch create 4.07.1+flambda && eval $(opam env)`
1. Clone the repository: `git clone https://github.com/danjenson/CoqGym-GNN.git`
1. Install Coq, SerAPI and CoqHammer: `cd CoqGym && source install.sh`
1. Build the Coq projects (can take a while): `cd coq_projects && make && cd ..`
1. Setup the python environment (see requirements.txt for version details):
  - `curl https://pyenv.run | bash`
  -  `pyenv install 3.7.1 && pyenv local 3.7.1`
  - `pip install numpy ipython lark-parser==0.6.5 lmdb==0.94 pandas==0.24.2 pexpect==4.6.0 sexpdata==0.0.3 progressbar2`
  - `pip install torch pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html`
<!-- *Note*: If you would prefer to use Anaconda run: `conda env create -f coq_gym.yml && conda activate coq_gym` -->

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

Our encoder-decoder models are trained on individual proof steps rather than entire proofs. This allows use to directly use teacher forcing.

To extract proof steps from the CoqGym dataset, run `python extract_proof_steps.py` from the ASTactic directory. Note, this can take a while (8-12 hours). To help, we provide an alternate multiprocessing script to parallelize extraction across proof libraries (coq projects) `python multiprocess_extract.py`.

The extracted proof steps are in proof_steps/. You can double-check the number of proof steps to make sure everything works as expected:
Directory |  # files
------------ | -------------
proof_steps/train | 121,644
proof_steps/valid | 68,180

We also provide pre-extracted download tarballs for `train` and `valid` proof steps [here](https://drive.google.com/drive/folders/16xgR02z174s1CC1askWoZrpNUhS_yzpA).

### 4.2 Training the encoder-decoder models

To train on the proof steps in training + validation set, run the following command from the ASTactic directory:
`python main.py --no_validation --exp_id <model_id> --model_type <model_type> --heads <num_heads>`

Model checkpoints will be saved to runs/astactic/checkpoints/. See options.py for command line options.

CoqGym's pre-trained astatic model can be downloaded [here](https://drive.google.com/drive/folders/1AzLaEpoGS3BPMUz9Bl63MHAFRqlF4CtH?usp=sharing).

Our pre-trained GNN models can be downloaded [here](https://drive.google.com/drive/folders/16xgR02z174s1CC1askWoZrpNUhS_yzpA)

### 4.3 Testing the rl agent

To test a trained model on unseen proof libraries, run the following command from the ASTactic directory:
`python evaluate ours <model_id> --path runs/<model_id>/checkpoints/model_<epoch#>.pth --filter <proof_library_name>`

* To execute testing just a single proof (e.g. `get_set_name` from `../data/StructTact/Assoc.json`):
  `python evaluate.py ours ours-TEST --path runs/astactic/checkpoints/model_003.pth --file ../data/StructTact/Assoc.json --proof "get_set_same"`

* Testing an automated tactic X (may be "auto", "trivial", "easy", "intuition", or "hammer"):
    `python -u evaluate.py X X-TEST --file ../data/StructTact/Assoc.json --proof "get_set_same"`

* Testing ASTactic+X:
    `python -u evaluate.py ours+X ours+X-TEST --path runs/astactic/checkpoints/model_003.pth --file ../data/StructTact/Assoc.json --proof "get_set_same"`

*Caveat*: Testing is computationally expensive, but the workloads are very parallelizable. We provide the code for this in `multiprocess_test.py`.


## 5. FAQ and Known Bugs

- Error in `source install.sh`
  - Double check requirements
- Failed to build `coqhammer`
  - `make clean` in `ASTactic/coqhammer` and remake `coqhammer`
- `make` in `coq_projects` fails
  - `make clean` and remake the entire folder
  - If that doesn't work, double check requirements and reset the directory to its fresh state
- `pip` failed to install `lmdb==0.94`
  - Try installing `lmdb==1.0`

- `EOF` error while loading `.pt` objects
  - Rebuild that project/file specifically


TODO: Add more?


## 6. Resources

<!-- Add resources on pre-built data and  -->

Data can be obtained from the original CoqGym repo [here](https://github.com/princeton-vl/CoqGym#14-downloading-the-pre-extracted-proofs-recommended)
