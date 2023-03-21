# CoqGym-GNN

![Example proof](images/example_proof.jpg)

GNN extension and improvements on [CoqGym](https://github.com/princeton-vl/CoqGym) repository.


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

In order to learn using GNNs efficiently, `x` and `edge_index` information need to be extracted from the `lark.Tree` representations of ASTs. This computation is very costly, so it is delegated to the proof extraction stage. Here are the pipeline modifications that facilitate this change:

- Modified proof step data representation from `dict` to `torch_geometric.data.Batch` objects
- Modified merge operation to facilitate `Batch` merging
- Used `torch_geometric.data.Dataset` object over `torch.data.Dataset` object
- Changed saving protocol from `.pickle` to PyTorch-optimized `.pt` using `torch.save()`

More explicitly, a comparison of the proof steps is outlined below:

**CoqGym proof step**:
```json

```

**CoqGym-GNN proof step**:
```json
```

Along with this, optimizations were made to the data generation process which facilitates easier updates to the extracted dataset and lighter computational resource requirements. Specifically:

- Added `filter_file` option in `iter_proofs` and its derivatives to skip loading of proof if not being considered
- Added multi-processing script for both `extract_proof.py` and `evaluate.py` for more efficient use of computational resources.

### 2.2 Design Modifications





## 3. Setup and Installation


## 4. FAQ and Known Bugs



## 5. Resources

<!-- Add resources on pre-built data and  -->

## 6. Reproducing Our Work


## 7. Future Work