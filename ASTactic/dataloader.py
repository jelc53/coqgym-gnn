import os
import random
import sys

import networkx as nx
import torch
# from torch.utils.data import Dataset, DataLoader
from options import parse_args
from progressbar import ProgressBar
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx

sys.setrecursionlimit(100000)
import json
import pdb
import pickle
from collections import defaultdict
from glob import glob

import numpy as np


class ProofStepsData(Dataset):
    def __init__(self, split, opts):
        super().__init__()
        self.opts = opts

        if split in ["train", "valid"]:
            self.proof_steps = glob(os.path.join(opts.datapath, split, "*.pickle"))
        elif split == "train_valid":
            self.proof_steps = glob(
                os.path.join(opts.datapath, "train/*.pickle")
            ) + glob(os.path.join(opts.datapath, "valid/*.pickle"))
        random.shuffle(self.proof_steps)
        print("%d proof steps in %s" % (len(self), split))

    def __len__(self):
        return len(self.proof_steps)

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        """
        step = {
            'file': STR,
            'proof_name': STR,
            'n_step': INT,
            'env': [{
                'qualid': STR,
                'ast': LARK.TREE.TREE,
                'x': TENSOR.TENSOR
                'edge_index': TENSOR.TENSOR
            }],
            'local_context': [{
                'ident': STR,
                'ast': LARK.TREE.TREE,
                'x': TENSOR.TENSOR
                'edge_index': TENSOR.TENSOR
            }],
            'goal': {
                "id": STR,
                "text": STR,
                "ast": LARK.TREE.TREE,
                "x": TENSOR.TENSOR,
                "edge_index": TENSOR.TENSOR,
            },
            'tactic_actions':  [INT|STR],
            'tactic_str': STR,
        }
        """
        proof_step = pickle.load(open(self.proof_steps[idx], "rb"))
        # proof_step["goal"] = proof_step["goal"]["ast"]
        proof_step["tactic_actions"] = proof_step["tactic"]["actions"]
        proof_step["tactic_str"] = proof_step["tactic"]["text"]
        # del proof_step["tactic"]

        # create xs
        xs = [ps["x"] for ps in proof_step["env"]]
        xs += [lc["x"] for lc in proof_step["local_context"]]
        xs += [proof_step["goal"]["x"]]
        # create graphs
        Gs = []
        for env in proof_step["env"]:
            G = to_nx_graph(env)
            Gs.append(G)
            # remove so we can add the regular dict to the Data object
            del env["x"]
            del env["edge_index"]
        for lc in proof_step["local_context"]:
            G = to_nx_graph(lc)
            Gs.append(G)
            # remove so we can add the regular dict to the Data object
            del lc["x"]
            del lc["edge_index"]
        Gs.append(to_nx_graph(proof_step["goal"]))
        del proof_step["goal"]["x"]
        del proof_step["goal"]["edge_index"]
        Gu = nx.disjoint_union_all(Gs)
        G = from_networkx(Gu)
        # Assign attributes of proof step to Data object
        for k, v in proof_step.items():
            G[k] = v
        return G

    def get(self, idx):
        return self.__getitem__(idx)


def to_nx_graph(term):
    G = nx.Graph()
    for i, idx in enumerate(term["x"]):
        G.add_node(i, nonterminals_idx=idx)
    G.add_edges_from(term["edge_index"].T.numpy())
    return G


def create_dataloader(split, opts):
    def merge(batch):
        fields = [
            "file",
            "proof_name",
            "n_step",
            "env",
            "local_context",
            "goal",
            "is_synthetic",
            "tactic_actions",
            "tactic_str",
        ]
        data_batch = {key: [] for key in fields}
        for G_step in batch:
            for key, value in G_step.items():
                if key not in fields:
                    continue
                data_batch[key].append(value)
        for k, v in data_batch.items():
            batch[k] = v
        return batch

    ds = ProofStepsData(split, opts)
    return DataLoader(
        ds,
        opts.batchsize,
        shuffle=split.startswith("train"),
        collate_fn=merge,
        num_workers=opts.num_workers,
    )


if __name__ == "__main__":
    opts = parse_args()
    loader = create_dataloader("train", opts)
    bar = ProgressBar(max_value=len(loader))
    for i, data_batch in enumerate(loader):
        if i == 0:
            print(data_batch)
        bar.update(i)
