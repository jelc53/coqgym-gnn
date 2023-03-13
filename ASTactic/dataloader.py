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
        data = pickle.load(open(self.proof_steps[idx], "rb"))
        # TODO: Postprocess data so that x is one-hot?
        return data

    def get(self, idx):
        return self.__getitem__(idx)


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
        for proof_step in batch:
            for key, value in proof_step._store.items():
                if key not in fields:
                    continue
                data_batch[key].append(value)
        for k, v in data_batch.items():
            batch[k] = v
        return batch

    ds = ProofStepsData(split, opts)
    print(ds[0])
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
            pickle.dump(data_batch, open("data_batch.pickle", "wb"))
        bar.update(i)
