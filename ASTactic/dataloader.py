import os
import random
import sys

import networkx as nx
import torch
from torch.utils.data import DataLoader
from options import parse_args
from progressbar import ProgressBar
from torch_geometric.data import Data, Dataset, Batch

sys.setrecursionlimit(100000)
import pdb
from collections import defaultdict
from glob import glob

import numpy as np


class ProofStepsData(Dataset):
    def __init__(self, split, opts):
        super().__init__()
        self.opts = opts

        if split in ["train", "valid"]:
            self.proof_steps = glob(os.path.join(opts.datapath, split, "*.pt"))
        elif split == "train_valid":
            self.proof_steps = glob(
                os.path.join(opts.datapath, "train/*.pt")
            ) + glob(os.path.join(opts.datapath, "valid/*.pt"))
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
        data = torch.load(self.proof_steps[idx])
        return data

    def get(self, idx):
        return self.__getitem__(idx)


def merge(batch_list):
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
    for proof_step in batch_list:
        for key, value in proof_step._store.items():
            if key not in fields:
                continue
            data_batch[key].append(value)
    batch = Batch.from_data_list(batch_list)
    for k, v in data_batch.items():
        batch[k] = v
    return batch

def create_dataloader(split, opts):

    ds = ProofStepsData(split, opts)
    return DataLoader( # Use original pytorch data loader since we want to define custom collate_fn
        ds,
        opts.batchsize,
        shuffle=split.startswith("train"),
        collate_fn=merge,
        num_workers=opts.num_workers,
    )


if __name__ == "__main__":
    import json

    opts = parse_args()
    loader = create_dataloader("train_valid", opts)
    bar = ProgressBar(max_value=len(loader))
    num_nodes = defaultdict(int)
    for i, data_batch in enumerate(loader):
        if i == 0:
            print(data_batch)
            # pickle.dump(data_batch, open("data_batch.pickle", "wb"))
        for proof_step in data_batch.to_data_list():
            _, counts = torch.unique(proof_step.batch, return_counts=True)
            for c in counts:
                num_nodes[int(c.item())] += 1
        bar.update(i+1)
    avg = sum([k * v for k, v in num_nodes.items()]) / sum(num_nodes.values())
    save_dict = {
        'num_nodes': dict(num_nodes),
        'avg': avg,
    }

    json.dump(save_dict, open("num_nodes_train_valid.json", "w"))
    print(avg)
