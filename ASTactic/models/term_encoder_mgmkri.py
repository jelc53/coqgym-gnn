import gc
import math
import os
import pdb
from collections import defaultdict
from itertools import chain
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from lark.tree import Tree
from torch_geometric.nn import GCNConv

from gallina import traverse_postorder

from non_terminals import nonterminals


class TermEncoder(gnn.MessagePassing):
    def __init__(self, opts):
        super().__init__(aggr="max")
        self.opts = opts
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 126)

    def forward(self, asts):
        # ipdb.set_trace()
        if len(asts) == 0:
            return [
                torch.zeros(len(asts), self.opts.term_embedding_dim).to(
                    self.opts.device
                )
            ]
        embeddings = []
        for i, ast in enumerate(asts):
            edge_index = self.create_edge_index(ast).to(self.opts.device)
            if not len(edge_index):
                x = torch.zeros(self.opts.term_embedding_dim).to(self.opts.device)
            else:
                x = self.create_x(ast).to(self.opts.device)
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
                x = x.flatten()
                reshaper = nn.Linear(len(x), self.opts.term_embedding_dim).to(
                    self.opts.device
                )
                x = reshaper(x)

            embeddings.append(x)
            del edge_index, x
            gc.collect()
        return torch.stack(embeddings).to(self.opts.device)

    def message(self, x_i, x_j):
        # x_i has shape [E, F_in]
        # x_j has shape [E, F_in]
        edge_features = torch.cat([x_i, x_j - x_i], dim=1)  # shape [E, 2 * F_in]
        return self.mlp(edge_features)  # shape [E, F_out]

    def create_edge_index(self, ast):
        index_map = {}
        counter = [0]

        def index_callbck(node):
            index_map[node.meta] = counter[-1]
            counter.append(counter[-1] + 1)

        traverse_postorder(ast, index_callbck)

        edge_index = []

        def callbck(node):
            for child in node.children:
                parent_child = [index_map[node.meta], index_map[child.meta]]
                child_parent = [index_map[child.meta], index_map[node.meta]]
                edge_index.append(parent_child)
                edge_index.append(child_parent)

        traverse_postorder(ast, callbck)

        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def create_x(self, ast):
        x = []

        def callbck(node):
            x.append([nonterminals.index(node.data)])

        traverse_postorder(ast, callbck)

        return torch.tensor(x, dtype=torch.float)
