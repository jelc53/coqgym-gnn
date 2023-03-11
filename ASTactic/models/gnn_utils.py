from gallina import traverse_postorder
import torch

from non_terminals import nonterminals


def create_edge_index(ast):
    "Creates an edge index for the given ast."
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

def create_x(ast):
    x = []

    def callbck(node):
        x.append([nonterminals.index(node.data)])

    traverse_postorder(ast, callbck)

    return torch.tensor(x, dtype=torch.float)
