import json
import os
import pickle
import sys

from tac_grammar import CFG, NonterminalNode, TerminalNode, TreeBuilder

sys.setrecursionlimit(100000)
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)
import argparse
import gc
import pdb
from hashlib import md5

import networkx as nx
import torch
from agent import filter_env
from lark.exceptions import ParseError, UnexpectedCharacters
from models.gnn_utils import create_edge_index, create_x
from models.non_terminals import nonterminals
from torch_geometric.data import Batch
from torch_geometric.utils import from_networkx

from gallina import GallinaTermParser, traverse_postorder
from utils import SexpCache, iter_proofs

term_parser = GallinaTermParser(caching=True)
sexp_cache = SexpCache("../sexp_cache", readonly=True)


def parse_goal(g):
    goal_tree = term_parser.parse(sexp_cache[g["sexp"]])
    goal = {
        "id": g["id"],
        "text": g["type"],
        "ast": goal_tree,
        "x": create_x(goal_tree),
        "edge_index": create_edge_index(goal_tree),
    }
    local_context = []
    for i, h in enumerate(g["hypotheses"]):
        for ident in h["idents"]:
            context_tree = term_parser.parse(sexp_cache[h["sexp"]])
            local_context.append(
                {
                    "ident": ident,
                    "text": h["type"],
                    "ast": context_tree,
                    "x": create_x(context_tree),
                    "edge_index": create_edge_index(context_tree),
                }
            )
    return local_context, goal


grammar = CFG("tactics.ebnf", "tactic_expr")
tree_builder = TreeBuilder(grammar)


def tactic2actions(tac_str):
    tree = tree_builder.transform(grammar.parser.parse(tac_str))
    assert tac_str.replace(" ", "") == tree.to_tokens().replace(" ", "")
    actions = []

    def gather_actions(node):
        if isinstance(node, NonterminalNode):
            actions.append(grammar.production_rules.index(node.action))
        else:
            assert isinstance(node, TerminalNode)
            actions.append(node.token)

    tree.traverse_pre(gather_actions)
    return actions


projs_split = json.load(open("../projs_split.json"))
proof_steps = {"train": [], "valid": [], "test": []}

num_discarded = 0


def to_nx_graph(term):
    G = nx.Graph()
    for i, x in enumerate(term["x"]):
        G.add_node(i, x=x)
    G.add_edges_from(term["edge_index"].T.numpy())
    return G


def process_proof(filename, proof_data):
    if "entry_cmds" in proof_data:
        is_synthetic = True
    else:
        is_synthetic = False
    global num_discarded

    if args.filter != filename.split(os.path.sep)[2]:
        return  # skip proof folders not included in filter flag
    # if not md5(filename.encode()).hexdigest().startswith(args.filter):
    #     return

    proj = filename.split(os.path.sep)[2]
    if proj in projs_split["projs_train"]:
        split = "train"
    elif proj in projs_split["projs_valid"]:
        split = "valid"
        if is_synthetic:
            return
    else:
        split = "test"
        if is_synthetic:
            return

    for i, step in enumerate(proof_data["steps"]):
        # consider only tactics
        if step["command"][1] in [
            "VernacEndProof",
            "VernacBullet",
            "VernacSubproof",
            "VernacEndSubproof",
        ]:
            continue
        assert step["command"][1] == "VernacExtend"
        assert step["command"][0].endswith(".")
        # environment
        env = filter_env(proof_data["env"])
        # local context & goal
        if step["goal_ids"]["fg"] == []:
            num_discarded += 1
            continue
        goal_id = step["goal_ids"]["fg"][0]
        local_context, goal = parse_goal(proof_data["goals"][str(goal_id)])
        # tactic
        tac_str = step["command"][0][:-1]
        try:
            actions = tactic2actions(tac_str)
        except (UnexpectedCharacters, ParseError) as ex:
            num_discarded += 1
            continue
        proof_step = {
            "file": filename,
            "proof_name": proof_data["name"],
            "n_step": i,
            "env": env,
            "local_context": local_context,
            "goal": goal,
            "tactic": {"text": tac_str, "actions": actions},
        }
        if is_synthetic:
            proof_step["is_synthetic"] = True
            proof_step["goal_id"] = proof_data["goal_id"]
            proof_step["length"] = proof_data["length"]
        else:
            proof_step["is_synthetic"] = False
        proof_step["tactic_actions"] = proof_step["tactic"]["actions"]
        proof_step["tactic_str"] = proof_step["tactic"]["text"]
        # del proof_step["tactic"]

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
        Gs = [from_networkx(G) for G in Gs]
        B = Batch.from_data_list(Gs)
        for k, v in proof_step.items():
            B[k] = v
        path = os.path.join(
            args.output, split, f"{args.filter}-{B['proof_name']}-{B['n_step']:08d}.pt"
        )
        torch.save(B, path)
        # proof_steps[split].append(B)
        del B, Gs, env, proof_step
        gc.collect()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Extract the proof steps from CoqGym for trainig ASTactic via supervised learning"
    )
    arg_parser.add_argument(
        "--data_root", type=str, default="../data", help="The folder for CoqGym"
    )
    arg_parser.add_argument(
        "--output", type=str, default="./proof_steps/", help="The output file"
    )
    arg_parser.add_argument("--filter", type=str, help="filter the proofs")
    args = arg_parser.parse_args()
    print(args)

    iter_proofs(
        args.data_root, process_proof, include_synthetic=False, show_progress=True
    )

    # for split in ["train", "valid"]:
    #     for i, step in enumerate(proof_steps[split]):
    #         dirname = os.path.join(args.output, split)
    #         if not os.path.exists(dirname):
    #             os.makedirs(dirname)
    #         if args.filter:
    #             pickle.dump(
    #                 step,
    #                 open(
    #                     os.path.join(dirname, "%s-%08d.pickle" % (args.filter, i)), "wb"
    #                 ),
    #             )
    #         else:
    #             pickle.dump(step, open(os.path.join(dirname, "%08d.pickle" % i), "wb"))
    #
    # print("\nOutput saved to ", args.output)
