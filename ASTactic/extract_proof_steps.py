from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch, Data
from lark.exceptions import ParseError, UnexpectedCharacters
import torch
from hashlib import md5
import pdb
import gc
import argparse
import json
import os
import sys
from glob import glob


sys.setrecursionlimit(100000)
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)
from utils import SexpCache, iter_proofs
from agent import filter_env
from gallina import GallinaTermParser, traverse_postorder
from tac_grammar import CFG, NonterminalNode, TerminalNode, TreeBuilder
from models.gnn_utils import create_edge_index, create_x

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
total_count = 0
path_dict = {}

def to_pyg_data(term):
    return Data(x=torch.tensor(term["x"]), edge_index=torch.tensor(term["edge_index"]))

def process_proof(filename, proof_data):
    if "entry_cmds" in proof_data:
        is_synthetic = True
    else:
        is_synthetic = False
    global num_discarded, total_count, path_dict

    if args.filter and args.filter != filename.split(os.path.sep)[2]:
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
        # local context & goal
        if step["goal_ids"]["fg"] == []:
            num_discarded += 1
            continue
        # tactic
        tac_str = step["command"][0][:-1]
        try:
            actions = tactic2actions(tac_str)
        except (UnexpectedCharacters, ParseError) as ex:
            num_discarded += 1
            continue

        path = os.path.join(
            args.output, split, f"{proj}-{proof_data['name']}-{i:08d}.pt"
        )
        if path_dict.get(path, 0) > 0:
            # Path exists already and is not unique
            # print(path, path_dict[path])
            pass
        count = path_dict.get(path, 0) + 1
        path_dict[path] = count
        if os.path.exists(path):
            total_count += 1
        total_count += 1

        assert step["command"][1] == "VernacExtend"
        assert step["command"][0].endswith(".")
        # environment
        env = filter_env(proof_data["env"])
        goal_id = step["goal_ids"]["fg"][0]
        local_context, goal = parse_goal(proof_data["goals"][str(goal_id)])
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
            G = to_pyg_data(env)
            Gs.append(G)
            # remove so we can add the regular dict to the Data object
            del env["x"]
            del env["edge_index"]
        for lc in proof_step["local_context"]:
            G = to_pyg_data(lc)
            Gs.append(G)
            # remove so we can add the regular dict to the Data object
            del lc["x"]
            del lc["edge_index"]
        Gs.append(to_pyg_data(proof_step["goal"]))
        del proof_step["goal"]["x"]
        del proof_step["goal"]["edge_index"]
        B = Batch.from_data_list(Gs)
        for k, v in proof_step.items():
            B[k] = v
        torch.save(B, path)
        # proof_steps[split].append(B)
        del B, env, proof_step, Gs
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

    filter_file = \
        lambda f: f.split(os.path.sep)[2] in \
            (projs_split['projs_valid'] # + projs_split['projs_test']
            if not args.filter else [args.filter])

    iter_proofs(
        args.data_root, process_proof, include_synthetic=False, show_progress=True, filter_file=filter_file
    )
    print(f'{total_count} total, {num_discarded} discarded, {len(path_dict)} unique')
