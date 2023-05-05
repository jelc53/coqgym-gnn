import os
import json

from tqdm import tqdm

from multiprocess_utils import mp_iter_libs


def extract_proof_tree(
    project,
    file_name,
    proofs,
    output,
    _process_list,
):
    """
    Extracts proof trees from a file

    Proof tree json structure (grouped by library):
    {
        "proof_name": {
            "goal" : goal_str,
            "tactic": tactic_str,
            "children": [
                {
                    "goal": goal_str,
                    "tactic": tactic_str,
                    "children": []
                },
                ...
            ]
        }
    }
    """
    proof_trees = {}
    for proof_data in tqdm(
        proofs,
        desc=file_name[file_name.find(project) :],
        leave=False,
        position=_process_list.index(os.getpid()) + 1,
    ):
        # Recurse through proof tree
        proof_trees[proof_data["name"]] = recurse(proof_data["proof_tree"], proof_data)

    # Save proof trees
    save_file = os.path.join(output, file_name[file_name.find(project) :])
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    json.dump(proof_trees, open(save_file, "w"), indent=2)


def recurse(proof_tree, proof_data):
    proof_tree_data = {}
    goal_id = proof_tree["goal_id"]
    proof_tree_data["goal"] = proof_data["goals"][str(goal_id)]["type"]
    # Find right tactic in step
    for step in proof_data["steps"]:
        if step["command"][1] != "VernacExtend":
            continue
        if not step["goal_ids"]["fg"]:
            continue
        if step["goal_ids"]["fg"][0] == goal_id:
            proof_data["tactic"] = step["command"][0]
            break
    proof_tree_data["tactic"] = proof_data["tactic"]
    proof_tree_data["children"] = []
    if proof_tree["children"]:
        for child in proof_tree["children"]:
            proof_tree_data["children"].append(recurse(child, proof_data))
    return proof_tree_data


def main(split):
    mp_iter_libs(
        extract_proof_tree,
        ["./proof_trees"],
        n_cpu=8,
        split=split,
    )


if __name__ == "__main__":
    main("train")
    main("test")
    main("valid")
