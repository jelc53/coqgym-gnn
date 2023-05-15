#!/usr/bin/env python3
from pathlib import Path
from tqdm import tqdm
import argparse
import itertools as it
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import sys

from extract_proof_steps import parse_goal, filter_env, tactic2actions


def collect(projects, n_cpu):
    if not projects:
        with open("../projs_split.json") as f:
            splits = json.load(f)
        projects = [project for split in splits.values() for project in split]
    with mp.Pool(n_cpu) as pool:
        for project in tqdm(projects, desc="projects", position=0):
            d = {}
            dir = Path("../data") / project
            pkl = dir.with_suffix(".pkl")
            if not pkl.exists():
                for j in tqdm(
                    list(dir.glob("**/*.json")), desc=project, position=1, leave=False
                ):
                    with open(j) as f:
                        proofs = json.load(f)["proofs"]
                        d[str(j)[str(j).find(project) + len(project) + 1 :]] = dict(
                            pool.map(stats, proofs)
                        )
                with open(pkl, "wb") as f:
                    pickle.dump(d, f)


def stats(proof):
    env = {}
    if "env" in proof:
        for item in filter_env(proof["env"]):
            env[item["qualid"]] = {
                "height": item["ast"].height,
                "n": len(item["x"]),
            }
    steps = []
    for step in proof["steps"]:
        if step["command"][1] in [
            "VernacEndProof",
            "VernacBullet",
            "VernacSubproof",
            "VernacEndSubproof",
        ]:
            continue
        tac_str = step["command"][0][:-1]
        try:
            actions = tactic2actions(tac_str)
        except:
            continue
        if step["goal_ids"]["fg"]:
            g, lc = {}, {}
            goal_id = step["goal_ids"]["fg"][0]
            local_context, goal = parse_goal(proof["goals"][str(goal_id)])
            g = {"height": goal["ast"].height, "n": len(goal["x"])}
            for item in local_context:
                lc[item["ident"]] = {
                    "height": item["ast"].height,
                    "n": len(item["x"]),
                }
            steps.append(
                {
                    "local_context": lc,
                    "goal": g,
                    "actions": actions,
                    "tactic_str": tac_str,
                }
            )
    return proof["name"], {"steps": steps, "env": env}


def analyze_project(path):
    with open(path, "rb") as f:
        project = pickle.load(f)
    v = []
    for lib, proofs in project.items():
        for name, d in proofs.items():
            # TODO(danj): wtf is going on with the env
            summary = summarize(d["steps"])
            v += [{"project": path.stem, "lib": lib, "proof": name, **summary}]
    return v


def tfidf_corpus(path):
    with open(path, "rb") as f:
        project = pickle.load(f)
    v = []
    for lib, proofs in project.items():
        for name, d in proofs.items():
            if d["steps"] == [] or "tactic_str" not in d["steps"][0]:
                summary = {"proof_tactic_str": "99"}
            else:
                summary = {
                    "proof_tactic_str": ' '.join(
                        [step["tactic_str"] for step in d["steps"]]
                    )
                }
            v += [{"project": path.stem, "lib": lib, "proof": name, **summary}] 
    return v


TACTICS = {
    "intro",
    "apply",
    "auto",
    "rewrite",
    "simpl",
    "unfold",
    "destruct",
    "induction",
    "elim",
    "split",
    "assumption",
    "trivial",
    "reflexivity",
    "case",
    "clear",
    "subst",
    "generalize",
    "exists",
    "red",
    "omega",
    "discriminate",
    "inversion",
    "simple_induction",
    "constructor",
    "congruence",
    "left",
    "right",
    "ring",
    "symmetry",
    "f_equal",
    "tauto",
    "revert",
    "specialize",
    "idtac",
    "hnf",
    "inversion_clear",
    "exfalso",
    "cbv",
    "contradict",
    "lia",
    "field",
    "easy",
    "cbn",
    "exact",
    "intuition",
    "eauto",
}


def summarize(steps):
    heights = []
    ns = []
    used_tactics = dict.fromkeys(TACTICS, 0)
    for step in steps:
        for d in step["local_context"].values():
            heights.append(d["height"])
            ns.append(d["n"])
        heights.append(step["goal"]["height"])
        ns.append(step["goal"]["n"])
        for tac in step["tactic_str"].split():
            if tac in TACTICS:
                used_tactics[tac] += 1
    qs = [0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    if not heights:
        heights = [-1]
    if not ns:
        ns = [-1]
    h_qs = np.quantile(heights, qs, interpolation="higher")
    n_qs = np.quantile(ns, qs, interpolation="higher")
    h_d = {f"height_p{int(q*100)}": v for q, v in zip(qs, h_qs)}
    n_d = {f"nodes_p{int(q*100)}": v for q, v in zip(qs, n_qs)}
    return {"n_steps": len(steps), **h_d, **n_d, **used_tactics}


def analyze(dir, n_cpu=mp.cpu_count(), func=analyze_project):
    with mp.Pool(n_cpu) as pool:
        results = pool.map(func, Path(dir).glob("*.pkl"))
    return pd.DataFrame(it.chain(*results))


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-n", "--n_cpu", type=int, default=mp.cpu_count())
    cmd = parser.add_subparsers(help="command", dest="command", title="commands")
    collect = cmd.add_parser(
        "collect", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    collect.add_argument(
        "-p",
        "--projects",
        nargs="*",
        help="only collect stats for given projects",
    )
    analyze = cmd.add_parser(
        "analyze", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    analyze.add_argument(
        "-d", "--dir", help="directory with pickled stats", default="../data"
    )
    tf_idf = cmd.add_parser(
        "tf-idf", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    tf_idf.add_argument(
        "-d", "--dir", help="directory with pickled stats", default="../data"
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    if args.command == "collect":
        collect(args.projects, args.n_cpu)
    elif args.command == "analyze":
        df = analyze(args.dir, args.n_cpu, analyze_project)
        df.to_csv("stats.csv", index=False)
    elif args.command == "tf-idf":
        corpus = analyze(args.dir, args.n_cpu, tfidf_corpus)
        with open('corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f)
