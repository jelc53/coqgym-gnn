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

from extract_proof_steps import parse_goal, filter_env


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
                        d[j.name] = dict(pool.map(stats, proofs))
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
            steps.append({"local_context": lc, "goal": g})
    return proof["name"], {"steps": steps, "env": env}


def analyze(dir, n_cpu=mp.cpu_count()):
    with mp.Pool(n_cpu) as pool:
        results = pool.map(analyze_project, Path(dir).glob("*.pkl"))
    return pd.DataFrame(it.chain(*results))


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


def summarize(steps):
    heights = []
    ns = []
    for step in steps:
        for d in step["local_context"].values():
            heights.append(d["height"])
            ns.append(d["n"])
        heights.append(step["goal"]["height"])
        ns.append(step["goal"]["n"])
    qs = [0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    h_qs = np.quantile(heights, qs, interpolation="higher")
    n_qs = np.quantile(ns, qs, interpolation="higher")
    h_d = {f"height_p{int(q*100)}": v for q, v in zip(qs, h_qs)}
    n_d = {f"nodes_p{int(q*100)}": v for q, v in zip(qs, n_qs)}
    return {"n_steps": len(steps), **h_d, **n_d}


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
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    if args.command == "collect":
        collect(args.projects, args.n_cpu)
    elif args.command == "analyze":
        df = analyze(args.dir, args.n_cpu)
        df.to_csv("stats.csv", index=False)
