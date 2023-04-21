#!/usr/bin/env python3
from extract_proof_steps import parse_goal, filter_env
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys


def main(n_cpu=mp.cpu_count()):
    with open("../projs_split.json") as f:
        splits = json.load(f)
    projects = [project for split in splits.values() for project in split]
    with mp.Pool(n_cpu) as pool:
        for project in tqdm(projects[:3], desc='projects', position=0):
            d = {}
            dir = Path("../data") / project
            for j in tqdm(list(dir.glob("**/*.json")), desc=project, position=1, leave=False):
                with open(j) as f:
                    proofs = json.load(f)["proofs"]
                    d[j.name] = dict(pool.map(stats, proofs))
            with open(dir.with_suffix('.pkl'), 'wb') as f:
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


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-n", "--n_cpu", type=int, default=1)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    main(args.n_cpu)
