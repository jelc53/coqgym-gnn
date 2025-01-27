from fileinput import filename
import torch
import numpy as np
import random
from glob import glob
import argparse
import json
import os
import sys

sys.setrecursionlimit(100000)
sys.path.append(os.path.normpath(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)
from hashlib import md5
from utils import log
from progressbar import ProgressBar
from agent import Agent
from models.prover import Prover
import pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("eval_id", type=str)
    parser.add_argument("--datapath", type=str, default="../data")
    parser.add_argument("--projs_split", type=str, default="../projs_split.json")
    parser.add_argument(
        "--split", choices=["train", "valid", "test"], type=str, default="test"
    )
    parser.add_argument("--file", type=str)
    parser.add_argument("--proof", type=str)
    parser.add_argument("--filter", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--output_dir", type=str, default="evaluation")
    parser.add_argument("--max_num_tactics", type=int, default=300)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--hammer_timeout", type=int, default=100)
    parser.add_argument("--depth_limit", type=int, default=50)
    parser.add_argument(
        "--beam_width", type=int, default=20
    )  # lots of timeout when >200
    parser.add_argument("--num_tactic_candidates", type=int, default=20)
    parser.add_argument(
        "--lens_norm", type=float, default=0.5, help="lengths normalization"
    )
    parser.add_argument("--tac_grammar", type=str, default="tactics.ebnf")
    parser.add_argument("--term_embedding_dim", type=int, default=256)
    parser.add_argument("--size_limit", type=int, default=50)
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
        help="dimension of the grammar embeddings",
    )
    parser.add_argument(
        "--symbol_dim",
        type=int,
        default=256,
        help="dimension of the terminal/nonterminal symbol embeddings",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="dimension of the LSTM controller"
    )
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()
    log(opts)
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opts.device.type == "cpu":
        log("using CPU", "WARNING")

    torch.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    if "ours" in opts.method:
        model = Prover(opts)
        log("loading model checkpoint from %s.." % opts.path)
        if opts.device.type == "cpu":
            checkpoint = torch.load(opts.path, map_location="cpu")
        else:
            checkpoint = torch.load(opts.path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(opts.device)
    else:
        model = None

    agent = Agent(model, None, None, opts)

    if opts.file:
        files = [opts.file]
    else:
        files = []
        projs = json.load(open(opts.projs_split))["projs_" + opts.split]
        for proj in projs:
            files.extend(
                glob(os.path.join(opts.datapath, "%s/**/*.json" % proj), recursive=True)
            )

    if opts.filter:
        files = [
            f
            for f in files
            # if md5(f.encode("utf-8")).hexdigest().startswith(opts.filter)
            if opts.filter in f
        ]
    # if opts.filter != filename.split(os.path.sep)[2]:
    #     return

    print(files)
    results = []
    errors = []
    bar = ProgressBar(max_value=len(files))
    for i, f in enumerate(files):
        print("file: ", f)
        # print('cuda memory allocated before file: ', torch.cuda.memory_allocated(opts.device), file=sys.stderr)
        r, e = agent.evaluate(f, opts.proof)
        results.extend(r)
        errors.extend(e)
        bar.update(i)

    oup_dir = os.path.join(opts.output_dir, opts.eval_id)
    err_dir = os.path.join(oup_dir, "errors")
    if not os.path.exists(oup_dir):
        os.makedirs(oup_dir)
    if not os.path.exists(err_dir):
        os.mkdir(err_dir)
    if opts.filter is None and opts.file is None and not skip_projects and not skip_libs:
        oup_file = os.path.join(oup_dir, "results.json")
        err_file = os.path.join(err_dir, "errors.json")
    else: # Same file names
        if opts.file is None:
            file_name = "%s" % opts.filter
        elif opts.proof is None:
            file_name = "%s" \
                % os.path.sep.join(opts.file.split(os.path.sep)[2:]).replace(
                    os.path.sep, "-"
                )[:-5]
        else:
            file_name = "%s-%s" \
                % (
                    os.path.sep.join(opts.file.split(os.path.sep)[2:]).replace(
                        os.path.sep, "-"
                    )[:-5],
                    opts.proof,
                )
        if opts.skip:
            file_name += f"-{opts.skip.split(os.path.sep)[-1].split('.')[0]}"
        file_name += ".json"
        oup_file = os.path.join(oup_dir, file_name)
        err_file = os.path.join(err_dir, file_name)
    opts = vars(opts)
    del opts["device"]
    json.dump({"options": opts, "results": results}, open(oup_file, "wt"))
    log("\nResults saved to " + oup_file)
    if errors:
        json.dump(errors, open(err_file, "wt"))
        log("Errors saved to " + err_file)
