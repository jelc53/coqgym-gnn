#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import subprocess
import sys


<<<<<<< Updated upstream
def rage(name: str, model_path: str, extra: str = '', n_cpu: int = mp.cpu_count()):
=======
def rage(name: str, model_path: str, model_type: str, extra: str = '', n_cpu: int = mp.cpu_count()):
>>>>>>> Stashed changes
    projects = [
        'zorns-lemma',
        'coqoban',
        'coq-procrastination',
        'fermat4',
        'zfc',
        'hoare-tut',
        'coqrel',
        'UnifySL',
        'jordan-curve-theorem',
        'buchberger',
    ]
    cmds = [
<<<<<<< Updated upstream
        f"python evaluate.py ours {name} --path {model_path} --filter {project}" for project in projects]
=======
        f"python evaluate.py ours {name} --path {model_path} --model_type {model_type} {extra} --filter {project}" for project in projects]
>>>>>>> Stashed changes
    with mp.Pool(n_cpu) as p:
        p.map(x_output, cmds)


def x_output(cmd, echo=True):
    "execute command and collect stdout"
    if echo:
        p(cmd)
    try:
        out = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as err:
        q(str(err))
    return out.decode("utf-8").strip()


def p(msg, fmt="debug", file=sys.stdout):
    "print colorized by status"
    begin = {
        "begin": "\x1b[0;30;43m",
        "debug": "\x1b[0m",
        "error": "\x1b[0;30;41m",
        "success": "\x1b[0;30;42m",
    }[fmt]
    end = "\x1b[0m"
    print(begin + msg + end, file=file)


def q(msg):
    "print and quit"
    p(msg, "error", file=sys.stderr)
    sys.exit(1)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("name")
    parser.add_argument("model_path")
<<<<<<< Updated upstream
=======
    parser.add_argument("model_type")
    parser.add_argument("-e", "--extra", help="extra args",
                        default="--num_heads 2")
>>>>>>> Stashed changes
    parser.add_argument("-n_cpu", default=mp.cpu_count(), type=int)
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    print(args)
<<<<<<< Updated upstream
    rage(args.name, args.model_path, '', args.n_cpu)
=======
    rage(args.name, args.model_path, args.model_type, args.extra, args.n_cpu)
>>>>>>> Stashed changes
