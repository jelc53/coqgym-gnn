#!/usr/bin/env python3
import argparse
import asyncio
import json
import multiprocessing as mp
import subprocess
import sys
import os

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)

from utils import update_env


def rage(n_cpu: int = mp.cpu_count()):
    with open("../projs_split.json") as f:
        d = json.load(f)
    n = len(d["projs_valid"])
    tasks = d["projs_valid"]
    cmds = [
        f"python extract_proof_steps.py --filter {task} --output ./proof_steps_gnn"
        for task in tasks
    ]
    with mp.Pool(n_cpu) as p:
        p.map(x_output, cmds)


def x(cmd, check=True, echo=True):
    "execute command streaming stdin, stdout, and stderr"
    if echo:
        p(cmd)
    loop = asyncio.new_event_loop()
    return_code = loop.run_until_complete(_x(cmd))
    if check and return_code:
        q(f'"{cmd}" failed!')


async def _x(cmd):
    pipe = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    await asyncio.wait(
        [
            asyncio.create_task(_stream(pipe.stdout)),
            asyncio.create_task(_stream(pipe.stderr)),
        ]
    )
    return await pipe.wait()


async def _stream(stream):
    "stream output"
    while True:
        line = await stream.readline()
        if line:
            p("\t" + line.decode("utf-8").strip())
        else:
            break


def x_output(cmd, echo=True):
    "execute command and collect stdout"
    if echo:
        p(cmd)
    try:
        out = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as err:
        q(str(err))
        return ""
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
    parser.add_argument("--n_cpu", default=mp.cpu_count(), type=int)
    parser.add_argument(
        "-s",
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "train_valid"],
    )
    parser.add_argument(
        "-f",
        "--filter",
        type=str,
        default="",
        help="Filter proof extraction by project name. Multiple projects can be separated by comma.",
    )
    parser.add_argument("-o", "--output", type=str, default="./proof_steps_gnn")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--proj_splits_file", type=str, default="../projs_split.json")
    parser.add_argument("-l", "--log", type=str, default="")
    args = parser.parse_args(argv[1:])
    if args.split not in ["train", "valid", "train_valid"]:
        q(f"Invalid split {args.split}")
    if args.split == "train_valid":
        args.splits = ["train", "valid"]
    else:
        args.splits = [args.split]

    if not os.path.exists(args.output):
        os.makedirs(args.output)
        for split in args.splits:
            os.makedirs(os.path.join(args.output, split))
    if args.verbose:
        p(args)
    args.mute = not args.verbose
    if args.filter:
        args.filter = args.filter.split(",")
    else:
        args.filter = []
    return args


def process_lib(project, file_name, proofs, output, split):
    env = {"constants": [], "inductives": []}
    # Process proof data
    for proof_data in proofs:
        env = update_env(env, proof_data["env_delta"])
        del proof_data["env_delta"]
        proof_data["env"] = env
        process_proof(project, file_name, proof_data, output, split)


if __name__ == "__main__":
    args = parse_args(sys.argv)
    print(args)
    # rage(args.n_cpu)
    from multiprocess_utils import MPSelections, mp_iter_libs
    from extract_proof_steps import process_proof

    filters = MPSelections(args.filter, [], [])
    for split in args.splits:
        process_proof_args = [
            args.output,
            split,
        ]
        print(process_proof_args, filters)
        mp_iter_libs(
            process_lib,
            process_proof_args,
            n_cpu=args.n_cpu,
            filters=filters,
            data_path=args.data_path,
            proj_splits_file=args.proj_splits_file,
            mute=args.mute,
            split=split,
            log_file=args.log,
        )
