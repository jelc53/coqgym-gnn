#!/usr/bin/env python3
import argparse
import asyncio
import itertools as it
import json
import multiprocessing as mp
import subprocess
import sys


def mp_extract_proofs(n_cpu):
    cmd = "python extract_proof_steps.py --filter {task}"
    rage(cmd, ["train", "valid", "test"], n_cpu)


def mp_evaluate(n_cpu):
    cmd = "python evaluate.py ours+hammer ours+hammer-TEST --path runs/astactic/checkpoints/model_003.pth --filter {task}"
    rage(cmd, ["test"], n_cpu)


def rage(cmd, splits, n_cpu: int = mp.cpu_count()):
    with open("../projs_split.json") as f:
        d = json.load(f)
    tasks = it.chain(*[d[split] for split in splits])
    cmds = [cmd.format(task=task) for task in tasks]
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
    parser.add_argument("-ep", "--extract_proofs", action='store_true')
    parser.add_argument("-t", "--evaluate_model", action='store_true')
    parser.add_argument("-n_cpu", default=mp.cpu_count())
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    if args.extract_proofs:
        mp_extract_proofs(args.n_cpu)
    elif args.evaluate_model:
        mp_evaluate(args.n_cpu)