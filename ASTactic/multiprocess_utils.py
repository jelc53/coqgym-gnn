import multiprocessing as mp
import os
import sys
import json
from tqdm import tqdm
from pathlib import Path
from typing import Callable, Iterable, Optional, overload, Tuple
from collections import defaultdict
from itertools import repeat
from functools import partial
from copy import deepcopy

sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
)

from utils import update_env

__all__ = ["mp_iter_proofs", "mp_iter_libs", "MPSelections"]


class MPSelections(dict):
    def __init__(self, projects=[], libs=[], proofs=[]):
        super().__init__()
        self["projects"] = projects
        self["libs"] = libs
        self["proofs"] = proofs

    def __repr__(self):
        return f"MPSelections(projects={self['projects']}, libs={self['libs']}, proofs={self['proofs']})"


DEFAULT_FILTERS = MPSelections(
    [],  # Projects
    [],  # Libs
    [],  # Proofs
)

DEFAULT_SKIPS = MPSelections(
    [],  # Projects
    [],  # Libs
    [],  # Proofs
)

SPLIT_CONVERSION = {
    "train": "projs_train",
    "valid": "projs_valid",
    "test": "projs_test",
}


@overload
def mp_iter_proofs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
) -> dict:
    ...


@overload
def mp_iter_proofs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    proj_callback: Callable = ...,
    proj_callback_args: Iterable = ...,
    proj_callback_kwargs: dict = ...,
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
) -> Tuple[dict, dict]:
    ...


# TODO: Add callbacks on library and project level
def mp_iter_proofs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    proj_callback: Optional[Callable] = None,
    proj_callback_args: Optional[Iterable] = None,
    proj_callback_kwargs: Optional[dict] = None,
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
):
    """
    Applies `fn(project, library, proof_data, *args, **kwargs)` in parallel using multiprocessing. Can whitelist/blacklist projects, proof files (libs), and individual proofs.

    Filters and skips are a dictionary of the form:
        {
            "projects": list[str],
            "libs":     list[str],
            "proofs":   list[str],
        }

    Filtering is applied AND-wise (i.e. take all projects in filters["projects"] AND all libs in filters["libs"] AND all proofs in filters["proofs"])
    THEN, apply all skips AND-wise.

    Project callback is applied after obtaining results from all proofs in a project. It is of the form:

    Output is of the form:
    {
        project: {
            lib: {
                proof: result
                ...
            }
            ...
        }
        ...
    }
    """

    def _mute():
        sys.stdout = open(os.devnull, "w")

    def _log():
        sys.stdout = open(log_file, "w")

    out = defaultdict(lambda: defaultdict(dict))
    splits = json.load(open(proj_splits_file))
    if split is not None:
        if split not in SPLIT_CONVERSION:
            raise ValueError(
                f"Invalid split {split}. Must be one of {list(splits.keys())}"
            )
        projects = splits[SPLIT_CONVERSION[split]]
    projects = [project for split in splits.values() for project in split]
    # Filter projects if there are project filters but no lib or proof filters
    if filters["projects"] and (not filters["libs"] or not filters["proofs"]):
        projects = [p for p in projects if p in filters["projects"]]
    projects = [p for p in projects if p not in skips["projects"]]
    proj_callback_results = defaultdict()
    init = None
    if log_file:
        init = _log
    elif mute:
        init = _mute
    with mp.Pool(n_cpu, initializer=init) as pool:
        # Load all async tasks (assumes all proofs can be loaded into memory)
        for project in tqdm(projects, desc=f"{fn.__name__} Total", position=1):
            for j in tqdm(
                list(str(p) for p in (Path(data_path) / project).glob("**/*.json")),
                desc=project,
                position=0,
            ):
                lib = j.split(os.path.sep)[-1]
                if lib in skips["libs"]:
                    continue
                if filters["libs"] and lib not in filters["libs"]:
                    continue
                proofs = json.load(open(j))["proofs"]
                env = {"constants": [], "inductives": []}
                # Process proof data
                run_proofs = []
                for proof_data in proofs:
                    env = update_env(env, proof_data["env_delta"])
                    del proof_data["env_delta"]
                    proof_data["env"] = deepcopy(env)
                    # Filters
                    if (
                        filters["proofs"]
                        and proof_data["name"] not in filters["proofs"]
                    ):
                        continue
                    # Skips
                    if proof_data["name"] in skips["proofs"]:
                        continue
                    run_proofs.append(proof_data)

                fn_args = (
                    (project, lib, proof_data, *args) for proof_data in run_proofs
                )
                r = pool.starmap_async(
                    _fn_wrapper,
                    zip(
                        repeat(fn),
                        fn_args,
                        repeat(kwargs),
                    ),
                )
                for proof_data, result in zip(run_proofs, r.get()):
                    out[project][lib][proof_data["name"]] = result
                if proj_callback is not None:
                    if proj_callback_args is None:
                        proj_callback_args = []
                    if proj_callback_kwargs is None:
                        proj_callback_kwargs = {}
                    proj_callback_results[project] = proj_callback(
                        out[project], *proj_callback_args, **proj_callback_kwargs
                    )

    if proj_callback is not None:
        return dict(out), dict(proj_callback_results)
    return dict(out)


def _fn_wrapper(fn, args, kwargs):
    return fn(*args, **kwargs)


def mp_iter_libs(
    fn: Callable,
    args: Iterable = [],
    kwargs: dict = {},
    proj_callback: Optional[Callable] = None,
    proj_callback_args: Iterable = [],
    proj_callback_kwargs: dict = {},
    *,  # Force keyword-only arguments
    split: Optional[str] = None,
    filters: dict = DEFAULT_FILTERS,
    skips: dict = DEFAULT_SKIPS,
    mute: bool = False,
    n_cpu: int = mp.cpu_count(),
    proj_splits_file: str = "../projs_split.json",
    data_path: str = "../data",
    log_file: str = "",
):
    def _mute():
        sys.stdout = open(os.devnull, "w")

    def _log():
        sys.stdout = open(log_file, "w")

    results = defaultdict(dict)
    out = defaultdict(dict)
    splits = json.load(open(proj_splits_file))
    if split is not None:
        if split not in SPLIT_CONVERSION:
            raise ValueError(
                f"Invalid split {split}. Must be one of {list(splits.keys())}"
            )
        projects = splits[SPLIT_CONVERSION[split]]
    projects = [project for split in splits.values() for project in split]
    # Filter projects if there are project filters but no lib or proof filters
    if filters["projects"] and not filters["libs"]:
        projects = [p for p in projects if p in filters["projects"]]
    projects = [p for p in projects if p not in skips["projects"]]
    proj_callback_results = defaultdict()
    init = None
    if log_file:
        init = _log
    elif mute:
        init = _mute
    with mp.Pool(n_cpu, initializer=init) as pool:
        # Load all async tasks (assumes all proofs can be loaded into memory)
        to_eval = []
        for project in tqdm(projects, desc=f"{fn.__name__} Loading...", position=0):
            for j in list(
                str(p) for p in (Path(data_path) / project).glob("**/*.json")
            ):
                lib = j.split(os.path.sep)[-1]
                if lib in skips["libs"]:
                    continue
                if filters["libs"] and lib not in filters["libs"]:
                    continue
                to_eval.append((project, lib, j))
        fn_args = ((p, l, json.load(open(j))["proofs"], *args) for p, l, j in to_eval)
        map_results = pool.imap(
            partial(_fn_wrapper, fn, kwargs=kwargs),
            fn_args,
        )
        for (project, lib, _), result in tqdm(
            zip(to_eval, map_results), total=len(to_eval), desc="Results", position=0
        ):
            out[project][lib] = result
        # Process callbacks
        if proj_callback is not None:
            for project, lib_dict in tqdm(results.items()):
                for lib, result in lib_dict.items():
                    if proj_callback_args is None:
                        proj_callback_args = []
                    if proj_callback_kwargs is None:
                        proj_callback_kwargs = {}
                    proj_callback_results[project] = proj_callback(
                        out[project], *proj_callback_args, **proj_callback_kwargs
                    )

    if proj_callback is not None:
        return dict(out), dict(proj_callback_results)
    return dict(out)


if __name__ == "__main__":
    import time

    def print_name(proj, lib, proof, *args, **kwargs):
        print(proof["name"], args, kwargs)

    def sleep(_, __, p, **kwargs):
        time.sleep(0.1)
        return p["name"]

    filter = {
        "projects": [
            "twoSquare",
            "QuickChick",
            "regexp",
            "domain-theory",
            "tarski-geometry",
            "ieee754",
            "ctltctl",
        ],
        "libs": [],
        "proofs": [],
    }
    kwargs = {"a": 1, "b": 2}
    # mp_iter_proofs(print_name, filters=filter)
    result = mp_iter_proofs(print_name, kwargs=kwargs, filters=filter, n_cpu=12)
    print(result)
