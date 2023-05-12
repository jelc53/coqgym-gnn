import json
import os
from glob import glob
from collections import defaultdict
from argparse import ArgumentParser
import pandas as pd

TIME_LIMIT = 600


def process_arguments():
    parser = ArgumentParser()

    parser.add_argument("results_folder", type=str)
    parser.add_argument("-o", "--output", type=str, default="")

    args = parser.parse_args()
    if not args.output:
        return args.results_folder, os.path.join(args.results_folder, "results.csv")
    return args.results_folder, args.output


def main():
    results_folder, output_file = process_arguments()

    # Load results into scope

    all_parsed_results = []
    for f in glob(os.path.join(results_folder, "*.json")):
        results = json.load(open(f, "r"))["results"]
        for r in results:
            all_parsed_results.append(parse_result(r))

    df = pd.DataFrame(all_parsed_results)
    df.to_csv(output_file, index=False)


def parse_result(result):
    project_name = result["filename"].split(os.path.sep)[2]
    lib = result["filename"][
        result["filename"].find(project_name) + len(project_name) + 1 :
    ]
    return {
        "project": project_name,
        "lib": lib,
        "proof": result["proof_name"],
        "success": result["success"] and result["time"] < TIME_LIMIT,
        "num_tactics": result["num_tactics"],
        "time": result["time"],
    }


if __name__ == "__main__":
    main()
