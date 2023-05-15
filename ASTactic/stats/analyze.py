#!/usr/bin/env python3
from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import pandas as pd
import sys


def tf_idf(args):
    df = pd.read_csv(args.csv).dropna()
    non_features = args.targets + ["proof_name", "lib_name", "project_name"]
    features = list(set(df.columns) - set(non_features))
    X = df[features].values
    m = LogisticRegression()
    dfs = []
    for target in args.targets:
        y = df[target].values.astype(int)
        m.fit(X, y)
        dfs += [
            pd.DataFrame(
                [
                    {"feature": f, f"{target}_coef": c}
                    for f, c in zip(features, np.squeeze(m.coef_))
                ]
            )
        ]
    dfm = dfs[0]
    for dft in dfs[1:]:
        dfm = dfm.merge(dft, on="feature")
    dfm.to_csv("tf_idf_coefs.csv", index=False)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analysis = parser.add_subparsers(
        title="analysis",
        dest="analysis",
        description="Analysis to conduct.",
    )
    tf_idf = analysis.add_parser(
        "tf_idf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tf_idf.add_argument(
        "-csv",
        default="tf_idf.csv",
        help="CSV with TF-IDF tactic data.",
    )
    tf_idf.add_argument(
        "-t",
        "--targets",
        nargs="+",
        default="graph_sage",
        help="Target column to predict.",
    )
    tf_idf.add_argument(
        "-n",
        "--top_n",
        type=int,
        default=25,
        help="Top 'n' features to visualize.",
    )
    return parser.parse_args(argv[1:])


if __name__ == "__main__":
    args = parse_args(sys.argv)
    match args.analysis:
        case "tf_idf":
            tf_idf(args)
        case _:
            print("Invalid analysis.")
