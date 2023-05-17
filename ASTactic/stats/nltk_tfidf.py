#!usr/bin/env python3
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import argparse
import pickle
import json
import os
# import tqdm

# # Let's consider two simple documents
# tactic1 = "red in |- *; auto with zarith."
# tactic2 = "red in |- *; auto with induction."


def _insert_columns(df, data_dict):
    for key, value in data_dict.items():
        df.insert(loc=0, column=key, value=value)
    return df


def prepare_counts(corpus, metadata, test_projs):
    my_stop_words = text.ENGLISH_STOP_WORDS.union([";", ".", " "])
    vectorizer = CountVectorizer(stop_words=my_stop_words)
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    out_df = pd.DataFrame(denselist, columns=feature_names)
    out_df = _insert_columns(out_df, metadata)
    test_proj_mask = [proj in test_projs for proj in out_df['project_name']]
    return out_df[test_proj_mask]


def prepare_tfidf(corpus, metadata, test_projs):
    my_stop_words = text.ENGLISH_STOP_WORDS.union([";", ".", " "])
    vectorizer = TfidfVectorizer(stop_words=my_stop_words)
    vectors = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    out_df = pd.DataFrame(denselist, columns=feature_names)
    out_df = _insert_columns(out_df, metadata)
    test_proj_mask = [proj in test_projs for proj in out_df['project_name']]
    return out_df[test_proj_mask]


def prepare_corpus(df, m1_name, m2_name, m1_dir, m2_dir, level='proof'):
    results = load_and_process_model_results(m1_name, m2_name, m1_dir, m2_dir)
    df = df.merge(results, on=['project', 'lib', 'proof'], how='left')
    df = df[df['proof_tactic_str'] != '99']
    df = df.dropna()

    if level == 'project':
        cols = ['project', 'success_'+m1_name, 'success_'+m2_name]
        proj_df = df[cols].groupby(by=['project']).sum().reset_index()
        corpus = [
            ' '.join((df[df['project'] == proj]['proof_tactic_str']).tolist())
            for proj in proj_df['project']
        ]
        metadata = {
            'project_name': proj_df['project'].tolist(),
            'y_'+m1_name: proj_df['success_'+m1_name].tolist(),
            'y_'+m2_name: proj_df['success_'+m2_name].tolist(),
        }
    elif level == 'proof_step':
        corpus = df['proof_tactic_str'].tolist()
        metadata = {
            'project_name': df['project'].tolist(),
            'lib_name': df['lib'].tolist(),
            'proof_name': df['proof'].tolist(),
            'proof_step': df['step'].tolist(),
            'y_'+m1_name: df['success_'+m1_name].tolist(),
            'y_'+m2_name: df['success_'+m2_name].tolist(),
        }
    else:
        corpus = df['proof_tactic_str'].tolist()
        metadata = {
            'project_name': df['project'].tolist(),
            'lib_name': df['lib'].tolist(),
            'proof_name': df['proof'].tolist(),
            'y_'+m1_name: df['success_'+m1_name].tolist(),
            'y_'+m2_name: df['success_'+m2_name].tolist(),
        }
    return corpus, metadata


def load_and_process_model_results(m1_name, m2_name, m1_dir, m2_dir):
    res_model1 = pd.read_csv(
        os.path.join(
            os.path.pardir, 'evaluation', m1_dir, 'results.csv'
        )
    )
    res_model2 = pd.read_csv(
        os.path.join(
            os.path.pardir, 'evaluation', m2_dir, 'results.csv'
        )
    )
    res_model1.rename(columns={
        "success": "success_" + m1_name,
        "num_tactics": "num_tactics_" + m1_name,
        "time": "time_" + m1_name,
        }, inplace=True
    )
    res_model2.rename(columns={
        "success": "success_" + m2_name,
        "num_tactics": "num_tactics_" + m2_name,
        "time": "time_" + m2_name,
        },  inplace=True
    )
    res_model1['success_'+m1_name] = res_model1['success_'+m1_name].astype(int)
    res_model2['success_'+m2_name] = res_model2['success_'+m2_name].astype(int)
    return res_model1.merge(res_model2, on=['project', 'lib', 'proof'], how='left')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="corpus.pkl")
    parser.add_argument("--projs_file", type=str, default="../../projs_split.json")
    # parser.add_argument("--project", action='store_true')
    parser.add_argument("--level", type=str, default='proof')

    parser.add_argument("--model1_name", type=str, default="astactic")
    parser.add_argument("--model2_name", type=str, default="gsmax")

    parser.add_argument("--model1_dir", type=str, default="test_astactic")
    parser.add_argument("--model2_dir", type=str, default="test_gs_max_int_emb")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.in_file, 'rb') as f:
        df = pickle.load(f)

    with open(args.projs_file, 'rb') as f:
        projs = json.load(f)
        test_projs = projs['projs_test']

    corpus, metadata = prepare_corpus(
        df, args.model1_name, args.model2_name,
        args.model1_dir, args.model2_dir, args.level
    )
    tfidf_df = prepare_tfidf(corpus, metadata, test_projs)
    counts_df = prepare_counts(corpus, metadata, test_projs)

    if args.project:
        filename_tfidf = 'tfidf_projects.csv'
        filename_counts = 'counts_projects.csv'
    elif args.step:
        filename_tfidf = 'tfidf_proof_steps.csv'
        filename_counts = 'counts_proof_steps.csv'
    else:
        filename_tfidf = 'tfidf_proofs.csv'
        filename_counts = 'counts_proofs.csv'

    tfidf_df.to_csv(filename_tfidf, index=False)
    counts_df.to_csv(filename_counts, index=False)
