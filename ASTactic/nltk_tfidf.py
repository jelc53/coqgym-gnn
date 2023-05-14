from cgi import test
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import argparse
import pickle
import json
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


def prepare_corpus(df, projects_flag):
    results = load_and_process_model_results()
    df = df.merge(results, on=['project', 'lib', 'proof'], how='left')
    df = df[df['proof_tactic_str'] != '99']

    if projects_flag:
        cols = ['project', 'success_astactic', 'success_gsmax']
        proj_df = df[cols].groupby(by=['project']).sum().reset_index()
        corpus = [
            ' '.join((df[df['project'] == proj]['proof_tactic_str']).tolist()) 
            for proj in proj_df['project']
        ]
        metadata = {
            'project_name': proj_df['project'].tolist(),
            'astactic_result': proj_df['success_astactic'].tolist(),
            'gsmax_result': proj_df['success_gsmax'].tolist(),
        }
    else:
        corpus = df['proof_tactic_str'].tolist()
        metadata = {
            'project_name': df['project'].tolist(),
            'lib_name': df['lib'].tolist(),
            'proof_name': df['proof'].tolist(),
            'astactic_result': df['success_astactic'].tolist(),
            'gsmax_result': df['success_gsmax'].tolist(),
        }
    return corpus, metadata


def load_and_process_model_results():
    res_astactic = pd.read_csv('evaluation/test_astactic/results.csv')
    res_gsmax = pd.read_csv('evaluation/test_gs_max_int_emb/results.csv')
    res_astactic.rename(columns={
        "success": "success_astactic",
        "num_tactics": "num_tactics_astactic",
        "time": "time_astactic"
        }, inplace=True
    )
    res_gsmax.rename(columns={
        "success": "success_gsmax",
        "num_tactics": "num_tactics_gsmax",
        "time": "time_gsmax"
        },  inplace=True
    )
    res_astactic['success_astactic'] = res_astactic['success_astactic'].astype(int)
    res_gsmax['success_gsmax'] = res_gsmax['success_gsmax'].astype(int)
    return res_astactic.merge(res_gsmax, on=['project', 'lib', 'proof'], how='left')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="corpus.pkl")
    parser.add_argument("--projs_file", type=str, default="../projs_split.json")
    parser.add_argument("--project_flag", type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(args.in_file, 'rb') as f:
        df = pickle.load(f)

    with open(args.projs_file, 'rb') as f:
        projs = json.load(f)
        test_projs = projs['projs_test']

    corpus, metadata = prepare_corpus(df, args.project_flag)
    tfidf_df = prepare_tfidf(corpus, metadata, test_projs)
    counts_df = prepare_counts(corpus, metadata, test_projs)

    filename_tfdif = 'tfidf_projects.csv' if args.project_flag else 'tfidf_proofs.csv'
    filename_counts = 'counts_projects.csv' if args.project_flag else 'counts_proofs.csv'

    tfidf_df.to_csv(filename_tfdif, index=False)
    counts_df.to_csv(filename_counts, index=False)
