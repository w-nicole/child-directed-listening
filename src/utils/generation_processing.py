
import glob
import os
import pandas as pd

from src.utils import configuration, paths
config = configuration.Config()


def get_dfs_by_age(prior_folder):
    raw_across_time_df_paths = glob.glob(os.path.join(prior_folder, 'levdist_run_models_across_time_*.pkl'))
    dfs = {}
    for path in raw_across_time_df_paths:
        age_str = paths.extract_age_str(path)
        dfs[age_str] = pd.read_pickle(path)
    return dfs


def get_prior_folders():
    folders = glob.glob(os.path.join(config.eval_dir, f"n={config.n_across_time}", "*"))
    assert all(map(lambda path : os.path.isdir(path), folders))
    return folders
    
def get_stopword_set():
    with open('./nltk_stopwords.txt') as f:
        lines = list(map(lambda s : s.strip(), f.readlines()[1:])) # Omit the citation
        stopword_set = set(lines)
    return stopword_set
    
def filter_for_stopwords(df):
    stopword_set = get_stopword_set()
    return df[df['token'].isin(stopword_set)].copy()