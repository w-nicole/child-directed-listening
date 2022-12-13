
import argparse
import os
import torch
import pandas as pd

import sys
# 12/13/22: https://github.com/smeylan/child-directed-listening/blob/d782ef5b24998b215a1724b48fcf57e8b7b73898/src/run/run_models_across_time.py
sys.path.append('.')
sys.path.append('src/.')
# end cite
from src.utils import configuration, generation, generation_processing, load_splits, paths
config = configuration.Config()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--prior_folder', type=str, help='The folder section naming the model parameters.')
    args = parser.parse_args()
    
    subsample_path = paths.get_subsample_path()
    subsamples = torch.load(subsample_path)
    
    full_prior_folder = os.path.join(config.eval_dir, f'n={config.n_across_time}', args.prior_folder)
    dfs = generation_processing.get_dfs_by_age(full_prior_folder)
    all_tokens_phono = load_splits.load_phono()
    stopwords_set = generation_processing.get_stopword_set()
    
    all_updated_dfs = []
    all_viewable_dfs = []
    for age_str in sorted(dfs.keys()):
        age_based_id_set = subsamples[age_str]
        print(f'Updating glosses: {age_str}')
        raw_df_uncut = dfs[age_str]
        raw_df_cut = raw_df_uncut[raw_df_uncut.bert_token_id.isin(age_based_id_set)].copy()
        updated_df = generation.replace_gloss_single_entry(raw_df_cut, all_tokens_phono)
        viewable_df = updated_df[['bert_token_id', 'new_gloss']]
        if not all(updated_df.token.isin(stopwords_set)):
            import pdb; pdb.set_trace()
        all_updated_dfs.append(updated_df)
        all_viewable_dfs.append(viewable_df)
        
    all_updated_scores_path = os.path.join(full_prior_folder, 'subsampled_levdist_generated_glosses.pkl')
    all_updated_scores = pd.concat(all_updated_dfs)
    all_updated_scores.to_pickle(all_updated_scores_path)
    
    all_viewable_scores_path = os.path.join(full_prior_folder, 'viewable_levdist_generated_glosses.csv')
    all_viewable_scores = generation_processing.shuffle_dataframe(pd.concat(all_viewable_dfs))
    all_viewable_scores.to_csv(all_viewable_scores_path)
        
    print(f'Written all scores to: {all_updated_scores_path}')
    print(f'Written viewable glosses to: {all_viewable_scores_path}')
    
    
    