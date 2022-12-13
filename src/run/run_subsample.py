
import glob
import os
import pandas as pd
import numpy as np
import json
import torch

import sys
# 12/12/22: https://github.com/smeylan/child-directed-listening/blob/master/src/run/run_beta_search.py
sys.path.append('.')
sys.path.append('src/.')
# end cite
from src.utils import configuration, generation_processing, load_splits
config = configuration.Config()

def sample_bert_token_ids():
    
    folders = generation_processing.get_prior_folders()
    dfs_by_folder = {}
    for folder in folders:
        dfs_by_folder[folder] = { age_str : df for age_str, df in generation_processing.get_dfs_by_age(folder).items() }
    
    first_key = sorted(list(dfs_by_folder.keys()))[0]
    reference_dict = dfs_by_folder[first_key]

    for key in dfs_by_folder:
        compare_df_dict = dfs_by_folder[key]
        assert list(compare_df_dict.keys()) == list(reference_dict.keys())
        for age_str in reference_dict:
            if not np.all(list(reference_dict[age_str].bert_token_id) == list(compare_df_dict[age_str].bert_token_id)):
                import pdb; pdb.set_trace()
                
    subsamples = {}
    for age_str, raw_df in reference_dict.items():
        stopword_df = generation_processing.filter_for_stopwords(raw_df)
        pool = list(stopword_df.bert_token_id)
        current_n = min(len(pool), config.n_sentences_per_age)
        subsamples[age_str] = list(np.random.choice(pool, size = current_n))
        print(f'For age: {age_str}, subsample size: {current_n} / {len(pool)}') 
        
    subsample_path = os.path.join(config.eval_dir, 'subsampled_bert_token_ids.pt')
    
    all_ids = set()
    for current_set in list(map(lambda samples : set(samples), subsamples.values())):
        all_ids |= current_set
    all_tokens_phono = load_splits.load_phono()
    stopword_set = generation_processing.get_stopword_set()
    stopword_in_samples_set = set(all_tokens_phono[all_tokens_phono.bert_token_id.isin(all_ids)].token)
    if not stopword_in_samples_set.issubset(stopword_set):
        import pdb; pdb.set_trace()
    
    torch.save(subsamples, subsample_path)
    print(f'Wrote subsamples to {subsample_path}')
    
    return subsamples

if __name__ == '__main__':
    
    sample_bert_token_ids()
    
    