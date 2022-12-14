
import glob
import os
import pandas as pd
import numpy as np
import json
import torch

import sys
# Adapted 12/12/22: https://github.com/smeylan/child-directed-listening/blob/master/src/run/run_beta_search.py
sys.path.append('.')
sys.path.append('src/.')
# end cite
from src.utils import configuration, generation_processing, load_splits, paths
config = configuration.Config()

# 12/13/22: https://github.com/smeylan/child-directed-listening/blob/master/src/utils/split_gen.py
SEED = config.SEED
np.random.seed(SEED)
# end cite

def sample_bert_token_ids():
    
    folders = generation_processing.get_prior_folders()
    dfs_by_folder = {}
    for folder in folders:
        dfs_by_folder[folder] = { age_str : df for age_str, df in generation_processing.get_dfs_by_age(folder).items() }
    
    first_key = sorted(list(dfs_by_folder.keys()))[0]
    reference_dict = dfs_by_folder[first_key]

    for key in dfs_by_folder:
        if 'human' in key:
            print('Skipping human folder as expected'); continue
        compare_df_dict = dfs_by_folder[key]
        if not list(compare_df_dict.keys()) == list(reference_dict.keys()):
            import pdb; pdb.set_trace()
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
        
    subsample_path = paths.get_subsample_path()
    
    full_prior_folder = os.path.join(config.eval_priors_dir, 'human')
    if not os.path.exists(full_prior_folder): os.makedirs(full_prior_folder)
    
    # Write human in prep for plotting
    
    all_ids = set()
    for current_set in list(map(lambda samples : set(samples), subsamples.values())):
        all_ids |= current_set
    all_tokens_phono = load_splits.load_phono()
    stopword_set = generation_processing.get_stopword_set()
    all_phono_in_subset = all_tokens_phono[all_tokens_phono.bert_token_id.isin(all_ids)]
    stopword_in_samples_set = set(all_phono_in_subset.token)
    if not stopword_in_samples_set.issubset(stopword_set):
        import pdb; pdb.set_trace()
    assert set(all_phono_in_subset.speaker_code) == {'CHI'}
    
    all_shuffled_phono_in_subset_path = os.path.join(full_prior_folder, 'viewable_levdist_generated_glosses.csv')
    all_shuffled_phono = generation_processing.shuffle_dataframe(all_phono_in_subset[['bert_token_id', 'gloss']])
    all_shuffled_phono.to_csv(all_shuffled_phono_in_subset_path)
    
    print(f'Wrote viewable human data to {all_shuffled_phono_in_subset_path}')
    
    torch.save(subsamples, subsample_path)
    print(f'Wrote subsamples to {subsample_path}')
    
    return subsamples

if __name__ == '__main__':
    
    sample_bert_token_ids()
    
    