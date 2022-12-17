
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
from src.utils import configuration, generation_processing, load_splits, paths, load_models, generation_processing
config = configuration.Config()

# 12/13/22: https://github.com/smeylan/child-directed-listening/blob/master/src/utils/split_gen.py
SEED = config.SEED
np.random.seed(SEED)
# end cite

def sample_bert_token_ids():
    
    folders = generation_processing.get_prior_folders()
    dfs_by_folder = {}
    for folder in folders:
        if paths.is_prior_name_human(folder.split('/')[-1]):
            print('Skipping human folder, as expected.')
            continue
        dfs_by_folder[folder] = { age_str : df for age_str, df in generation_processing.get_dfs_by_age(folder).items() }
        
    first_key = sorted(list(dfs_by_folder.keys()))[0]
    reference_dict = dfs_by_folder[first_key]

    for key in dfs_by_folder:
        compare_df_dict = dfs_by_folder[key]
        if not list(compare_df_dict.keys()) == list(reference_dict.keys()):
            import pdb; pdb.set_trace()
        for age_str in reference_dict:
            if not np.all(list(reference_dict[age_str].bert_token_id) == list(compare_df_dict[age_str].bert_token_id)):
                import pdb; pdb.set_trace()
                
    total_selected = 0
    subsamples = {}
    for age_str, raw_df in reference_dict.items():
        stopword_df = generation_processing.filter_for_stopwords(raw_df)
        pool = list(stopword_df.bert_token_id)
        current_n = min(len(pool), config.n_utterances_per_age)
        total_selected += current_n
        subsamples[age_str] = list(np.random.choice(pool, size = current_n, replace = False))
        print(f'For age: {age_str}, subsample size: {current_n} / {len(pool)}') 
        
    # Check no duplicates
    ages = sorted(subsamples.keys())
    pool_total = set()
    for age1 in ages:
        for age2 in ages:
            if age1 == age2 : continue
            intersection = set(subsamples[age1]) & set(subsamples[age2])
            if len(intersection) > 0:
                import pdb; pdb.set_trace()
            if len(set(subsamples[age1])) != len(subsamples[age1]):
                import pdb; pdb.set_trace()
            pool_total |= set(subsamples[age1])
        
    subsample_path = paths.get_subsample_path()
    human_folder = paths.get_human_folder()
    
    # Write human in prep for plotting
    
    all_ids = set()
    for current_set in list(map(lambda samples : set(samples), subsamples.values())):
        all_ids |= current_set
    if not total_selected == len(all_ids):
        import pdb; pdb.set_trace()
    all_tokens_phono = load_splits.load_phono()
    stopword_set = generation_processing.get_stopword_set()
    
    all_phono_in_subset = all_tokens_phono[all_tokens_phono.bert_token_id.isin(all_ids)].copy()
    stopword_in_samples_set = set(all_phono_in_subset.token)
    if not stopword_in_samples_set.issubset(stopword_set):
        import pdb; pdb.set_trace()
    assert set(all_phono_in_subset.speaker_code) == {'CHI'}
    assert set(all_phono_in_subset.phase) == {config.eval_phase}
    
    all_shuffled_phono_in_subset_path = os.path.join(human_folder, 'viewable_levdist_generated_glosses.csv')
    tokenizer = load_models.get_primary_tokenizer()
    
    new_glosses = generation_processing.process_glosses_with_tokenizer(all_phono_in_subset, tokenizer)
    all_phono_in_subset['new_gloss'] = new_glosses
    all_shuffled_phono = generation_processing.shuffle_dataframe(all_phono_in_subset[['bert_token_id', 'gloss', 'new_gloss']])
    
    all_shuffled_phono.to_csv(all_shuffled_phono_in_subset_path)
    
    print(f'Wrote viewable human data to {all_shuffled_phono_in_subset_path}')
    
    torch.save(subsamples, subsample_path)
    print(f'Wrote subsamples to {subsample_path}')
    
    return subsamples

if __name__ == '__main__':
    
    sample_bert_token_ids()
    
    