
import os
import pandas as pd
import sys
import numpy as np

# Adapted from 12/13/22: https://github.com/smeylan/child-directed-listening/blob/d782ef5b24998b215a1724b48fcf57e8b7b73898/src/run/run_models_across_time.py
sys.path.append('.')
sys.path.append('src/.')
from utils import configuration
config = configuration.Config()
# end cite

def merge_viewable_and_score_df(prior_folder, all_tokens_phono):
    
    full_prior_folder = os.path.join(config.eval_priors_dir, prior_folder)
    raw_is_grammatical_df = pd.read_csv(os.path.join(full_prior_folder, 'scored_levdist_generated_glosses.csv'))
    if prior_folder != 'human':
        try:
            if not set(raw_is_grammatical_df.is_grammatical) == {0, 1}:
                import pdb; pdb.set_trace()
        except: import pdb; pdb.set_trace()
    is_grammatical_df = raw_is_grammatical_df.sort_values(by='bert_token_id')
    bert_ids = set(is_grammatical_df.bert_token_id)
    
    all_tokens_phono_subset = all_tokens_phono[all_tokens_phono.bert_token_id.isin(bert_ids)].sort_values(by='bert_token_id')
    
    if not list(is_grammatical_df.bert_token_id) == list(all_tokens_phono_subset.bert_token_id):
        import pdb; pdb.set_trace()
    
    is_grammatical_df['year'] = list(all_tokens_phono_subset['year'])
    return is_grammatical_df


def calculate_percentage_for_ages(merged_df, all_ages):
    percentages = []
    sorted_ages = sorted(all_ages)
    for age in sorted_ages:
        age_df = merged_df[np.isclose(merged_df.year, age)]
        if age_df.shape[0] == 0:
            import pdb; pdb.set_trace()
        else:
            percentage = sum(age_df['is_grammatical']) / age_df.shape[0]
            print(f'For age: {age}, pool: {age_df.shape[0]}')
        percentages.append(percentage)
    return sorted_ages, percentages
    
    
    
    
    
    