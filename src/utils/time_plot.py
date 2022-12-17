
import os
import pandas as pd
import sys
import numpy as np
from collections import defaultdict

# Adapted from 12/13/22: https://github.com/smeylan/child-directed-listening/blob/d782ef5b24998b215a1724b48fcf57e8b7b73898/src/run/run_models_across_time.py
sys.path.append('.')
sys.path.append('src/.')
from utils import configuration
config = configuration.Config()
# end cite

def merge_time_plot_df_per_prior(viewable_grammatical_df, all_tokens_phono):

    all_priors_present = set(viewable_grammatical_df['prior'])
    assert len(all_priors_present) == 1, "Need to have only one prior in the data in this function."
    prior = list(all_priors_present)[0]
    
    if not set(viewable_grammatical_df.is_grammatical) == {0, 1, -1}:
        import pdb; pdb.set_trace()

    # If any of the ALO was grammatical, mark entire thing as grammatical
    bert_token_id_to_grammatical = defaultdict(int)
    for entry_index in range(viewable_grammatical_df.shape[0]):
        if viewable_grammatical_df.is_grammatical == 1:
            bert_token_id_to_grammatical[entry_index.bert_token_id] = 1
    
    all_bert_token_ids = sorted(bert_token_id.keys())
    bert_token_id_to_grammatical_df = pd.DataFrame.from_records({
        'bert_token_id' : all_bert_token_ids,
        'is_grammatical' : list(map(lambda token_id : bert_token_id_to_grammatical[token_id]))
    }).sort_values(by='bert_token_id')
    bert_token_id_to_grammatical_df['prior'] = prior
    
    bert_ids = set(viewable_grammatical_df.bert_token_id)
    assert len(set(all_tokens_phono.bert_token_id)) == all_tokens_phono.shape[0]
    all_tokens_phono_subset = all_tokens_phono[all_tokens_phono.bert_token_id.isin(bert_ids)].sort_values(by='bert_token_id')
    
    if not list(bert_token_id_to_grammatical_df.bert_token_id) == list(all_tokens_phono_subset.bert_token_id):
        import pdb; pdb.set_trace()
    
    bert_token_id_to_grammatical_df['year'] = list(all_tokens_phono_subset['year'])
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
    
    
    
    
    
    