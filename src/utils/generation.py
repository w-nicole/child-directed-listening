
import os
import numpy as np
import pandas as pd

from src.utils import configuration, generation_processing
config = configuration.Config()

# 12/13/22: https://github.com/smeylan/child-directed-listening/blob/master/src/utils/split_gen.py
SEED = config.SEED
np.random.seed(SEED)
# end cite

def process_single_prior_glosses(prior_folder_name, subsamples, stopwords_set, all_tokens_phono):
    full_prior_folder = os.path.join(config.eval_dir, f'n={config.n_across_time}', prior_folder_name)
    dfs = generation_processing.get_dfs_by_age(full_prior_folder)    
    
    all_updated_dfs = []
    for age_str in sorted(dfs.keys()):
        age_based_id_set = subsamples[age_str]
        print(f'\tUpdating glosses: {age_str}')
        raw_df_uncut = dfs[age_str]
        raw_df_cut = raw_df_uncut[raw_df_uncut.bert_token_id.isin(age_based_id_set)].copy()
        updated_df = replace_gloss_single_entry(raw_df_cut, all_tokens_phono)
        if not all(updated_df.token.isin(stopwords_set)):
            import pdb; pdb.set_trace()
        all_updated_dfs.append(updated_df)
        
    return pd.concat(all_updated_dfs)

def replace_gloss_single_entry(current_scores, all_tokens_phono):
    
    assert not any(list(map(lambda s : ' ' in s, all_tokens_phono.token)))
    assert list(all_tokens_phono.bert_token_id) == list(range(all_tokens_phono.shape[0]))
    
    all_generation_dfs = []
    
    for entry_index in range(current_scores.shape[0]):
        
        token_entry = current_scores.iloc[entry_index]
        
        # Identify the relevant sections of the dataframe
        bert_token_id = token_entry.bert_token_id
        token_in_general = all_tokens_phono[all_tokens_phono.bert_token_id == bert_token_id].copy()
        assert token_in_general.shape[0] == 1

        utterance_ids = list(set(token_in_general.utterance_id))
        if not len(utterance_ids) == 1:
            import pdb; pdb.set_trace()
        utterance_id = utterance_ids[0]
        
        # Perform replacements for every relevant highest posterior word
        
        highest_posterior_words = generation_processing.get_tied_highest_posterior_words(token_entry)
        
        current_gloss_df = all_tokens_phono[all_tokens_phono.utterance_id == utterance_id]
        replace_indices = np.where(np.array(current_gloss_df.bert_token_id) == bert_token_id)[0]
        assert replace_indices.shape[0] == 1, replace_indices
        replace_index = replace_indices.item()

        for inserted_word in highest_posterior_words:
            # Generate a new substituted entry for every word of interest
            generation_entry = token_entry.copy()
            new_gloss_list = []
            for gloss_index in range(current_gloss_df.shape[0]):
                new_gloss_list.append(current_gloss_df['token'].iloc[gloss_index] if gloss_index != replace_index else inserted_word)
            new_gloss = ' '.join(new_gloss_list)
            generation_entry['new_gloss'] = new_gloss
            all_generation_dfs.append(dict(generation_entry))
            
    return pd.DataFrame.from_records(all_generation_dfs)
