
import numpy as np
import pandas as pd

def replace_gloss_single_entry(current_scores, all_tokens_phono):
    
    assert not any(list(map(lambda s : ' ' in s, all_tokens_phono.token)))
    assert list(all_tokens_phono.bert_token_id) == list(range(all_tokens_phono.shape[0]))
    
    all_generation_dfs = []
    for entry_index in range(current_scores.shape[0]):
        
        token_entry = current_scores.iloc[entry_index]
        posterior_words = token_entry['highest_posterior_words'].split()
        highest_posterior_word = posterior_words[0]
        generation_entry = token_entry.copy()
        bert_token_id = token_entry.bert_token_id

        token_in_general = all_tokens_phono[all_tokens_phono.bert_token_id == bert_token_id].copy()
        assert token_in_general.shape[0] == 1
        bert_token_id = token_in_general.bert_token_id.item()

        utterance_ids = list(set(token_in_general.utterance_id))
        if not len(utterance_ids) == 1:
            import pdb; pdb.set_trace()
        utterance_id = utterance_ids[0]

        current_gloss_df = all_tokens_phono[all_tokens_phono.utterance_id == utterance_id]
        replace_indices = np.where(np.array(current_gloss_df.bert_token_id) == bert_token_id)[0]
        assert replace_indices.shape[0] == 1, replace_indices
        replace_index = replace_indices.item()

        new_gloss_list = []
        for gloss_index in range(current_gloss_df.shape[0]):
            new_gloss_list.append(current_gloss_df['token'].iloc[gloss_index] if gloss_index != replace_index else highest_posterior_word)
        new_gloss = ' '.join(new_gloss_list)
        generation_entry['new_gloss'] = new_gloss
        all_generation_dfs.append(dict(generation_entry))
    return pd.DataFrame.from_records(all_generation_dfs)
