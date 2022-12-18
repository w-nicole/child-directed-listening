
import argparse
import os
import torch
import pandas as pd
import numpy as np

import sys
# Adapted 12/13/22: https://github.com/smeylan/child-directed-listening/blob/d782ef5b24998b215a1724b48fcf57e8b7b73898/src/run/run_models_across_time.py
sys.path.append('.')
sys.path.append('src/.')
# end cite
from src.utils import configuration, generation, generation_processing, load_splits, paths
config = configuration.Config()

# 12/13/22: https://github.com/smeylan/child-directed-listening/blob/master/src/utils/split_gen.py
SEED = config.SEED
np.random.seed(SEED)
# end cite

if __name__ == "__main__":

    stopwords_set = generation_processing.get_stopword_set()
    all_tokens_phono = load_splits.load_phono()
    
    subsample_path = paths.get_subsample_path()
    subsamples = torch.load(subsample_path)
    
    all_updated_scores_list = []
    for prior_name in sorted(config.prior_folders.values()):
        if paths.is_prior_name_human(prior_name):
            print('Skipping human folder, as expected.')
            continue
        print(f'Processing glosses for prior: {prior_name}')
        updated_scores_for_prior = generation.process_single_prior_glosses(prior_name, subsamples, stopwords_set, all_tokens_phono)
        updated_scores_for_prior['prior'] = prior_name
        all_updated_scores_list.append(updated_scores_for_prior)
        
    all_updated_scores = pd.concat(all_updated_scores_list)    
    
    # Add in human data for scoring
    human_folder = paths.get_human_folder()
    human_df = pd.read_csv(os.path.join(human_folder, 'viewable_levdist_generated_glosses.csv'))
    human_df['prior'] = config.prior_folders['Human']
    
    get_for_index_tuples = lambda df : [(bert_token_id, new_gloss, prior) for bert_token_id, new_gloss, prior in zip(df.bert_token_id, df.new_gloss, df.prior)]
    # Add viewable index
    human_tuples = get_for_index_tuples(human_df)
    scored_tuples = get_for_index_tuples(all_updated_scores)
    all_tuples = human_tuples + scored_tuples
    if not len(set(all_tuples)) == len(all_tuples):
        import pdb; pdb.set_trace()
    
    # Add viewable_index
    raw_viewable_index = np.arange(len(all_tuples))
    np.random.shuffle(raw_viewable_index)
    viewable_indices = raw_viewable_index.tolist()
    tuple_to_index_dict = {
        current_tuple : viewable_index
        for current_tuple, viewable_index in zip(all_tuples, viewable_indices)
    }
    
    tuple_to_index = lambda current_tuple, tuple_dict : tuple_dict[current_tuple]
    tuple_list_to_indices = lambda tuple_list, tuple_dict : list(map(lambda present_tuple : tuple_to_index(present_tuple, tuple_dict), tuple_list))
    
    human_df['viewable_index'] = tuple_list_to_indices(human_tuples, tuple_to_index_dict)
    all_updated_scores['viewable_index'] = tuple_list_to_indices(scored_tuples, tuple_to_index_dict)
    
    get_full_viewable_df = lambda df : df[['viewable_index', 'bert_token_id', 'new_gloss', 'prior']]
    raw_viewable_df = pd.concat([get_full_viewable_df(human_df), get_full_viewable_df(all_updated_scores)])
    
    if not len(set(raw_viewable_df['viewable_index'])) == len(raw_viewable_df['viewable_index']):
        import pdb; pdb.set_trace()
        
    get_viewable_df = lambda df : df[['viewable_index', 'new_gloss']]
    viewable_df = get_viewable_df(generation_processing.shuffle_dataframe(raw_viewable_df.copy()))
    
    # Write everything
    
    all_updated_scores_path = os.path.join(config.eval_priors_dir, 'subsampled_levdist_generated_glosses.pkl')
    all_updated_scores.to_pickle(all_updated_scores_path)
    print(f'Written all scores to: {all_updated_scores_path}')
    
    viewable_to_prior_path = os.path.join(config.eval_priors_dir, 'full_levdist_generated_glosses.csv')
    raw_viewable_df.to_csv(viewable_to_prior_path)
    
    all_viewable_scores_path = os.path.join(config.eval_priors_dir, 'viewable_levdist_generated_glosses.csv')
    viewable_df.to_csv(all_viewable_scores_path)
    
    print(f'Written all scores to: {all_updated_scores_path}')
    print(f'Written viewable glosses to: {all_viewable_scores_path}')
 
    
    