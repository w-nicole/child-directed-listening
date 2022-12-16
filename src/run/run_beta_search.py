import os
from os.path import join, exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import sys

sys.path.append('.')
sys.path.append('src/.')
from src.utils import load_splits, load_models, split_gen, parsers, hyperparameter_utils, sample_across_models, configuration, paths
config = configuration.Config()

def optimize_beta(fitting_dict):
    '''
        Find the values of beta and lambda which minimize posterior surprisal; save this information in a place that run_models_across_time can load

        Args:
        fitting dict: a dictionary with keys for training_split, training_dataset, test_split, test_dataset, etc. See utils/paths.py for a full description
        
        Return: the best parameter values for WFST and Levenshtein distance likelihoods; saves the scores for each hyperparameter value as a side effect

    '''

    beta_sample = hyperparameter_utils.get_hyperparameter_search_values('beta') 
        
    # initial_vocab determines the softmax mask used by BERT, leave it as mask for all evaluations/training
    
    initial_vocab, cmu_in_initial_vocab, cmu_indices_for_initial_vocab  = load_models.get_initial_vocab_info()
    fitting_path = paths.get_directory(fitting_dict)    
    
    if not exists(fitting_path):
        os.makedirs(fitting_path)
    
    success_utts_sample_path = paths.get_sample_csv_path(task_phase_to_sample_for='fit', val_eval_phase='val', split=fitting_dict['test_split'], dataset=fitting_dict['test_dataset'], data_type='success', age = None, n=config.n_beta)
    success_utts_sample  = pd.read_csv(success_utts_sample_path).utterance_id
        
    # Don't use failures for beta search
    hyperparam_search_results = sample_across_models.sample_across_models(success_utts_sample, [], fitting_dict, beta_sample)
    
    this_raw_beta_results = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'levdist']
 
    # Log the beta results
    this_beta_results_surp = hyperparam_search_results.loc[hyperparam_search_results.likelihood_type == 'levdist'].groupby(['beta_value']).posterior_probability.agg(lambda x: np.mean(-1 * np.log(x))).reset_index()
    this_beta_results_surp = this_beta_results_surp.rename(columns = {'posterior_probability' : 'posterior_surprisal'})
    beta_results_path = join(fitting_path, f'beta_search_results_{config.n_beta}.csv')
    this_beta_results_surp.to_csv(beta_results_path)
    print("Writing beta results to", {beta_results_path})

    return this_raw_beta_results, this_beta_results_surp
    
if __name__ == '__main__':    
    
    start_time = str(datetime.today())
    parser = parsers.split_parser()
        
    raw_args = parser.parse_known_args()[0]    
    this_model_args = vars(raw_args)

    this_model_args['task_phase'] = 'fit'
    this_model_args['n_samples'] = config.n_across_time   
    print(this_model_args)             
    
    this_model_dict = load_models.get_fitted_model_dict(this_model_args)

    print('Loaded the model!')    
    this_raw_beta_results, this_beta_results_surp = optimize_beta(this_model_dict)

    print(f'Computations complete for model:')
    print(this_model_dict)
    print(f'Started computations at: {start_time}')
    print(f'Finished computations at: {str(datetime.today())}')
