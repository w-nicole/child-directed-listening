import os
from os.path import join, exists
import sys
import json

class Config:
    def __init__(self):

        '''
        Reads the following parameters from the JSON
        
        SEED : the random seed to use
        
        n_utterances_per_age : number of utterances to score with grammaticality
        
        n_beta : number of success samples used to evaluate scores of Levenshtein distance scaling parameter

        n_across_time: Note this is the base pool sample, not necessarily the sample size used.
        
        number_of_posterior_words : The number of words to save from highest posterior words in the scoring process
        
        val_ratio : ??? "Proportion of CHILDES to use for Validation" .2", ???
        
        prior_folders: The names of the folders (circumvents needing to go through `src/utils/paths`) to use for visualizing the time plot

        regenerate : specifies if CHILDES data is regenerated in Providence - Retrieve data.ipynb

        eval_phase: {'val', 'eval'} -- ??? what to compute the scores on. Switch to eval at the end?

        exp_determiner: Name of the model run in which to place all results in experiments/

        training_version_name : Name of the model run, in case you want to use different trained models for scoring. This allows an experiment folder to have scores but no trained models. Doesn't appear to be in use
    
        verbose: {0,1}, useful for debugging or data generation.

        age_split : age in months to distinguish old vs. young children (30)

        beta_low : lowest value of beta to test (2.5)
        
        beta_high : highest value of beta to test (4.5)
    
        beta_num_values : number of values to test between the low and the high value of beta(20)

        fail_on_beta_edge : should the code fail if the best value is on the edge of the range of betas tested? (1)
        
        task_phases : identifiers for phase that the model is in
        
        training_datasets : valid training datasets to specify
        
        test_datasets: valid test datasets to specify
        
        spec_dict_params : used for path constructions
        
        '''

        # read in the environment variables        

        # get the path of the current file
        config_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root =  os.path.split(os.path.split(config_dir)[0])[0]
        
        self.json_path = os.path.join(self.project_root, 'config.json')
        
        # read in the JSON path and set everything
        f = open(self.json_path,)
        data = json.load(f)

        
        #set keys and vals from the JSON
        for key, value in data.items():
            setattr(self, key, value)
        
        self.set_defaults()
        self.check()
        
        # these all need to be defined with respect to the root
        self.make_folders([self.finetune_dir, self.prov_dir, self.prov_csv_dir])
        self.make_folders([self.sample_dir, self.eval_dir, self.fitting_dir])
        self.make_folders(['output/csv', 'output/pkl'])


    def set_defaults(self):
        # compute defaults
        

        self.finetune_dir_name = 'finetune'
        self.finetune_dir = join(self.project_root, 'output', self.finetune_dir_name)

        # Beta and across time evaluations
        self.prov_dir = join(self.project_root, 'output', 'prov') # Location of data for evaluations (in yyy)

        self.prov_csv_dir = join(self.project_root, 'output', 'prov_csv')
        self.cmu_path = join(self.project_root, 'output', 'pkl/cmu_in_childes.pkl') # The location of the cmu dictionary
        self.phon_map_path = join(self.project_root, 'data/phon_map_populated.csv')
        
        self.exp_dir = join(join(self.project_root, 'output/experiments'), self.exp_determiner)

        self.sample_dir = join(self.exp_dir, 'sample')
        self.fitting_dir = join(self.exp_dir, 'fit')
        self.eval_dir = join(self.exp_dir, 'eval')
        self.eval_priors_dir = join(self.eval_dir, f'n={self.n_across_time}')


    def check(self):
        assert self.n_beta == self.n_across_time, "The codebase generally assumes this for convenience."

    def make_folders(self, paths):        
        for p in paths:
            p = os.path.join( self.project_root, p)
            if not exists(p):
                os.makedirs(p)