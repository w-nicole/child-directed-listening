
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Adapted 12/13/22: https://github.com/smeylan/child-directed-listening/blob/d782ef5b24998b215a1724b48fcf57e8b7b73898/src/run/run_models_across_time.py
sys.path.append('.')
sys.path.append('src/.')
from utils import configuration, load_splits, time_plot
config = configuration.Config()
# end cite

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    all_tokens_phono = load_splits.load_phono()
    all_ages = list(pd.unique(all_tokens_phono.year.dropna()))
    
    plt.title("Grammaticality over time")
    
    for prior_name, prior_folder in config.prior_folders.items():
        with_grammar_df = time_plot.merge_viewable_and_score_df(prior_folder, all_tokens_phono)
        sorted_ages, percentages = time_plot.calculate_percentage_for_ages(with_grammar_df, all_ages)
        plt.plot(sorted_ages, percentages, label = prior_name, alpha = 0.5)
    
    plt.xlabel('Age (in years)')
    plt.ylabel('Percent generations grammatical')
    plt.legend()
    
    figure_path = os.path.join(config.eval_priors_dir, 'grammaticality_over_time.png')
    plt.savefig(figure_path)
        
    