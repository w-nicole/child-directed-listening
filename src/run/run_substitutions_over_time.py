
import os
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
# Adapted 12/12/22: https://github.com/smeylan/child-directed-listening/blob/master/src/run/run_beta_search.py
sys.path.append('.')
sys.path.append('src/.')
# end cite
from src.utils import configuration, substitutions
config = configuration.Config()

if __name__ == '__main__':
    
    all_nonhuman_updated_scores_path = os.path.join(config.eval_priors_dir, 'subsampled_levdist_generated_glosses.pkl')
    all_nonhuman_updated_scores = pd.read_pickle(all_nonhuman_updated_scores_path)
    
    counts_path = os.path.join(config.eval_priors_dir, 'token_substitution_counts.csv')
    counts_df = pd.read_csv(counts_path)
    
    if not substitutions.check_counts_descending(counts_df):
        import pdb; pdb.set_trace()
    
    substitutions_df = counts_df[counts_df.original != counts_df.substitution].copy()

    cut_for_plot = lambda current_list : current_list[:config.number_of_substitutions_to_plot]
    popular_substitutions = [ (original, substitution)
        for original, substitution in zip(
            cut_for_plot(substitutions_df['original'].tolist()),
            cut_for_plot(substitutions_df['substitution'].tolist())
        )
    ]
    
    all_ages = sorted(list(pd.unique(all_nonhuman_updated_scores.age.dropna())))
    
    plt.title(f"Counts of top substitutions over time")
    
    for substitution_pair in popular_substitutions:
        original, substitute = substitution_pair
        counts_for_ages = []
        for age in all_ages:
            scores_for_age = all_nonhuman_updated_scores[np.isclose(all_nonhuman_updated_scores.age, age)]
            counter = substitutions.get_substitution_counter(scores_for_age)
            count = counter[substitution_pair]
            counts_for_ages.append(count)
        plt.plot(all_ages, counts_for_ages, label = f'{original} -> {substitute}')
        
    plt.xlabel('Age (in years)')
    plt.ylabel('Count')
    plt.legend()
    
    figure_path = os.path.join(config.eval_priors_dir, f'top_{config.number_of_substitutions_to_plot}_substitutions.png')
    plt.savefig(figure_path)
    