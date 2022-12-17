
import os
import pandas as pd
import sys
# Adapted 12/12/22: https://github.com/smeylan/child-directed-listening/blob/master/src/run/run_beta_search.py
sys.path.append('.')
sys.path.append('src/.')
# end cite
from src.utils import configuration, substitutions
config = configuration.Config()

if __name__ == "__main__":
    
    all_nonhuman_updated_scores_path = os.path.join(config.eval_priors_dir, 'subsampled_levdist_generated_glosses.pkl')
    all_nonhuman_updated_scores = pd.read_pickle(all_nonhuman_updated_scores_path)
    
    counts = substitutions.get_substitution_counter(all_nonhuman_updated_scores)
    original_tokens = [original for original, _ in unique_tuples]
    substituted_tokens = [substitute for _, substitute in unique_tuples]
    counts_df = pd.DataFrame.from_records({
        'original' : original_tokens,
        'substitution' : substituted_tokens,
        'count' : list(map(lambda current_tuple : counts[current_tuple], unique_tuples))
    }).sort_values(by='count', ascending = False)
    
    counts_path = os.path.join(config.eval_priors_dir, 'token_substitution_counts.csv')
    counts_df.to_csv(counts_path)
    print(f'Written to: {counts_path}')
    