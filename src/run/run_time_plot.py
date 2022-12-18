
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
    
    scored_path = os.path.join(config.eval_priors_dir, 'scored_levdist_generated_glosses.csv')
    full_viewable_path = os.path.join(config.eval_priors_dir, 'full_levdist_generated_glosses.csv')
    
    viewable_grammatical_df = pd.read_csv(scored_path).sort_values(by='viewable_index')
    full_viewable_df = pd.read_csv(full_viewable_path).sort_values(by='viewable_index')
    
    if not list(viewable_grammatical_df.viewable_index) == list(full_viewable_df.viewable_index):
        import pdb; pdb.set_trace()
       
    raw_merged_scored_df = full_viewable_df.copy()
    raw_merged_scored_df['is_grammatical'] = viewable_grammatical_df['is_grammatical']
    
    merged_scored_df = raw_merged_scored_df[raw_merged_scored_df.is_grammatical != -1].copy()
    
    # Number of distinct and unique predictive positions (bert_token_ids), pooled over priors
    number_omitted = len(set(raw_merged_scored_df[raw_merged_scored_df.is_grammatical == -1].bert_token_id))
    number_of_bert_id_tokens = len(set(raw_merged_scored_df.bert_token_id))
    
    all_tokens_phono = load_splits.load_phono()
    all_ages = list(pd.unique(all_tokens_phono.year.dropna()))
    
    plt.title(f"Grammaticality over time, omitting {number_omitted} / {number_of_bert_id_tokens} tokens")
    
    for prior_name, prior_title in config.prior_folders.items():
        prior_df = merged_scored_df[merged_scored_df['prior'] == prior_title].copy()
        with_grammar_df = time_plot.merge_time_plot_df_per_prior(prior_df, all_tokens_phono)
        sorted_ages, percentages = time_plot.calculate_percentage_for_ages(with_grammar_df, all_ages)
        plt.plot(sorted_ages, percentages, label = prior_name, alpha = 0.5)
    
    plt.xlabel('Age (in years)')
    plt.ylabel('Percent generations grammatical')
    plt.legend()
    
    figure_path = os.path.join(config.eval_priors_dir, 'grammaticality_over_time.png')
    plt.savefig(figure_path)
        
    