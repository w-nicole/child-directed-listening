
import glob
import os
import pandas as pd
import numpy as np

from src.utils import configuration, paths
config = configuration.Config()

# 12/13/22: https://github.com/smeylan/child-directed-listening/blob/master/src/utils/split_gen.py
SEED = config.SEED
np.random.seed(SEED)
# end cite
    
    
def process_glosses_with_tokenizer(all_shuffled_phono, tokenizer):
    
    process_gloss = lambda gloss : '[chi] ' + ' '.join(tokenizer.tokenize(gloss))
    old_glosses = list(all_shuffled_phono['gloss'])
    new_glosses = list(map(process_gloss, old_glosses))
    return new_glosses

def get_tied_highest_posterior_words(token_entry):
    
    posterior_words = token_entry['highest_posterior_words'].split()
    posterior_probabilities = list(map(lambda s : float(s), token_entry['highest_posterior_probabilities'].split()))
    max_posterior_probability = np.max(posterior_probabilities)
    if not np.isclose(max_posterior_probability, posterior_probabilities[0]):
        import pdb; pdb.set_trace()
    matches = np.where(np.isclose(posterior_probabilities, max_posterior_probability))[0]
    if not np.all(matches == np.arange(matches.shape[0])):
        import pdb; pdb.set_trace()
        
    highest_posterior_words = posterior_words[:matches.shape[0]]
    if len(highest_posterior_words) >= config.number_of_posterior_words:
        print(f"Increase the number of posterior words to save, {config.number_of_posterior_words}-way tie detected.")
        import pdb; pdb.set_trace()
    return highest_posterior_words
    
    
def shuffle_dataframe(df):
    # 12/13/22: shuffling dataframe
    # Adapted from https://www.geeksforgeeks.org/pandas-how-to-shuffle-a-dataframe-rows/
    return df.sample(frac=1, random_state = config.SEED)
    # end cite
    
    
def get_dfs_by_age(prior_folder):
    raw_across_time_df_paths = glob.glob(os.path.join(prior_folder, 'levdist_run_models_across_time_*.pkl'))
    dfs = {}
    for path in raw_across_time_df_paths:
        age_str = paths.extract_age_str(path)
        dfs[age_str] = pd.read_pickle(path)
    return dfs


def get_prior_folders():
    possible_folders = glob.glob(os.path.join(config.eval_dir, f"n={config.n_across_time}", "*"))
    folders = list(filter(lambda path : os.path.isdir(path), possible_folders))
    return folders
    
def get_stopword_set():
    with open('./nltk_stopwords.txt') as f:
        lines = list(map(lambda s : s.strip(), f.readlines()[1:])) # Omit the citation
        stopword_set = set(lines)
    return stopword_set
    
def filter_for_stopwords(df):
    stopword_set = get_stopword_set()
    return df[df['token'].isin(stopword_set)].copy()
