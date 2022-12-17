
import collections
from src.utils import generation_processing
import numpy as np

def check_counts_descending(df):
    return np.sort(df['count'])[::-1].tolist() == list(df['count'])
    
def get_substitution_counter(all_scores):
    
    word_tuples = []
    for entry_index in range(all_scores.shape[0]):
        entry = all_scores.iloc[entry_index]
        highest_posterior_words = generation_processing.get_tied_highest_posterior_words(entry)
        word_tuples.extend([(entry.token, word) for word in highest_posterior_words])
        
    unique_tuples = sorted(list(set(word_tuples)))
    # Adapted from 12/17/22: https://note.nkmk.me/en/python-collections-counter/
    counts = collections.Counter(word_tuples)
    # end cite
    
    return counts
    
    