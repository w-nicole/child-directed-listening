
from collections import defaultdict
from src.utils import generation_processing
import numpy as np

def check_counts_descending(df):
    return np.sort(df['count'])[::-1].tolist() == list(df['count'])
    
def get_substitution_counter(all_scores):
    
    counts = defaultdict(int)
    for entry_index in range(all_scores.shape[0]):
        entry = all_scores.iloc[entry_index]
        highest_posterior_words = generation_processing.get_tied_highest_posterior_words(entry)
        for word in highest_posterior_words:
            current_tuple = (entry.token, word)
            counts[current_tuple] += 1
            
    unique_tuples = sorted(list(counts.keys()))
    
    return unique_tuples, counts
    
    