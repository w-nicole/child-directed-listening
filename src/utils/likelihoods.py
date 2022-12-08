import sys
from joblib import Parallel, delayed
import os
import copy
import pandas as pd
import numpy as np
import time
import Levenshtein

from src.utils import configuration
config = configuration.Config()

def normalize_log_probs(vec):
    vec = vec.values.flatten()
    ps = np.exp(-1 * vec)
    total = np.sum(ps)
    return(ps / total)

def normalize_partition(x): 
    '''for a given selection of FST arcs, for example all where input is a particular symbol, normalize the log probs'''
    df = x[1]
    df[[4]] = -1 * np.log(normalize_log_probs(df[[4]]))
    return(df)

def split(a, n):
    '''split a list into n approximately equal length sublists, appropriate for parallelization'''
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def get_edit_distance_matrix(all_tokens_phono, prior_data,  cmu_2syl_inchildes):    
    '''
    Get an edit distance matrix for matrix-based computation of the posterior.

    all_tokens_phono: corpus in tokenized from, with phonological transcriptions
    prior_data: priors of the form output by `compare_successes_failures_*`    
    cmu_2syl_inchildes: cmu pronunctiations, must have 'word' and 'ipa_short' columns 

    returns: a matrix where each row is an input string from prior_data and each column is a different pronunciation in cmu_2syl_inchildes.
    thus a word type may correspond to multiple columns, and must be reduced using the wfst.reduce_duplicates function
    '''

    print('Getting the Levenshtein distance matrix')

    bert_token_ids = prior_data['scores']['bert_token_id']
    ipa = pd.DataFrame({'bert_token_id':bert_token_ids}).merge(all_tokens_phono[['bert_token_id',
        'actual_phonology_no_dia']])


    levdists = np.vstack([np.array([Levenshtein.distance(target,x) for x in cmu_2syl_inchildes.ipa_short
    ]) for target in ipa.actual_phonology_no_dia]) 
    return(levdists)


def reduce_duplicates(wfst_dists, cmu_2syl_inchildes, initial_vocab, max_or_min, cmu_indices_for_initial_vocab):
    '''
    Take a (d x w) distance matrix that includes multiple pronunciations for the same word as separate columns, and return a distance matrix that takes the highest-probability (or lowest distance) true pronunciation for every observation d.
    `wfst_dists`: matrix that includes multiple pronunciations for the same word as separate columns
    `cmu_2syl_inchildes`: DataFrame with `word` column, where words include duplicates for multiple pronunciations, and words are in the same order corresponding to `wfst_dists`
    
    outputs a matrix `wfst_dists_by_word` where each row corresponds to a production and each column correpsonds to a word in initial_vocab
    '''    
    
    wfst_dists_by_word = np.zeros([wfst_dists.shape[0], len(initial_vocab)])  

    for target_production_index in range(wfst_dists.shape[0]):
        for vocab_index in range(len(initial_vocab)):
        
            #find indices where 
            cmu_2syl_indices = cmu_indices_for_initial_vocab[vocab_index]
            if max_or_min == 'max':
                dist = np.max(wfst_dists[target_production_index,cmu_2syl_indices])
            elif max_or_min == 'min':
                dist = np.min(wfst_dists[target_production_index,cmu_2syl_indices])
        
            wfst_dists_by_word[target_production_index, vocab_index] = dist

    return wfst_dists_by_word
