import os
from os.path import join, exists
import pandas as pd
import numpy as np
import transformers 
import json
import copy
from transformers import BertTokenizer, BertForMaskedLM
from src.utils import configuration, transformers_bert_completions, split_gen, load_splits, paths
config = configuration.Config()
        
    
def get_vocab_tok_modules():
    
    cmu_2syl_inchildes = get_cmu_dict_info()
    
    adult_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    adult_softmax_mask, adult_vocab = transformers_bert_completions.get_softmax_mask(adult_tokenizer, cmu_2syl_inchildes.word)
    
    initial_tokenizer = get_primary_tokenizer()
    
    _, initial_vocab = transformers_bert_completions.get_softmax_mask(initial_tokenizer,
    cmu_2syl_inchildes.word)  
    
    return adult_tokenizer, adult_softmax_mask, initial_tokenizer, initial_vocab


def get_data_unigram_dict(fitted_dict):
    
    adult_tokenizer, adult_softmax_mask, _, initial_vocab = get_vocab_tok_modules()
    

    fitted_dict['title'] = paths.get_file_identifier(fitted_dict)
    fitted_dict['kwargs'] = {'child_counts_path': paths.get_chi_vocab_path(),
                    'tokenizer': adult_tokenizer,
                    'softmax_mask': adult_softmax_mask,
                    'vocab': initial_vocab,

                    # Added these default args 7/9/21 for compatibility with rest of the code
                    'context_width_in_utts': 0,
                    'use_speaker_labels': False,
                   }
    return(fitted_dict)

def get_flat_unigram_dict(fitted_dict):
    
    adult_tokenizer, adult_softmax_mask, _, initial_vocab = get_vocab_tok_modules()
    
    fitted_dict['title'] = paths.get_file_identifier(fitted_dict)
            # Note that this assumes that flat prior = no information at all.
            # That means it doesn't observe any train/val split.
            # It just assigns uniform probability to every single word,
            # regardless of if that word appears in the train set or not. 
    fitted_dict['kwargs'] = {'child_counts_path': None,
                        'tokenizer': adult_tokenizer,
                        'softmax_mask': adult_softmax_mask,
                        'vocab': initial_vocab,
                       
                        # Added these default args 7/9/21 for compatibility with rest of the code
                        'context_width_in_utts': 0,
                        'use_speaker_labels': False,
                       }
    return(fitted_dict)

       
        
def get_finetune_dict(fitted_dict):

    # make a model dict 
    model_dict = copy.copy(fitted_dict)
    model_dict['task_phase'] = 'train'
    model_dict['test_dataset'] = None
    model_dict['test_split'] = None  
    model_dict['context_width'] = None  


    fitted_dict['title'] =  paths.get_file_identifier(fitted_dict)
    fitted_dict['kwargs'] = get_model_from_split(model_dict)    
    fitted_dict['kwargs']['context_width_in_utts'] = fitted_dict['context_width']
    fitted_dict['kwargs']['use_speaker_labels'] = fitted_dict['use_tags']
    return fitted_dict



def get_fitted_model_dict(fitted_dict):
    """
    Only for age/all splits.
    """
    
    # The format for the name is:
    # split name/dataset name/tags/{context width}_context
    # for childes data.
    
    # If it's a pretrained BERT model with no finetuning, it has /shelf added to its model id
    # If it's a unigram model, it's just: split name/dataset name/unigram_{unigram type}
    
    if fitted_dict['model_type'] == 'BERT':
        assert not fitted_dict['training_split'] == 'adult-written'
        model_dict = get_finetune_dict(fitted_dict)
        
        vocab = [x for x in model_dict['kwargs']['tokenizer'].vocab]

        for token in ['[chi]','[cgv]', 'yyy','xxx']:
            if token not in vocab:
                model_dict['kwargs']['tokenizer'].add_tokens(token)
        

    elif fitted_dict['model_type'] == 'data_unigram': 
        model_dict = get_data_unigram_dict(fitted_dict)
    elif fitted_dict['model_type'] == 'flat_unigram':
        model_dict = get_flat_unigram_dict(fitted_dict)
    
    return model_dict
    
    
def get_primary_tokenizer():

    shelf_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    shelf_tok.add_tokens(['[chi]', '[cgv]', 'yyy', 'xxx'])

    return shelf_tok

    
def get_initial_vocab_info(initial_tokenizer = None):
    
    """
    Satisfies all constraints of successes if syllables counted via CMU (not by actual_phonology or model_phonology) 
    """
    
    # tokenize with the most extensive tokenizer, which is the one used for model #2
    
    if initial_tokenizer is None:
        initial_tokenizer = get_primary_tokenizer() # default argument
        
    cmu_2syl_inchildes = get_cmu_dict_info()
    cmu_2syl_inchildes_reduced = np.unique(cmu_2syl_inchildes.word)
    
    inital_vocab_mask, initial_vocab = transformers_bert_completions.get_softmax_mask(initial_tokenizer,
        cmu_2syl_inchildes_reduced)
        
    # get the indiices in cmu_2syl for each unique word type in initial_vocab
    cmu_indices_for_initial_vocab = [np.argwhere(cmu_2syl_inchildes.word.values == x).flatten() for x in initial_vocab]
    return initial_vocab, cmu_2syl_inchildes, cmu_indices_for_initial_vocab


def get_model_from_split(model_dict):
    
    model_path = paths.get_directory(model_dict)
        
    word_info_all = get_cmu_dict_info()
    word_info = word_info_all.word 
    
    try:
        model = BertForMaskedLM.from_pretrained(model_path)
    except BaseException as e:
        print('Model loading failed. Does a model actually exist at '+model_path)
        print(e)
        raise ValueError('Terminating!')

    
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(model_path)
    softmax_mask, vocab = transformers_bert_completions.get_softmax_mask(tokenizer, word_info)
    
    return {'modelLM' : model, 'tokenizer' : tokenizer, 'softmax_mask' : softmax_mask, 'use_speaker_labels' : model_dict['use_tags']}


def get_cmu_dict_info():
    cmu_in_childes = pd.read_pickle(config.cmu_path)    
    return cmu_in_childes 

