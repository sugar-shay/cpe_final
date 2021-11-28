# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 09:05:04 2021

@author: Shadow
"""

import torch 
import transformers
from datasets import load_dataset

from sklearn.metrics import classification_report
import pickle

from tokenize_data import *
from LIT_SENTIMENT import *

def main():
    
    #model_checkpoint = 'ProsusAI/finbert'
    model_checkpoint = 'bert-base-uncased'
    finetune_dataset = 'financial_phrasebank'
    
    #label 2 correspnds to positive sentiment 
    #label 1 is neutral 
    #label 0 is negative 
    train_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[:70%]')
    val_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[70%:85%]')
    test_data = load_dataset(finetune_dataset, 'sentences_75agree', split='train[85%:]')
    
    
    #need to find the average length of the sequences
    total_avg = sum( map(len, list(train_data['sentence'])) ) / len(train_data['sentence'])
    print('Avg. sentence length: ', total_avg)
    
    tokenizer = Sentiment_Tokenizer(max_length=256, tokenizer_name = model_checkpoint, continious_output=False)
    
    train_dataset = tokenizer.tokenize_and_encode_labels(train_data)
    val_dataset = tokenizer.tokenize_and_encode_labels(val_data)
    test_dataset = tokenizer.tokenize_and_encode_labels(test_data)
    
    model = LIT_SENTIMENT(model_checkpoint = model_checkpoint,
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     save_fp='best_model.pt',
                     continious_output=False)
    
    model = train_LitModel(model, train_dataset, val_dataset, max_epochs=15, batch_size=16, patience = 3, num_gpu=1)
    
    
    #saving the training stats
    with open('train_stats.pkl', 'wb') as f:
        pickle.dump(model.training_stats, f)
    
    model = LIT_SENTIMENT(model_checkpoint = model_checkpoint,
                     hidden_dropout_prob=.1,
                     attention_probs_dropout_prob=.1,
                     save_fp='best_model.pt', 
                     continious_output=False)
    
    model.load_state_dict(torch.load('best_model.pt'))
    
    preds, ground_truths = model_testing(model, test_dataset)
    
    
    '''
    print('raw preds shape: ', len(preds))
    
    #final_preds = postprocess_predictions(preds)
    print('final preds shape: ', len(final_preds))
    print('test data shape: ', len(test_data['label']))
    '''
    cr = classification_report(y_true=test_data['label'], y_pred = preds, output_dict = False)
    
    print()
    print(cr)


if __name__ == "__main__":
    main()
    



        
    


