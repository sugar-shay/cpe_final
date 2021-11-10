# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:34:41 2021

@author: Shadow
"""


import numpy as np
import torch


from transformers import AutoTokenizer

class Sentiment_Tokenizer():
    
    def __init__(self, max_length, tokenizer_name = None): #unique_labels
        #self.unique_tags = unique_labels
        #self.tag2id = {tag: id for id, tag in enumerate(self.unique_tags)}
        #self.id2tag = {id: tag for tag, id in self.tag2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True) if tokenizer_name is not None else AutoTokenizer.from_pretrained('bert-base-cased', use_fast=True)
        self.max_len = max_length
    
    
    def tokenize_and_encode_labels(self, dataset):
        
        labels = dataset['label']
        sentences = dataset['sentence']
        
        encodings = self.tokenizer(sentences, is_split_into_words=False, return_offsets_mapping=False, max_length = self.max_len, padding=True, truncation=True)
        
        polarity = self.encode_tags(labels)
        
        dataset = Sentiment_Dataset(encodings, polarity)
        
        return dataset
            
        
    def encode_tags(self, tags):
        
        #label 2 correspnds to positive sentiment 
        #label 1 is neutral 
        #label 0 is negative 
        polarity = []
        for i in range(len(tags)):
            
            tag = tags[i]
            if tag == 2:
                polarity.append(1.0)
            elif tag == 1:
                polarity.append(0.0)
            else:
                polarity.append(-1.0)
        
        return polarity
            

  
class Sentiment_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, polarity):
        self.encodings = encodings
        self.polarity = polarity
    
    def __getitem__(self, idx):
        item = {key: torch.LongTensor(val[idx]) for key, val in self.encodings.items()}
        if self.polarity is not None:
            item['polarity'] = torch.as_tensor(self.polarity[idx])
        return item
    
    def __len__(self):
        return len(self.encodings['input_ids'])

        


