# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 09:46:00 2021

@author: Shadow
"""

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, classification_report
from pytorch_lightning.callbacks import EarlyStopping
import os 

class LIT_SENTIMENT(pl.LightningModule):
    def __init__(self,  
                 model_checkpoint,
                 hidden_dropout_prob=.5,
                 attention_probs_dropout_prob=.2,
                 save_fp='best_model.pt'):
       
        super(LIT_SENTIMENT, self).__init__()
        
        
        self.build_model(hidden_dropout_prob, attention_probs_dropout_prob, model_checkpoint)
        
        self.training_stats = {'train_losses':[],
                               'val_losses':[]}
        
        self.save_fp = save_fp
        self.fc1 = nn.Linear(768, 1)
        self.criterion = nn.MSELoss()
    
    def build_model(self, hidden_dropout_prob, attention_probs_dropout_prob, model_checkpoint):
        config = AutoConfig.from_pretrained(model_checkpoint)
        #These are the only two dropouts that we can set
        config.hidden_dropout_prob = hidden_dropout_prob
        config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder = AutoModel.from_pretrained(model_checkpoint, config=config)
        
    def save_model(self):
        
        '''
        print()
        print('Class Attributes: ', self.__dict__)
        print()
        '''
        torch.save(self.state_dict(), self.save_fp)
        
    def forward(self, input_ids, attention_mask):
        pooler_output = self.encoder(input_ids= input_ids, attention_mask= attention_mask)
        
        (cls_hs, last_hidden_state) = pooler_output.to_tuple()
 
        x = self.fc1(last_hidden_state)
        
        logits =  torch.tanh(x)
        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        # Run Forward Pass
        logits = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

        # Compute Loss (Cross-Entropy)
        loss = self.criterion(batch['polarity'], logits)
        
        
        # Set up Data to be Logged
        return {"loss": loss, 'train_loss': loss}

    def training_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["train_loss"] for x in outputs]).mean()
        self.training_stats['train_losses'].append(avg_loss.detach().cpu())
        print('Train Loss: ', avg_loss.detach().cpu().numpy())
        self.log('train_loss', avg_loss)
        
    def validation_step(self, batch, batch_idx):

        # Run Forward Pass
        logits = self.forward(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

        # Compute Loss (Cross-Entropy)
        loss = self.criterion(batch['polarity'], logits)
        
       
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # Outputs --> List of Individual Step Outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        avg_loss_cpu = avg_loss.detach().cpu().numpy()
        if len(self.training_stats['val_losses']) == 0 or avg_loss_cpu<np.min(self.training_stats['val_losses']):
            self.save_model()
            
        self.training_stats['val_losses'].append(avg_loss_cpu)
        print('Val Loss: ', avg_loss_cpu)
        self.log('val_loss', avg_loss)
        
        


def train_LitModel(model, train_data, val_data, max_epochs, batch_size, patience = 3, num_gpu=1):
    
    #
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=False)#, num_workers=8)#, num_workers=16)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle = False)
    
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False, mode="min")
    
    trainer = pl.Trainer(gpus=num_gpu, max_epochs = max_epochs, callbacks = [early_stop_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
        
    return model


def model_testing(model, test_dataset):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = model.to(device)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    preds, total_polarity = [], []
    
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        
        seq = (batch['input_ids']).to(device)
        mask = (batch['attention_mask']).to(device)
        polarity = batch['polarity']
        
        logits = model(input_ids=seq, attention_mask=mask, labels=None)
        
        preds.extend(logits.detach().cpu().numpy())
        total_polarity.extend(polarity)
        
    
    return preds, total_polarity

def postprocess_predictions(raw_predictions):
    
    final_predictions = []
    for pred in raw_predictions:
        
        if -1.0 >= pred <-.3333:
            final_predictions.append('negative')
        elif -.3333 >= pred <= .3333:
            final_predictions.append('neutral')
        elif .3333 <= pred <= 1.0:
            final_predictions.append('neutral')
    
    return final_predictions

