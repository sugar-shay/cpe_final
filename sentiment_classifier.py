# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:54:54 2021

@author: Shadow
"""

import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, classification_report
from pytorch_lightning.callbacks import EarlyStopping
import os 

class SENTIMENT_CLASSIFIER(pl.LightningModule):
    def __init__(self,  
                 model_checkpoint,
                 save_fp='best_model.pt'):
       
        super(SENTIMENT_CLASSIFIER, self).__init__()
        
        self.build_model(model_checkpoint)
        
        self.training_stats = {'train_losses':[],
                               'val_losses':[]}
        
        self.save_fp = save_fp
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
    
    def build_model(self, model_checkpoint):
       
        config = AutoConfig.from_pretrained(model_checkpoint)
        config.num_labels = 3
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
        
    def save_model(self):
        torch.save(self.state_dict(), self.save_fp)
        
    def forward(self, input_ids, attention_mask):
        
        output = self.encoder(input_ids= input_ids, attention_mask= attention_mask)
        logits = self.softmax(output.logits)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        
        # Run Forward Pass
        logits = self.encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        #loss = output.loss
        # Compute Loss
        loss = self.criterion(logits, batch['label'])
        
        
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

        # Compute Loss
        loss = self.criterion(logits, batch['label'])
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
        
        


def train_sentiment_classifier(model, train_data, val_data, max_epochs, batch_size, patience = 3, num_gpu=1):
    
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
    
    preds, ground_truths = [], []
    
    model.eval()
    for idx, batch in enumerate(test_dataloader):
        
        seq = (batch['input_ids']).to(device)
        mask = (batch['attention_mask']).to(device)
        
        label = batch['label']
        ground_truths.extend(label)
        
        logits = model(input_ids=seq, attention_mask=mask)
        logits = logits.detach().cpu().numpy()
        preds.extend(np.argmax(logits, axis = -1))
        
    
    return preds, ground_truths
