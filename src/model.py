import os
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split
import timm 
import numpy as np 
from scipy import spatial
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms 
import pytorch_lightning as pl 
from torchmetrics.functional import pairwise_cosine_similarity

from transformers import AutoProcessor, BlipForConditionalGeneration

from dataset import DiffusionDataModule, BLIPDataModule
from config import CONFIG, BLIPConfig


class VisionTransformer(pl.LightningModule):

    def __init__(self, t_max):
        super(VisionTransformer, self).__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(
            CONFIG.model_name,
            pretrained = True, 
            num_classes = 384
        )
        self.model.set_grad_checkpointing()
        self.criterion = nn.CosineEmbeddingLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        _, X, y = batch 
        y_hat = self.model(X) 
        target = torch.ones(X.size(0)).to(self.device)
        loss = self.criterion(y_hat, y, target)
        cosine_similarity = self.cosine_similarity(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_cos_similarity", cosine_similarity, on_step=True, on_epoch=True, logger=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        _, X, y = batch 
        y_hat = self.model(X) 
        target = torch.ones(X.size(0)).to(self.device)
        loss = self.criterion(y_hat, y, target)
        cosine_similarity = self.cosine_similarity(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_cos_similarity", cosine_similarity, on_step=True, on_epoch=True, logger=True)
        return loss 

    def test_step(self, batch, batch_idx):
        _, X, y = batch 
        y_hat = self.model(X) 
        target = torch.ones(X.size(0)).to(self.device)
        loss = self.criterion(y_hat, y, target)
        cosine_similarity = self.cosine_similarity(y_hat, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_cos_similarity", cosine_similarity, on_step=True, on_epoch=True, logger=True)
        return loss 
    
    def cosine_similarity(self, y_preds, y_trues):
        y_trues = y_trues.detach().cpu().numpy()
        y_preds = y_preds.detach().cpu().numpy()
        return torch.tensor(np.mean([
            1 - spatial.distance.cosine(y_true, y_pred) 
            for y_true, y_pred in zip(y_trues, y_preds)
        ]))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=CONFIG.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.t_max, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser
    

class BLIP(pl.LightningModule):

    def __init__(self):
        super(BLIP, self).__init__()
        self.save_hyperparameters()
        self.model = BlipForConditionalGeneration.from_pretrained(BLIPConfig.model_name)

    def forward(self, input_ids, pixel_values, labels):
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        _, data = batch 
        input_ids = data['input_ids']
        pixel_values = data['pixel_values']
        y_hat = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        ) 
        loss = y_hat.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def validation_step(self, batch, batch_idx):
        _, data = batch 
        input_ids = data['input_ids']
        pixel_values = data['pixel_values']
        y_hat = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        ) 
        loss = y_hat.loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 

    def test_step(self, batch, batch_idx):
        _, data = batch 
        input_ids = data['input_ids']
        pixel_values = data['pixel_values']
        y_hat = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        ) 
        loss = y_hat.loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=BLIPConfig.learning_rate, weight_decay=BLIPConfig.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BLIPConfig.T_max, eta_min=BLIPConfig.min_lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        return parser

if __name__ == "__main__":
    metadata_df = pd.read_parquet('../metadata.parquet')
    images = os.listdir('../images/')
    metadata_df = metadata_df[metadata_df['image_name'].isin(images)]
    train_df, test_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((CONFIG.input_size, CONFIG.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dm = DiffusionDataModule('../images/', train_df, test_df, train_transform=transform, test_transform=transform, batch_size=CONFIG.batch_size)
    dm.prepare_data()
    dm.setup()

    num_epochs = 100
    t_max = num_epochs * len(dm.train_dataloader())

    model = VisionTransformer(t_max=t_max)

    print(model)

    for _, X, y in dm.train_dataloader():
        y_hat = model(X)
        print(y_hat, y)
        print(y_hat.shape, y.shape)
        print(F.cosine_embedding_loss(y_hat, y, torch.ones(X.size(0))))
        break

    processor =  AutoProcessor.from_pretrained(BLIPConfig.model_name)
    dm = BLIPDataModule('../images', train_df, test_df, processor)
    dm.prepare_data()
    dm.setup() 

    model = BLIP()

    print(model)

    for _, data in dm.train_dataloader():
        input_ids = data['input_ids']
        pixel_values = data['pixel_values']
        y_hat = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            labels=input_ids
        )

        print(y_hat.keys())
        print(y_hat.loss)
        break
