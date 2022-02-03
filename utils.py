from pickletools import optimize
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import functional as FM
from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


class LitModel(LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.model.fc = nn.Linear(512, 10)
        self.lr = lr
        self.result_dict = {'val_loss':[]}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        epoch = self.trainer.current_epoch
        train_loss, train_acc = self.trainer.callback_metrics['train_loss'], self.trainer.callback_metrics['train_acc']
        val_loss, val_acc = self.trainer.callback_metrics['val_loss'], self.trainer.callback_metrics['val_acc']
        print(f'epoch: {epoch:2d} [train_loss: {train_loss:0.4f} val_loss: {val_loss:0.4f}] [train_acc: {train_acc:0.4f} val_acc: {val_acc:0.4f}]')
        self.current_val_loss = val_loss 


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc} 

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = FM.accuracy(logits, y)
        loss = F.cross_entropy(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode='min', verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
       }