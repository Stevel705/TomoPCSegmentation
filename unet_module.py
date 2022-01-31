import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from model.unet import UNet
from utils import dice_coeff, multiclass_dice_coeff

class LightningUnet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, learning_rate, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.bilinear = bilinear
        self.net = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=self.bilinear)

        if self.n_classes > 1:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # --------------------------
        x, y = batch['image'], batch['mask']
        if self.n_classes == 1:
            mask_type = torch.float32
            y = y.type(mask_type)
        else:
            mask_type = torch.long
            y = y.type(mask_type).squeeze(0)

        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        
        if self.n_classes == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
            dice_acc = dice_coeff(y_pred, y).item()
        else:
            y = F.one_hot(y, self.n_classes).permute(0, 3, 1, 2).float()
            y_pred = F.one_hot(y_pred.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()
            dice_acc = multiclass_dice_coeff(y_pred, y).item()
    
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', dice_acc, on_epoch=True, prog_bar=True, logger=True)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        x, y = batch['image'], batch['mask']
        if self.n_classes == 1:
            mask_type = torch.float32
            y = y.type(mask_type)
        else:
            mask_type = torch.long
            y = y.type(mask_type).squeeze(0)

        y_pred = self.net(x)

        loss = self.criterion(y_pred, y)

        if self.n_classes == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > 0.5).float()
            dice_acc = dice_coeff(y_pred, y).item()
        else:
            y = F.one_hot(y, self.n_classes).permute(0, 3, 1, 2).float()
            y_pred = F.one_hot(y_pred.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()
            dice_acc = multiclass_dice_coeff(y_pred, y).item()

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', dice_acc, on_epoch=True, prog_bar=True, logger=True)
        # return loss
        # --------------------------

    def test_step(self, batch, batch_idx):
        # --------------------------
        x, y = batch['image'], batch['mask']
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=1e-8)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}
        return ([optimizer], [scheduler])