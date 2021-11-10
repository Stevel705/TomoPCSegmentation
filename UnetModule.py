# import os
# import logging
# from argparse import ArgumentParser
# from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torch import optim
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from torchmetrics.functional import dice_score

from datasets.olfactory_bulb import OlfactoryBulbDataset
from model.unet import UNet
import numpy as np

from utils import multiclass_dice_coeff

def UnetLoss(preds, targets, ce, n_classes):
    # targets = targets.squeeze(0)
    ce_loss = ce(preds, targets)
    
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    mask_true = F.one_hot(targets, n_classes).permute(0, 3, 1, 2).float()
    mask_pred = F.one_hot(preds.argmax(dim=1), n_classes).permute(0, 3, 1, 2).float()
    acc = multiclass_dice_coeff(mask_pred, mask_true)

    return ce_loss, acc

class UnetModule(pl.LightningModule):
    def __init__(self, dataset, n_channels, n_classes):
        super().__init__()
        # self.hparams = hparams
        self.dataset = dataset
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = True
        # self.criterion =  nn.CrossEntropyLoss() if self.n_classes > 1 else \
        #     nn.BCEWithLogitsLoss()

        # self.loss = UnetLoss
        self.net = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=self.bilinear)
        self.save_hyperparameters()

    def training_step(self, batch, batch_nb):
        x, y = batch['image'], batch['mask']
        y_hat = self.net(x)
        # print(np.unique(y_hat.cpu().detach().numpy()), np.unique(y.cpu().detach().numpy()))

        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)

        dice_acc = dice_score(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_acc", dice_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        tensorboard_logs = {'train_loss': loss, "train_acc": dice_acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch['image'], batch['mask']
        y_hat = self.net(x)

        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy_with_logits(y_hat, y)
        
        dice_acc = dice_score(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("val_acc", dice_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        tensorboard_logs = {'train_loss': loss, "train_acc": dice_acc}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.0001, weight_decay=1e-8)

    def __dataloader(self):
        dataset = self.dataset
        dataset = OlfactoryBulbDataset(dataset)
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=True, shuffle=False)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['val']
