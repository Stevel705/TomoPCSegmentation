import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from model.unet import UNet
from utils import dice_coeff, multiclass_dice_coeff, average_area_error
import segmentation_models_pytorch as smp


class LightningUnet(pl.LightningModule):
    def __init__(self, n_channels, n_classes, learning_rate, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.bilinear = bilinear
        self.net = UNet(n_channels=self.n_channels, n_classes=self.n_classes, bilinear=self.bilinear)
        # self.net = smp.Unet(
        #     encoder_name="efficientnet-b5",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     in_channels=self.n_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=self.n_classes,                      # model output channels (number of classes in your dataset)
        # )
        self.train_loss = []
        self.val_loss = []
        self.train_metric = []
        self.train_metric2 = []
        self.val_metric = []
        self.val_metric2 = []

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
            metric = dice_coeff(y_pred, y)
            metric2 = average_area_error(y_pred, y)
        else:
            y = F.one_hot(y, self.n_classes).permute(0, 3, 1, 2).float()
            y_pred = F.one_hot(y_pred.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()
            metric = multiclass_dice_coeff(y_pred, y)

        # self.log('train_loss', loss, prog_bar=True, logger=True)
        # self.log('train_acc', metric.item(), prog_bar=True, logger=True)
        return {"loss": loss, "metric": metric, "metric2": metric2}

    def training_epoch_end(self, outs):
        # --------------------------
        # outs is a list of whatever you returned in `validation_step`
        # loss = torch.stack(outs).mean()
        loss = torch.stack([x["loss"] for x in outs]).mean()
        metric = torch.stack([x["metric"] for x in outs]).mean()
        metric2 = torch.stack([x["metric2"] for x in outs]).mean()
        self.train_loss.append(loss.detach().cpu().item())
        self.train_metric.append(metric.detach().cpu().item())
        self.train_metric2.append(metric2.detach().cpu().item())
        
        # self.log('train_loss', loss,  prog_bar=True, logger=True)
        # self.log('train_acc', metric, prog_bar=True, logger=True)
        tensorboard_logs = {'train_loss': loss, 'train_acc': metric, "train_acc2": metric2, 'step': self.current_epoch}
        self.log_dict(tensorboard_logs)
        # return {'loss': loss, 'log': tensorboard_logs}

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
            metric = dice_coeff(y_pred, y)
            metric2 = average_area_error(y_pred, y)
        else:
            y = F.one_hot(y, self.n_classes).permute(0, 3, 1, 2).float()
            y_pred = F.one_hot(y_pred.argmax(dim=1), self.n_classes).permute(0, 3, 1, 2).float()
            metric = multiclass_dice_coeff(y_pred, y)

        # self.log('val_loss', loss, prog_bar=True, logger=True)
        # self.log('val_acc', metric.item(), prog_bar=True, logger=True)
        return {"loss": loss, "metric": metric, "metric2": metric2}
    
    def validation_epoch_end(self, outs):
        # --------------------------
        # outs is a list of whatever you returned in `validation_step`
        # loss = torch.stack(outs).mean()
        loss = torch.stack([x["loss"] for x in outs]).mean()
        metric = torch.stack([x["metric"] for x in outs]).mean()
        metric2 = torch.stack([x["metric2"] for x in outs]).mean()
        self.val_loss.append(loss.detach().cpu().item())
        self.val_metric.append(metric.detach().cpu().item())
        self.val_metric2.append(metric2.detach().cpu().item())

        # self.log('val_loss', loss, prog_bar=True, logger=True)
        # self.log('val_acc', metric, prog_bar=True, logger=True)
        tensorboard_logs = {'val_loss': loss, 'val_acc': metric,  "val_acc2": metric2, 'step': self.current_epoch}
        self.log_dict(tensorboard_logs)

    def test_step(self, batch, batch_idx):
        # --------------------------
        x, y = batch['image'], batch['mask']
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-8)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
        scheduler = {'scheduler': lr_scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}
        return ([optimizer], [scheduler])