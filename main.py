import os
import numpy as np
# from torch_snippets import *
from argparse import ArgumentParser

from UnetModule import UnetModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from torchvision import transforms
from pytorch_lightning import Trainer, seed_everything
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

def main(hparams):

    model = UnetModule(hparams.dataset, hparams.n_channels, hparams.n_classes)

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        filename='{epoch}-{val_loss:.2f}',
        save_last=True,
        verbose=True,
    )
    
    # stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     verbose=True,
    # )

    trainer = Trainer(
        gpus=1,
        max_epochs=20,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=6)

    hparams = parser.parse_args()
    
    main(hparams)