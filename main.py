import os
from argparse import ArgumentParser

from unet_module import LightningUnet
from datasets.olfactory_bulb import OlfactoryBulbDataset

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# from pl_bolts.callbacks import PrintTableMetricsCallback

def main(hparams):

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
    
    BATCH_SIZE = 1
    NUM_WORKERS = int(os.cpu_count() / 2)

    train = OlfactoryBulbDataset(hparams.dataset, 'train', is_transform=True)
    train = DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val = OlfactoryBulbDataset(hparams.dataset, 'val', is_transform=True)
    val = DataLoader(val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # test = OlfactoryBulbDataset(hparams.dataset, 'test', is_transform=False)
    # test = DataLoader(test, batch_size=BATCH_SIZE)

    # init model
    model = LightningUnet(hparams.n_channels, hparams.n_classes, hparams.learning_rate)

    # Initialize a trainer
    logger = TensorBoardLogger(hparams.log_dir, name="my_model")
    trainer = pl.Trainer(gpus=1, max_epochs=1, progress_bar_refresh_rate=20, logger=logger)

    # Train the model ⚡
    trainer.fit(model, train, val)
    
    # # testing
    # result = trainer.test(test_dataloaders=test)
    # print(result)


if __name__ == '__main__':
    pl.seed_everything(42)
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--log_dir', default='lightning_logs')
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    hparams = parser.parse_args()
    
    main(hparams)