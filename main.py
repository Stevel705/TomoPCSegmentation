import os
from argparse import ArgumentParser

from unet_module import LightningUnet
from datasets.olfactory_bulb import OlfactoryBulbDataset

from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from pl_bolts.callbacks import PrintTableMetricsCallback


def plot_grapic(epochs, loss_or_metric, label='', type_line='go-'):
    plt.plot(range(len(loss_or_metric)), loss_or_metric, type_line, label=label, linewidth=2)
    plt.grid(True)
    plt.legend()

def main(hparams):

    os.makedirs(hparams.log_dir, exist_ok=True)
    # os.makedirs(hparams.images_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join(log_dir, 'checkpoints'),
    #     filename='{epoch}-{val_loss:.2f}',
    #     save_last=True,
    #     verbose=True,
    # )
    
    BATCH_SIZE = 1
    NUM_WORKERS = int(os.cpu_count() / 2)
    K_FOLDS = 4
    MAX_EPOCHS = 3
    IS_KFOLD = False
    
    train = OlfactoryBulbDataset(hparams.dataset, 'train', scale=0.5, is_transform=True, n_classes=hparams.n_classes)
    val = OlfactoryBulbDataset(hparams.dataset, 'val', scale=0.5, is_transform=False, n_classes=hparams.n_classes)

    if IS_KFOLD:
        kfold = KFold(n_splits=K_FOLDS, shuffle=False)
        
        dataset = ConcatDataset([train, val])

        for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
            print(f'FOLD {fold}')
            print('--------------------------------')

            train_subsampler = SubsetRandomSampler(train_idx)
            test_subsampler = SubsetRandomSampler(test_idx)

            train = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=train_subsampler)
            val = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=test_subsampler)

            # test = OlfactoryBulbDataset(hparams.dataset, 'test', is_transform=False)
            # test_dataloader = DataLoader(test, batch_size=BATCH_SIZE)

            # init model
            model = LightningUnet(hparams.n_channels, hparams.n_classes, hparams.learning_rate)
            name = f"model_{hparams.n_classes}_classes_fold_{fold}"
            # Initialize a trainer
            logger = TensorBoardLogger(hparams.log_dir, name=name)
            trainer = pl.Trainer(gpus=1, max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=20, logger=logger)

            # Train the model ⚡
            trainer.fit(model, train, val)
            
            # # testing
            # result = trainer.test(test_dataloaders=test_dataloader)
            # print(result)

            # # automatically auto-loads the best weights from the previous run
            # predictions = trainer.predict(dataloaders=predict_dataloader)

            # # or call with pretrained model
            # model = MyLightningModule.load_from_checkpoint(PATH)
            # trainer = Trainer()
            # predictions = trainer.predict(model, dataloaders=test_dataloader)

            # ================================================
            plot1 = plt.figure()
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel("Epochs")
            plt.ylabel("Categorical Loss")
            plot_grapic(MAX_EPOCHS, model.train_loss, label="Train loss", type_line='-')
            plot_grapic(MAX_EPOCHS, model.val_loss, label="Validation loss", type_line='--')
            plt.savefig(f"images/loss_{name}.png")
            plt.close()

            plot2 = plt.figure(2)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel("Epochs")
            plt.ylabel("Dice Coefficient")
            plot_grapic(MAX_EPOCHS, model.train_metric, label="Train metric", type_line='-')
            plot_grapic(MAX_EPOCHS, model.val_metric, label="Validation metric", type_line='--')
            plt.savefig(f"images/metric_{name}.png")
            plt.close()

            plot3 = plt.figure(3)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel("Epochs")
            plt.ylabel("Average area error")
            plot_grapic(MAX_EPOCHS, model.train_metric, label="Train metric", type_line='-')
            plot_grapic(MAX_EPOCHS, model.val_metric, label="Validation metric", type_line='--')
            plt.savefig(f"images/metric2_{name}.png")
            plt.close()

    else: 
        train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        # test = OlfactoryBulbDataset(hparams.dataset, 'test', is_transform=False)
        # test_dataloader = DataLoader(test, batch_size=BATCH_SIZE)

        # init model
        model = LightningUnet(hparams.n_channels, hparams.n_classes, hparams.learning_rate)

        # Initialize a trainer
        logger = TensorBoardLogger(hparams.log_dir, name=f"model_{hparams.n_classes}_classes")
        trainer = pl.Trainer(gpus=1, max_epochs=MAX_EPOCHS, progress_bar_refresh_rate=20, logger=logger)

        # Train the model ⚡
        trainer.fit(model, train_dataloader, val_dataloader)

        # ================================================
        plot1 = plt.figure()
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Epochs")
        plt.ylabel("Categorical Loss")
        plot_grapic(MAX_EPOCHS, model.train_loss, label="Train loss", type_line='-')
        plot_grapic(MAX_EPOCHS, model.val_loss, label="Validation loss", type_line='--')
        plt.savefig(f"images/loss_{hparams.n_classes}_classes.png")
        plt.close()

        plot2 = plt.figure(2)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coefficient")
        plot_grapic(MAX_EPOCHS, model.train_metric, label="Train metric", type_line='-')
        plot_grapic(MAX_EPOCHS, model.val_metric, label="Validation metric", type_line='--')
        plt.savefig(f"images/metric_{hparams.n_classes}_classes.png")
        plt.close()

        plot3 = plt.figure(3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Epochs")
        plt.ylabel("Average area error")
        plot_grapic(MAX_EPOCHS, model.train_metric2, label="Train metric", type_line='-')
        plot_grapic(MAX_EPOCHS, model.val_metric2, label="Validation metric", type_line='--')
        plt.savefig(f"images/metric2_{hparams.n_classes}_classes.png")
        plt.close()

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