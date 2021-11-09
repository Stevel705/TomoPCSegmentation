from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
import torch
import os

from datasets.olfactory_bulb import OlfactoryBulbDataset

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

class OlfactoryBulbModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,)),
        #     ]
        # )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10

    # def prepare_data(self):
    #     # download
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        val_split: float = 0.2
        test_split: float = 0.1
        seed = 0

        olfactory_bulb_dataset = OlfactoryBulbDataset(self.data_dir)
        val_len = round(val_split * len(olfactory_bulb_dataset))
        test_len = round(test_split * len(olfactory_bulb_dataset))
        train_len = len(olfactory_bulb_dataset) - val_len - test_len

        trainset, valset, testset = random_split(
                olfactory_bulb_dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed)
        )

        if stage == "fit" or stage is None:
            self.trainset, self.valset = trainset, valset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.testset = testset

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)