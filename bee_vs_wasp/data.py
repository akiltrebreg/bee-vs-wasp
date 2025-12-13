from pathlib import Path

import lightning as l
from torch.utils.data import DataLoader

from .preprocessing import (
    BeeDataset,
    load_bee_dataset_csv,
    split_dataframes,
    test_transform,
    train_transform,
)


class BeeDataModule(l.LightningDataModule):
    def __init__(
        self, dataset_root: str, labels_path: str, imgdir: str, batch_size: int, num_workers: int
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.labels_path = Path(labels_path)
        self.imgdir = Path(imgdir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        df = load_bee_dataset_csv(self.labels_path)
        train_df, val_df, test_df = split_dataframes(df)

        root = self.imgdir

        self.train_dataset = BeeDataset(train_df, root, train=True, transforms=train_transform)
        self.val_dataset = BeeDataset(val_df, root, train=True, transforms=test_transform)
        self.test_dataset = BeeDataset(test_df, root, train=True, transforms=test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, True, self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, False, self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.batch_size, False, self.num_workers)
