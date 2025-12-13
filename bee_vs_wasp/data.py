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
        self, dataset_root: str, batch_size: int = 32, num_workers: int = 4, num_classes: int = 4
    ):
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes

    def setup(self, stage=None):
        labels_path = self.dataset_root / "labels.csv"
        df = load_bee_dataset_csv(labels_path)
        train_df, val_df, test_df = split_dataframes(df)

        root = self.dataset_root

        self.train_dataset = BeeDataset(train_df, root, train=True, transforms=train_transform)
        self.val_dataset = BeeDataset(val_df, root, train=True, transforms=test_transform)
        self.test_dataset = BeeDataset(test_df, root, train=True, transforms=test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
