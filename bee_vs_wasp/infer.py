import subprocess

import lightning as l
import torch

from bee_vs_wasp.data import BeeDataModule
from bee_vs_wasp.model import BeeClassifier
from bee_vs_wasp.module import BeeLightningModule


def infer(
    model_path: str,
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    lr: float = 0.001,
):
    subprocess.run(["dvc", "pull"], check=True)

    dm = BeeDataModule(dataset_root=dataset_root, batch_size=batch_size, num_workers=num_workers)

    model = BeeClassifier(num_classes=dm.num_classes)
    module = BeeLightningModule(model, lr=lr, num_classes=dm.num_classes)
    module.model.load_state_dict(torch.load(model_path))

    trainer = l.Trainer()
    trainer.test(module, datamodule=dm)
