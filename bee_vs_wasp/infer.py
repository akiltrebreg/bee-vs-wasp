import subprocess
from dataclasses import dataclass

import lightning as l
import torch

from bee_vs_wasp.data import BeeDataModule
from bee_vs_wasp.model import BeeClassifier
from bee_vs_wasp.module import BeeLightningModule


@dataclass
class InferConfig:
    model_path: str
    dataset_root: str
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 0.001
    momentum: float = 0.9


def infer(cfg: InferConfig):
    subprocess.run(["dvc", "pull"], check=True)

    dm = BeeDataModule(
        dataset_root=cfg.dataset_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = BeeClassifier(num_classes=dm.num_classes)
    module = BeeLightningModule(
        model=model,
        lr=cfg.lr,
        momentum=cfg.momentum,
        num_classes=dm.num_classes,
    )

    module.model.load_state_dict(torch.load(cfg.model_path))

    trainer = l.Trainer()
    trainer.test(module, datamodule=dm)
