import hydra
import lightning as l
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from bee_vs_wasp.data import BeeDataModule
from bee_vs_wasp.model import BeeClassifier
from bee_vs_wasp.module import BeeLightningModule


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig):
    # subprocess.run(["dvc", "pull"], check=True) return import subprocess
    dm = BeeDataModule(
        dataset_root=cfg.data.dataset_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_classes=cfg.data.num_classes,
    )

    model = BeeClassifier(cfg.model.num_classes)
    module = BeeLightningModule(
        model, lr=cfg.model.lr, momentum=cfg.model.momentum, num_classes=cfg.data.num_classes
    )

    logger = TensorBoardLogger("tb_logs", name=cfg.train.logger_name)

    trainer = l.Trainer(
        max_epochs=cfg.num_epochs,
        logger=logger,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
    )

    trainer.fit(module, datamodule=dm)

    torch.save(module.model.state_dict(), cfg.output_file)


if __name__ == "__main__":
    train()
