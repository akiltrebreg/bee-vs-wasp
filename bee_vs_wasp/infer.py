import subprocess

import hydra
import lightning as l
import torch
from omegaconf import DictConfig

from bee_vs_wasp.data import BeeDataModule
from bee_vs_wasp.model import BeeClassifier
from bee_vs_wasp.module import BeeLightningModule


class InferencePipeline:
    """Inference pipeline for evaluating trained models on test data.

    Handles data loading, model weight loading, and test evaluation
    using PyTorch Lightning Trainer.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize inference pipeline.

        Args:
            cfg: Hydra configuration with model_path, dataset_root,
                 batch_size, num_workers, num_classes, lr, momentum
        """
        self.cfg = cfg

    def _pull_data(self):
        """Pull test data using DVC.

        Ensures latest version of dataset is available before inference.
        """
        subprocess.run(["dvc", "pull"], check=True)

    def run(self):
        """Execute inference pipeline.

        Steps:
        1. Pull data from DVC
        2. Create DataModule for test set
        3. Initialize model and load trained weights
        4. Run evaluation using Lightning Trainer
        """
        self._pull_data()

        dm = BeeDataModule(
            dataset_root=self.cfg.dataset_root,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            num_classes=self.cfg.num_classes,
        )

        model = BeeClassifier(num_classes=self.cfg.num_classes)
        module = BeeLightningModule(
            model=model,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            num_classes=self.cfg.num_classes,
        )

        module.model.load_state_dict(torch.load(self.cfg.model_path))

        trainer = l.Trainer()
        trainer.test(module, datamodule=dm)


@hydra.main(version_base=None, config_path="../conf", config_name="infer_config")
def infer(cfg: DictConfig):
    """Main entry point for inference with Hydra configuration.

    Args:
        cfg: Hydra configuration loaded from infer_config.yaml
    """
    pipeline = InferencePipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    infer()
