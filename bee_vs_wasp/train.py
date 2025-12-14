import subprocess
from pathlib import Path

import git
import hydra
import lightning as l
import torch
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from bee_vs_wasp.data import BeeDataModule
from bee_vs_wasp.model import BeeClassifier, SimpleCNN
from bee_vs_wasp.module import BeeLightningModule


class TrainingPipeline:
    """Training pipeline with MLflow tracking."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.git_commit_id = self._get_git_commit()

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            repo = git.Repo(search_parent_directories=True)
            return repo.head.object.hexsha
        except Exception:
            return "unknown"

    def _pull_data(self):
        """Pull data using DVC."""
        subprocess.run(["dvc", "pull"], check=True)

    def _create_datamodule(self) -> BeeDataModule:
        """Create Lightning DataModule."""
        return BeeDataModule(
            dataset_root=self.cfg.data.dataset_root,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            num_classes=self.cfg.data.num_classes,
        )

    def _create_model(self):
        """Create model based on configuration."""
        if self.cfg.model.architecture == "resnet18":
            return BeeClassifier(self.cfg.model.num_classes)
        if self.cfg.model.architecture == "simple_cnn":
            return SimpleCNN(self.cfg.model.num_classes)
        raise ValueError(f"Unknown architecture: {self.cfg.model.architecture}")

    def _create_logger(self) -> MLFlowLogger:
        """Create MLflow logger."""
        return MLFlowLogger(
            experiment_name=self.cfg.mlflow.experiment_name,
            tracking_uri=self.cfg.mlflow.tracking_uri,
            run_name=self.cfg.train.logger_name,
        )

    def _log_hyperparameters(self, logger: MLFlowLogger):
        """Log hyperparameters and git commit to MLflow."""
        params = {
            "git_commit_id": self.git_commit_id,
            "architecture": self.cfg.model.architecture,
            "num_classes": self.cfg.model.num_classes,
            "learning_rate": self.cfg.model.lr,
            "momentum": self.cfg.model.momentum,
            "batch_size": self.cfg.data.batch_size,
            "num_epochs": self.cfg.num_epochs,
            "num_workers": self.cfg.data.num_workers,
        }
        logger.log_hyperparams(params)

    def run(self):
        """Execute training pipeline."""
        self._pull_data()

        dm = self._create_datamodule()
        model = self._create_model()

        module = BeeLightningModule(
            model,
            lr=self.cfg.model.lr,
            momentum=self.cfg.model.momentum,
            num_classes=self.cfg.data.num_classes,
        )

        logger = self._create_logger()
        self._log_hyperparameters(logger)

        trainer = l.Trainer(
            max_epochs=self.cfg.num_epochs,
            logger=logger,
            accelerator=self.cfg.train.accelerator,
            devices=self.cfg.train.devices,
        )

        trainer.fit(module, datamodule=dm)

        output_path = Path(self.cfg.output_file)
        torch.save(module.model.state_dict(), output_path)

        # Log model artifact to MLflow
        logger.experiment.log_artifact(logger.run_id, str(output_path))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    pipeline = TrainingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    train()
