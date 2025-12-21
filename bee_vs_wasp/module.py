import lightning as l
import torch
import torchmetrics
from torch import nn


class BeeLightningModule(l.LightningModule):
    """Lightning module wrapper for classification models.

    Handles training loop, validation, testing and metric tracking.
    Logs train_loss, train_acc, train_f1, val_loss, val_acc, val_f1
    metrics to experiment tracker.
    """

    def __init__(
        self, model: nn.Module, lr: float = 0.001, momentum: float = 0.9, num_classes: int = 4
    ):
        """Initialize Lightning module.

        Args:
            model: PyTorch model to train (ResNet18 or SimpleCNN)
            lr: Learning rate for SGD optimizer
            momentum: Momentum for SGD optimizer
            num_classes: Number of classification classes
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.num_classes = num_classes

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Accuracy metrics for each stage
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # F1-score (macro) metrics for each stage
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )

    def forward(self, inputs):
        """Forward pass through the model.

        Args:
            inputs: Batch of input images

        Returns:
            Model logits for each class
        """
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        """Training step for single batch.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch

        Returns:
            Training loss value
        """
        inputs, target = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, target)

        # Get predictions
        preds = torch.argmax(logits, 1)

        # Update metrics
        self.train_acc(preds, target)
        self.train_f1(preds, target)

        # Log metrics to experiment tracker
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        self.log("train_f1", self.train_f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for single batch.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        inputs, target = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, target)

        # Get predictions
        preds = torch.argmax(logits, 1)

        # Update metrics
        self.val_acc(preds, target)
        self.val_f1(preds, target)

        # Log metrics to experiment tracker
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step for single batch.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of current batch
        """
        inputs, target = batch
        logits = self(inputs)

        # Get predictions
        preds = torch.argmax(logits, 1)

        # Update metrics
        self.test_acc(preds, target)
        self.test_f1(preds, target)

        # Log metrics to experiment tracker
        self.log("test_acc", self.test_acc, prog_bar=True)
        self.log("test_f1", self.test_f1, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer for training.

        Returns:
            SGD optimizer with configured learning rate and momentum
        """
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
