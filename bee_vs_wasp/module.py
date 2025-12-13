import lightning as l
import torch
import torchmetrics
from torch import nn


class BeeLightningModule(l.LightningModule):
    def __init__(
        self, model: nn.Module, lr: float = 0.001, momentum: float = 0.9, num_classes: int = 4
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.num_classes = num_classes

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, target)
        self.train_acc(torch.argmax(logits, 1), target)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, target)
        self.val_acc(torch.argmax(logits, 1), target)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        logits = self(inputs)
        self.test_acc(torch.argmax(logits, 1), target)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
