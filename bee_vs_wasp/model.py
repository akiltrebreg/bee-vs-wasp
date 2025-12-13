from torch import nn
from torchvision import models


class BeeClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
