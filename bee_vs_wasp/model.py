from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Simple convolutional neural network for image classification.

    Architecture:
    - 4 convolutional blocks (Conv2d -> ReLU -> BatchNorm -> MaxPool)
    - 2 fully connected layers with dropout
    - Input: 224x224x3 images
    - Output: logits for num_classes
    """

    def __init__(self, num_classes: int):
        """Initialize SimpleCNN model.

        Args:
            num_classes: Number of output classes for classification
        """
        super().__init__()

        # Convolutional feature extractor: 4 blocks
        # Input: 224x224x3 -> Output: 14x14x256
        self.conv_layers = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )

        # Classification head
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 14x14x256 -> 50176
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(512, num_classes),
        )

    def forward(self, images):
        """Forward pass through the network.

        Args:
            images: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        features = self.conv_layers(images)
        return self.fc_layers(features)


class BeeClassifier(nn.Module):
    """Transfer learning classifier using pretrained ResNet18.

    Uses ImageNet pretrained weights with modified final layer
    for bee vs wasp classification task.
    """

    def __init__(self, num_classes: int):
        """Initialize BeeClassifier with pretrained ResNet18.

        Args:
            num_classes: Number of output classes for classification
        """
        super().__init__()
        # Load pretrained ResNet18 with ImageNet weights
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace final fully connected layer for custom classification
        # ResNet18 fc input: 512 features
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, images):
        """Forward pass through ResNet18.

        Args:
            images: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.model(images)
