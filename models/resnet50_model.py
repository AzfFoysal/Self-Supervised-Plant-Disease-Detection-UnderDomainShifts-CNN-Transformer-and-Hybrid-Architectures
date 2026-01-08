import torch
import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    """ResNet50 model with a replaced classification head."""
    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
