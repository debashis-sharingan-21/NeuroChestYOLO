
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DenseNetCustom(nn.Module):
    """
    Modified DenseNet-121 architecture featuring a Dual Pooling Head.
    Concatenates Global Average Pooling (GAP) and Global Max Pooling (GMP)
    yielding a rich 2048-dimensional feature representation.
    Non-inplace ReLU ensures backward gradient hooks work smoothly for Grad-CAM.
    """
    def __init__(self, num_classes=15, pretrained=True):
        super(DenseNetCustom, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)
        self.backbone = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # 1024 * 2 = 2048 dims from concatenating GAP and GMP
        self.classifier = nn.Linear(1024 * 2, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        out = F.relu(features, inplace=False)  # Critical for Grad-CAM activation hooks
        ap = self.avgpool(out).flatten(1)
        mp = self.maxpool(out).flatten(1)
        combined = torch.cat([ap, mp], dim=1)
        return self.classifier(combined)


def get_model(num_classes=15, pretrained=True):
    """Factory function for model instantiation."""
    return DenseNetCustom(num_classes=num_classes, pretrained=pretrained)
