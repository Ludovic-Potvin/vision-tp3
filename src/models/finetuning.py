import torch

from torchvision import models
import torchvision.transforms as transforms

class FineTuning():
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    mean = weights.transforms().mean
    std = weights.transforms().std


