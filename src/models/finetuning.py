import torch

from torchvision import models
import torchvision.transforms as transforms

class FineTuning():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
