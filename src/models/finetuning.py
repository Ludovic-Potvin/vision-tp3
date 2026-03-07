from torch import nn
from torchvision import models
from models.model_info import ModelInfo

weights = models.ResNet50_Weights.IMAGENET1K_V2

model = models.resnet50(weights=weights)
model.fc = nn.Linear(2048, 2)

FINETUNING = ModelInfo(
    weights = weights,
    model = model,
    mean = weights.transforms().mean,
    std = weights.transforms().std,
)
