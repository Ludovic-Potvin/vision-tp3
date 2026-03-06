from torchvision import models
from model_info import ModelInfo

weights = models.ResNet50_Weights.IMAGENET1K_V2

finetuning = ModelInfo(
    weights = weights,
    model = models.resnet50(weights=weights),
    mean = weights.transforms().mean,
    std = weights.transforms().std,
)
