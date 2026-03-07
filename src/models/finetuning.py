import os
from datetime import datetime

from torch import nn
import torch
from torchvision import models
from models.model_info import ModelInfo

from config import RESULT_PATH

weights = models.ResNet50_Weights.IMAGENET1K_V2

model = models.resnet50(weights=weights)
model.fc = nn.Linear(2048, 2)


def save(model, timestamp, epoch):
    os.makedirs(RESULT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"resnet50_{timestamp}_epoch{epoch + 1}.pth"
    torch.save(model.state_dict(), os.path.join(RESULT_PATH, file_name))


def load(file_name):
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(os.path.join(RESULT_PATH, file_name)))
    return model


FINETUNING = ModelInfo(
    weights=weights,
    model=model,
    mean=weights.transforms().mean,
    std=weights.transforms().std,
    save=save,
    load=load,
)
