import os
from datetime import datetime

import torch

from config import RESULT_PATH
from model_info import ModelInfo
from cnn_architecture import CNNClassifier


model = CNNClassifier()


def save(model, timestamp, epoch):
    os.makedirs(RESULT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"cnn_{timestamp}_epoch{epoch + 1}.pth"
    torch.save(model.state_dict(), os.path.join(RESULT_PATH, file_name))


def load(file_name):
    model = CNNClassifier()
    model.load_state_dict(torch.load(os.path.join(RESULT_PATH, file_name)))
    return model


FINETUNING = ModelInfo(
    weights=None,
    model=model,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    save=save,
    load=load,
)
