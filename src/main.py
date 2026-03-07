import torch
import config

from training.train import exec_training_process, exec_test_process
from models.finetuning import FINETUNING
from models.model_info import ModelInfo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model_info: ModelInfo):
    model = exec_training_process(
        device,
        model=model_info.model,
        mean=model_info.mean,
        std=model_info.std,
    )

    return model


def test_model(model, model_info: ModelInfo):
    model = exec_test_process(
        device,
        model=model_info.model,
        mean=model_info.mean,
        std=model_info.std,
    )

    return model

def main():
    model = train_model(FINETUNING)
    test_model(model, FINETUNING)

    print(model)
    # train_model(device, FineTuning.model, FineTuning.transform)


if __name__ == "__main__":
    main()
