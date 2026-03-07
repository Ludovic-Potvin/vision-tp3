import torch

from training.setup import setup_sets_and_loader
from training.train_process import train_model
from training.test_process import test_model
from models.finetuning import FINETUNING
from models.model_info import ModelInfo

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, model_info: ModelInfo):
    train_loader, validation_loader, _ = setup_sets_and_loader(
        model_info.mean, model_info.std
    )

    output_model = train_model(
        device,
        model,
        train_loader,
        validation_loader,
    )

    return output_model


def test(model, model_info: ModelInfo):
    _, _, test_loader = setup_sets_and_loader(model_info.mean, model_info.std)

    model = test_model(
        device,
        model_info.model,
        test_loader,
    )

    return model



def main():
    model = train(FINETUNING.model, FINETUNING)
    test(model, FINETUNING)

    print(model)
    # train_model(device, FineTuning.model, FineTuning.transform)


if __name__ == "__main__":
    main()
