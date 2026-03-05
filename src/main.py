import torch

from training.train import train_model
from models.finetuning import FineTuning

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    # Finetuning
    train_model(device, FineTuning.model, FineTuning.transform)

if __name__ == "__main__":
    main()
