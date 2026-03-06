from torch import nn
import torch.optim as optim

from training.setup import setup_loader, setup_transform
from training.dataset import get_sets
from training.train_process import train_model

from config import LEARNING_RATE


def exec_training_process(device, model, mean, std):
    transform = setup_transform(mean, std)
    train_set, validation_set, test_set = get_sets(transform)

    # Setup loaders
    train_loader = setup_loader(train_set)
    validation_loader = setup_loader(validation_set, shuffle=False)
    test_loader = setup_loader(test_set)

    # loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.to(device)
    train_model(
        device, model, train_loader, validation_loader, loss_function, optimizer
    )
