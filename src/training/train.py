from torch import nn
import torch.optim as optim

from training.setup import setup_loader, setup_transform
from training.dataset import get_sets
from training.train_process import train_model
from training.test_process import test_model

from config import LEARNING_RATE


def setup_sets_and_loader(mean, std):
    transform = setup_transform(mean, std)
    train_set, validation_set, test_set = get_sets(transform)

    # Setup loaders
    train_loader = setup_loader(train_set)
    validation_loader = setup_loader(validation_set, shuffle=False)
    test_loader = setup_loader(test_set)

    return train_loader, validation_loader, test_loader


def exec_training_process(device, model, mean, std):
    train_loader, validation_loader, _ = setup_sets_and_loader(mean, std)

    # loss function
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # training
    model.to(device)
    model = train_model(
        device, model, train_loader, validation_loader, loss_function, optimizer
    )
    return model


def exec_test_process(device, model, mean, std):
    _, _, test_loader = setup_sets_and_loader(mean, std)

    model.to(device)
    model = test_model(
        model, device, test_loader
    )
    return model
