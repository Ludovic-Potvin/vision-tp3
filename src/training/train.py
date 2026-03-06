from torch import nn
import torch.optim as optim

from training.setup import setup_loader, setup_transform
from training.dataset import get_sets

from config import LEARNING_RATE

def exec_training_process(device, model_info):
    transform = setup_transform(model_info.mean, model_info.std)
    train_set, validation_set, test_set = get_sets(transform)

    # Setup loaders
    train_loader = setup_loader(train_set)
    validation_loader = setup_loader(validation_set, shuffle=False)
    test_loader = setup_loader(test_set)
    
    # loss function
    loss_function = nn.CrossEntropyLoss(weight=model_info.weights)
    optimizer = optim.SGD(model_info.model.parameters(), lr=LEARNING_RATE)
    
    model_info.model.to(device)

def train_model(model):
    pass