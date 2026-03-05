from loaders import setup_loader
from dataset import get_sets

def train_model(device, model, transform):
    train_set, validation_set, test_set = get_sets(transform)

    # loader
    train_loader = setup_loader(train_set)
    validation_loader = setup_loader(validation_set, shuffle=False)
    test_loader = setup_loader(test_set)
    
    # training
    model.to(device)
    