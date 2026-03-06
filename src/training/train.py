from training.loaders import setup_loader
from training.dataset import get_sets

def exec_training_process(device, model, transform):
    transform = setup_transform(mean, std)
    train_set, validation_set, test_set = get_sets(transform)

    # Setup loaders
    train_loader = setup_loader(train_set)
    validation_loader = setup_loader(validation_set, shuffle=False)
    test_loader = setup_loader(test_set)
    
    # loss function
    
    model.to(device)

def train_model(model):
    pass