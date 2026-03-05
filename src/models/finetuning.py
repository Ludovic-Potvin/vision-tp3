import torch

from torchvision import models
import torchvision.transforms as transforms

from dataset import get_sets
from loaders import setup_loader


class FineTuning():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    transform = transforms.Compose([
        transforms.Resize((232, 232)),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def train_ft_model(device):
    # dataset
    train_set, validation_set, test_set = get_sets(transform)

    # loader
    train_loader = setup_loader(train_set)
    validation_loader = setup_loader(validation_set, shuffle=False)
    test_loader = setup_loader(test_set)
    
    # training
    ftmodel = 
    
    # L
	pass
