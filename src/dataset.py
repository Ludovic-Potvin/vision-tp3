import torch
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms

from config import DATASET_DISTRIBUTION, DATASET_PATH, SEED

TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_sets():
    """
    Import the dataset and return train, validation and test sets.

    Args:
        transform: Transformation applied to the images.
        target_transform: Transformation applied to the labels.

    Returns:
        tuple: (train_set, validation_set, test_set)
    """
    dataset = torchvision.datasets.ImageFolder(
        root=DATASET_PATH,
        transform=TRANSFORM,
    )

    class_number = len(list(dataset.classes))
    for class_name in list(dataset.classes):
        print(class_name)
    print(f"Class number: {class_number}")

    generator = torch.Generator().manual_seed(SEED)
    train_set, validation_set, test_set = random_split(
        dataset, DATASET_DISTRIBUTION, generator=generator
    )
    
    return train_set, validation_set, test_set
