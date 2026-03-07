import torch
from torch.utils.data import random_split

import torchvision

from config import DATASET_DISTRIBUTION, DATASET_PATH, SEED


def get_sets(transform):
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
        transform=transform,
    )

    generator = torch.Generator().manual_seed(SEED)
    train_set, validation_set, test_set = random_split(
        dataset, DATASET_DISTRIBUTION, generator=generator
    )

    return train_set, validation_set, test_set
