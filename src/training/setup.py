import torch
import torchvision.transforms as transforms
from training.dataset import get_sets

from config import BATCH_SIZE


def setup_loader(images_set, *, shuffle=True):
    """Setup a single loader for a single image set.

    Args:
        images_set (set): the image set to setup
        shuffle (bool, optional): If the loader shuffle the set. Defaults to True.

    Returns:
        loader: A loader for a certain set
    """
    image_loader = torch.utils.data.DataLoader(
        images_set, batch_size=BATCH_SIZE, shuffle=shuffle
    )

    return image_loader


def setup_transform(mean, std):
    """Setup a clear transform for models, with the only key difference being 
    the mean and std to adapt it to pre-train models.

    Args:
        mean (_type_): The normalisation mean
        std (_type_): The normalisation std

    Returns:
        transforms.Compose: a transform compose to make the data ready for 
        each models
    """
    transform = transforms.Compose(
        [
            transforms.Resize((232, 232)),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transform


def setup_sets_and_loader(mean, std):
    transform = setup_transform(mean, std)
    train_set, validation_set, test_set = get_sets(transform)

    # Setup loaders
    train_loader = setup_loader(train_set)
    validation_loader = setup_loader(validation_set, shuffle=False)
    test_loader = setup_loader(test_set)

    return train_loader, validation_loader, test_loader