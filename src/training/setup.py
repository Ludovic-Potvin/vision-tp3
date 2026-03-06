import torch
import torchvision.transforms as transforms

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
