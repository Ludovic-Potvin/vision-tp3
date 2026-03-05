import torch
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
