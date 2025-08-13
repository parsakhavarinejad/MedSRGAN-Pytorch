import torch
import torchvision.transforms as transforms
from typing import Tuple

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def get_data_pipelines(image_size: int = 256) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Returns augmentation, HR, and LR transformation pipelines."""
    
    # Augmentations to be applied to the PIL image before tensor conversion
    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ])

    # Transforms for the high-resolution image
    hr_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Transforms to create the low-resolution image from the HR tensor
    lr_transforms = transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.Resize((image_size // 4, image_size // 4)),
        AddGaussianNoise(mean=0.0, std=0.01),
    ])

    return augmentations, hr_transforms, lr_transforms