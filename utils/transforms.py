```python
# utils/transforms.py
import torch
import torchvision.transforms as transforms

class AddGaussianNoise:
    """
    Custom transform to add Gaussian noise to a tensor.
    """
    def __init__(self, mean=0.0, std=1.0):
        """
        Initializes the AddGaussianNoise transform.

        Args:
            mean (float): Mean of the Gaussian noise. Defaults to 0.0.
            std (float): Standard deviation of the Gaussian noise. Defaults to 1.0.
        """
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies Gaussian noise to the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added Gaussian noise.
        """
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self) -> str:
        """
        Returns a string representation of the transform.
        """
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

def get_transforms(image_size: int = 256):
    """
    Returns a tuple of torchvision transforms for high-resolution (HR) and
    low-resolution (LR) images.

    Args:
        image_size (int): The target size for HR images. LR images will be
                          downscaled by a factor of 4 from this size. Defaults to 256.

    Returns:
        tuple: (lr_transforms, hr_transforms)
            lr_transforms: Compose of transforms for low-resolution images.
            hr_transforms: Compose of transforms for high-resolution images.
    """
    hr_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    lr_transforms = transforms.Compose([
        transforms.Resize(image_size), # Resize to HR size first for consistent cropping/rotation
        transforms.RandomCrop(image_size),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.0001, 5.0)), # kernel_size must be odd
        AddGaussianNoise(mean=0.001, std=0.03),
        transforms.Resize([image_size // 4, image_size // 4]), # Downscale to LR size
    ])

    return lr_transforms, hr_transforms
