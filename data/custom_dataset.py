
# data/custom_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading low-resolution (LR) and high-resolution (HR) image pairs.
    It can either load actual images from paths or generate dummy images for testing/demonstration.
    """
    def __init__(self, dataframe: pd.DataFrame, lr_transforms, hr_transforms, use_dummy: bool = False):
        """
        Initializes the CustomDataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image paths.
            lr_transforms: Transformations to apply to low-resolution images.
            hr_transforms: Transformations to apply to high-resolution images.
            use_dummy (bool): If True, generates dummy images instead of loading from paths.
                              Useful for testing when actual data might not be available.
        """
        self.data = dataframe
        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms
        self.use_dummy = use_dummy
        if self.use_dummy:
            # Create a dummy image if not loading from file
            self.dummy_image = Image.new('RGB', (256, 256), color = 'red') # Example size and color

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the LR and HR image pair at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the low-resolution (LR) image and high-resolution (HR) image.
        """
        if self.use_dummy:
            # Generate dummy images
            image = self.dummy_image
            # Apply transforms with a fixed seed for reproducibility of dummy images
            torch.manual_seed(42 + idx) # Use idx to vary dummy images slightly
            lr_image = self.lr_transforms(image)
            torch.manual_seed(42 + idx)
            hr_image = self.hr_transforms(image)
        else:
            # Load actual images from disk
            image_path = self.data.iloc[idx]['paths']
            try:
                hr_images = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping this image.")
                # Fallback for corrupted images: return a black image or handle as appropriate
                hr_images = Image.new('RGB', (256, 256), color = 'black') # Placeholder for corrupted image

            # Apply transforms with a fixed seed for reproducibility of transformations
            torch.manual_seed(42)
            lr_image = self.lr_transforms(hr_images)

            torch.manual_seed(42)
            hr_image = self.hr_transforms(hr_images)

        return lr_image, hr_image
