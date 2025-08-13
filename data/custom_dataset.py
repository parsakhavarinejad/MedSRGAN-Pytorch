import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Tuple, Optional
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    """
    A custom PyTorch Dataset for loading low-resolution (LR) and high-resolution (HR) image pairs.
    It loads images as single-channel (grayscale).
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 hr_transforms: transforms.Compose,
                 lr_transforms: transforms.Compose,
                 use_dummy: bool = False):
        self.data = dataframe
        self.hr_transforms = hr_transforms
        self.lr_transforms = lr_transforms
        self.use_dummy = use_dummy
        if self.use_dummy:
            self.dummy_image = Image.new('L', (256, 256), color=128)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_dummy:
            image = self.dummy_image
        else:
            image_path = self.data.iloc[idx]['paths']
            try:
                image = Image.open(image_path).convert('L')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Using a black placeholder.")
                image = Image.new('L', (256, 256), color=0)

        hr_tensor = self.hr_transforms(image)
        # Apply LR transforms to the HR tensor to generate the LR input
        lr_tensor = self.lr_transforms(hr_tensor)

        return lr_tensor, hr_tensor