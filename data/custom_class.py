from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):

    def __init__(self, dataframe: list, lr_transforms, hr_transforms):
        self.data = dataframe
        self.lr_transforms = lr_transforms
        self.hr_transforms = hr_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['paths']

        hr_images = Image.open(image_path).convert('RGB')

        lr_image = self.lr_transforms(hr_images)
        hr_image = self.hr_transforms(hr_images)

        return lr_image, hr_image