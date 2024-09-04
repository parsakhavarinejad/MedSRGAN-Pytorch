import argparse
import glob

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from data.custom_class import CustomDataset
from data.noise_transform import NoiseTransform
from loss.discriminator_loss import DiscriminatorLossFunction
from loss.generator_loss import GeneratorLossFunction
from model.discriminator import Discriminator
from model.generator import Generator
from model.trainer import Trainer

parser = argparse.ArgumentParser(description="Arguments for training")

parser.add_argument('--data', type=str, default='fastmri', help='data to train, either fastmri or mrbrains')
parser.add_argument('--noise', type=str, default='gaussian', help='gaussian or bicubic')
parser.add_argument('--imsize', type=int, default=256, help='image size')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=30, help='epochs')

args = parser.parse_args()

data = glob.glob(f'data/{args.data}/*.jpg')
dataframe = pd.DataFrame(data, columns=['paths'])

image_size = args.imsize

hr_transforms = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
])

lr_transforms = transforms.Compose([
    transforms.Resize([image_size // 4, image_size // 4]),
    transforms.ToTensor(),
    NoiseTransform(mode=args.noise, std=0.02),
])

train_dataset = CustomDataset(dataframe, lr_transforms, hr_transforms)

batch_size = args.batchsize
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

generator = Generator().to('cuda')
discriminator = Discriminator().to('cuda')

gen_loss = GeneratorLossFunction()
dis_loss = DiscriminatorLossFunction()

trainer = Trainer(generator, discriminator, gen_loss, dis_loss, dataloader=dataloader)

losses_g, losses_d, real_scores, fake_scores = trainer.fit(epochs=args.epochs)
