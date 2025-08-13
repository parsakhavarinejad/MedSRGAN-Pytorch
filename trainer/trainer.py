import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm import tqdm
import os
import glob
from datetime import datetime

from losses import DiscriminatorLossFunction

class Trainer(nn.Module):
    def __init__(self, generator, discriminator, psnr, ssim, g_loss,
                 train_dataloader, test_dataloader, opt_g, opt_d, device,
                 log_dir='logs', outputs_dir='output_images',
                 model_save_dir='saved_models', save_interval=5):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.psnr = psnr
        self.ssim = ssim
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.g_loss = g_loss
        self.d_loss = DiscriminatorLossFunction(device=device)
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.device = device
        self.start_epoch = 0
        self.save_interval = save_interval

        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(log_dir, now)
        self.image_output_dir = os.path.join(outputs_dir, now)
        self.checkpoint_dir = os.path.join(model_save_dir, now)
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print(f"Output images will be saved to: {self.image_output_dir}")

    def save_samples(self, low_res_images, high_res_images, epoch):
        self.generator.eval()
        low_res_images = low_res_images.to(self.device)
        with torch.no_grad():
            super_res_images = self.generator(low_res_images)
            low_res_images_resized = F.interpolate(low_res_images, size=(256, 256), mode='bilinear', align_corners=False)

        lr_grid = make_grid(low_res_images_resized.cpu(), nrow=4, normalize=True, padding=2)
        hr_grid = make_grid(high_res_images.cpu(), nrow=4, normalize=True, padding=2)
        sr_grid = make_grid(super_res_images.cpu(), nrow=4, normalize=True, padding=2)

        save_path = os.path.join(self.image_output_dir, f'output_epoch_{epoch:04d}.png')
        save_image(torch.cat((lr_grid, hr_grid, sr_grid), 1), save_path)
        self.generator.train()

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            path = os.path.join(self.checkpoint_dir, "best_model.pth")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.opt_g.state_dict(),
            'optimizer_d_state_dict': self.opt_d.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved for epoch {epoch}")

    def load_checkpoint(self, path=None):
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            self.opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from checkpoint. Starting at epoch {self.start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

    def evaluate(self):
        self.generator.eval()
        psnr_scores, ssim_scores = [], []
        with torch.no_grad():
            for lr_image, hr_image in self.test_dataloader:
                lr_image, hr_image = lr_image.to(self.device), hr_image.to(self.device)
                sr_image = self.generator(lr_image)
                psnr_scores.append(self.psnr(sr_image, hr_image).item())
                ssim_scores.append(self.ssim(sr_image, hr_image).item())
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        self.generator.train()
        return avg_psnr, avg_ssim

    def train_one_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        loop = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}", leave=True)
        epoch_g_loss, epoch_d_loss = [], []
        epoch_real_scores, epoch_fake_scores = [], []

        for lr_image, hr_image in loop:
            lr_image, hr_image = lr_image.to(self.device), hr_image.to(self.device)

            # Train Discriminator
            self.opt_d.zero_grad()
            with torch.no_grad():
                sr_image = self.generator(lr_image)
            hr_preds = self.discriminator(lr_image, hr_image)[-1]
            sr_preds = self.discriminator(lr_image, sr_image.detach())[-1]
            loss_d = self.d_loss(hr_preds, sr_preds)
            loss_d.backward()
            self.opt_d.step()

            epoch_d_loss.append(loss_d.item())
            epoch_real_scores.append(hr_preds.mean().item())
            epoch_fake_scores.append(sr_preds.mean().item())
            
            # Train Generator
            self.opt_g.zero_grad()
            sr_image_g = self.generator(lr_image)
            loss_g = self.g_loss(self.discriminator, lr_image, hr_image, sr_image_g)
            loss_g.backward()
            self.opt_g.step()
            epoch_g_loss.append(loss_g.item())

            loop.set_postfix(G_Loss=f"{loss_g.item():.4f}", D_Loss=f"{loss_d.item():.4f}")

        avg_g_loss = np.mean(epoch_g_loss)
        avg_d_loss = np.mean(epoch_d_loss)
        avg_real_score = np.mean(epoch_real_scores)
        avg_fake_score = np.mean(epoch_fake_scores)

        return avg_g_loss, avg_d_loss, avg_real_score, avg_fake_score