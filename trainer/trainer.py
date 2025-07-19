import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from datetime import datetime
import os
import tqdm
import matplotlib.pyplot as plt
class Trainer(nn.Module):
    """
    A comprehensive trainer class for the MEDSRGAN model.
    """
    def __init__(self, generator: nn.Module, discriminator: nn.Module,
                 psnr: nn.Module, ssim: nn.Module,
                 g_loss: nn.Module, d_loss: nn.Module,
                 dataloader: DataLoader, test_dataloader: DataLoader,
                 device: str = 'cuda',
                 model_logs_dir: str = 'model_logs',
                 outputs_dir: str = 'output_images') -> None:
        """
        Initializes the Trainer.
        """
        super().__init__()

        now = datetime.now()
        output_name = f'Month{now.month}_Day{now.day}_Hour{now.hour}_Min{now.minute}'
        self.model_save_dir = os.path.join(model_logs_dir, output_name)
        self.image_output_dir = os.path.join(outputs_dir, output_name)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)

        self.generator = generator
        self.discriminator = discriminator
        self.psnr = psnr
        self.ssim = ssim
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.device = device

        self.psnr_scores = []
        self.ssim_scores = []
        self.loss_g = []
        self.loss_d = []
        self.real_score = []
        self.fake_score = []

    def save_samples(self, low_res_images: torch.Tensor, epoch: int = 0) -> None:
        """
        Generates super-resolved images from a batch of low-resolution images
        and saves them along with the original LR images.
        """
        self.generator.eval()
        low_res_images = low_res_images.to(self.device)

        with torch.no_grad():
            super_res_images = self.generator(low_res_images)

        lr_display = low_res_images[0].cpu().permute(1, 2, 0).numpy()
        sr_display = super_res_images[0].cpu().permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(lr_display)
        ax[0].set_title('Low-Resolution Input')
        ax[0].axis('off')

        ax[1].imshow(sr_display)
        ax[1].set_title('Super-Resolved Output')
        ax[1].axis('off')

        plt.suptitle(f"Epoch {epoch} Sample")
        plt.tight_layout()

        save_path = os.path.join(self.image_output_dir, f'output_image_epoch_{epoch:04d}.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close(fig)
        print(f'Sample images saved for Epoch: {epoch:04d} to {save_path}')

        self.generator.train()

    def save_model(self, epoch: int, empty_dir: bool = False) -> None:
        """
        Saves the state dictionaries of the Generator and Discriminator models.
        """
        gen_path = os.path.join(self.model_save_dir, f'generator_epoch_{epoch:04d}.pth')
        disc_path = os.path.join(self.model_save_dir, f'discriminator_epoch_{epoch:04d}.pth')

        if empty_dir:
            for item in os.listdir(self.model_save_dir):
                item_path = os.path.join(self.model_save_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            
        torch.save(self.generator.state_dict(), gen_path)
        torch.save(self.discriminator.state_dict(), disc_path)

        print(f"Models saved at epoch {epoch} to {self.model_save_dir}")

    def evaluate(self) -> tuple[float, float]:
        """
        Evaluates the Generator's performance on the test dataset.
        """
        self.generator.eval()
        psnr_scores = []
        ssim_scores = []

        with torch.no_grad():
            for lr_image, hr_image in tqdm.tqdm(self.test_dataloader, desc="Evaluating"):
                lr_image = lr_image.to(self.device)
                hr_image = hr_image.to(self.device)

                sr_image = self.generator(lr_image)

                psnr_score = self.psnr(sr_image, hr_image)
                ssim_score = self.ssim(sr_image, hr_image)

                psnr_scores.append(psnr_score.item())
                ssim_scores.append(ssim_score.item())

        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_ssim = sum(ssim_scores) / len(ssim_scores)

        print(f"Validation PSNR: {avg_psnr:.4f}, Validation SSIM: {avg_ssim:.4f}")
        self.generator.train()
        return avg_psnr, avg_ssim

    def train_discriminator(self, lr_image: torch.Tensor, hr_image: torch.Tensor,
                            opt_d: optim.Adam) -> tuple[float, float, float]:
        """
        Performs one training step for the Discriminator.
        """
        opt_d.zero_grad()

        hr_preds = self.discriminator(lr_image, hr_image)[-1]
        hr_targets = torch.ones(hr_image.size(0), 1, device=self.device)
        
        real_loss = self.d_loss(hr_preds, hr_targets)
        real_score = hr_preds.mean()

        sr_image = self.generator(lr_image).detach()
        sr_preds = self.discriminator(lr_image, sr_image)[-1]
        sr_targets = torch.zeros(sr_preds.size(0), 1, device=self.device)
        fake_loss = self.d_loss(sr_preds, sr_targets)
        fake_score = sr_preds.mean()

        loss_d = real_loss + fake_loss
        loss_d.backward()
        opt_d.step()

        return loss_d.item(), real_score.item(), fake_score.item()

    def train_generator(self, lr_image: torch.Tensor, hr_image: torch.Tensor,
                        opt_g: optim.Adam) -> tuple[float, float, float]:
        """
        Performs one training step for the Generator.
        """
        opt_g.zero_grad()

        sr_image = self.generator(lr_image)

        loss_g = self.g_loss(self.discriminator, lr_image, hr_image, sr_image)

        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        opt_g.step()

        with torch.no_grad():
            psnr_score = self.psnr(sr_image, hr_image)
            ssim_score = self.ssim(sr_image, hr_image)

        return loss_g.item(), psnr_score.item(), ssim_score.item()

    def train_one_epoch(self, epoch: int, learning_rate: float, betas: tuple[float, float]) -> tuple:
        """
        Trains the Generator and Discriminator for one epoch.
        """
        self.generator.train()
        self.discriminator.train()

        opt_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=betas)
        opt_g = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=betas)

        epoch_losses_g = []
        epoch_losses_d = []
        epoch_real_scores = []
        epoch_fake_scores = []
        epoch_psnr = []
        epoch_ssim = []

        for batch_idx, (lr_image, hr_image) in enumerate(tqdm.tqdm(self.dataloader, desc=f"Epoch {epoch+1}")):
            lr_image = lr_image.to(self.device)
            hr_image = hr_image.to(self.device)

            loss_d, real_score, fake_score = self.train_discriminator(lr_image, hr_image, opt_d)
            epoch_losses_d.append(loss_d)
            epoch_real_scores.append(real_score)
            epoch_fake_scores.append(fake_score)

            loss_g, psnr_score, ssim_score = self.train_generator(lr_image, hr_image, opt_g)
            epoch_losses_g.append(loss_g)
            epoch_psnr.append(psnr_score)
            epoch_ssim.append(ssim_score)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(self.dataloader)}], "
                      f"G Loss: {loss_g:.4f}, D Loss: {loss_d:.4f}, "
                      f"Real Score: {real_score:.4f}, Fake Score: {fake_score:.4f}, "
                      f"PSNR: {psnr_score:.4f}, SSIM: {ssim_score:.4f}")

        self.loss_g.append(sum(epoch_losses_g) / len(epoch_losses_g))
        self.loss_d.append(sum(epoch_losses_d) / len(epoch_losses_d))
        self.real_score.append(sum(epoch_real_scores) / len(epoch_real_scores))
        self.fake_score.append(sum(epoch_fake_scores) / len(epoch_fake_scores))
        self.psnr_scores.append(sum(epoch_psnr) / len(epoch_psnr))
        self.ssim_scores.append(sum(epoch_ssim) / len(epoch_ssim))

        print(f"Epoch [{epoch + 1}], Avg G Loss: {self.loss_g[-1]:.4f}, Avg D Loss: {self.loss_d[-1]:.4f}, "
              f"Avg Real Score: {self.real_score[-1]:.4f}, Avg Fake Score: {self.fake_score[-1]:.4f}, "
              f"Avg PSNR: {self.psnr_scores[-1]:.4f}, Avg SSIM: {self.ssim_scores[-1]:.4f}")

        return epoch_losses_g, epoch_losses_d, epoch_real_scores, epoch_fake_scores, epoch_psnr, epoch_ssim