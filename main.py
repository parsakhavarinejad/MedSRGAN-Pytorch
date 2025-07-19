import torch
import pandas as pd
import glob
import os
import warnings
import yaml  # PyYAML library for loading the config file
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from data.custom_dataset import CustomDataset
from models.generator import Generator
from models.discriminator import Discriminator
from losses.generator_loss import GeneratorLossFunction
from losses.discriminator_loss import DiscriminatorLossFunction
from trainer.trainer import Trainer
from utils.transforms import get_transforms
from utils.early_stopping import EarlyStopping
from utils.visualize import visualize_results

# --- Suppress Warnings and Set Environment Variables ---
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def load_config(config_path="config/config.yml"):
    """Loads the YAML configuration file."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}.")
        exit()
    except Exception as e:
        print(f"Error loading or parsing the configuration file: {e}")
        exit()

def main():
    # --- Load Configuration ---
    config = load_config()
    data_cfg = config['data']
    train_cfg = config['training']
    optim_cfg = config['optimizer']
    early_stop_cfg = config['early_stopping']
    saving_cfg = config['saving']

    # --- Determine Device ---
    if train_cfg['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg['device'])
    print(f"Using device: {device}")

    # --- Data Loading and Preparation ---
    print("Loading data...")
    if data_cfg['use_dummy_data']:
        print("Using dummy data for demonstration.")
        dataframe = pd.DataFrame([{'paths': f'dummy_image_{i}.png'} for i in range(data_cfg['dummy_data_size'])])
    else:
        try:
            print("Attempting to load Kaggle dataset...")
            chest_data = glob.glob(data_cfg['train_path']) + glob.glob(data_cfg['test_path'])
            if not chest_data:
                raise FileNotFoundError("No image files found at the specified paths.")
            dataframe = pd.DataFrame(chest_data, columns=['paths'])
            print(f"Found {len(dataframe)} images.")
        except Exception as e:
            print(f"FATAL: Real data loading failed: {e}.")
            print("Please check the paths in 'config/config.yml' or set 'use_dummy_data' to true.")
            return

    lr_transforms, hr_transforms = get_transforms(data_cfg['image_size'])
    
    # Split dataframe
    train_size = int(len(dataframe) * data_cfg['train_split_ratio'])
    train_df = dataframe[:train_size]
    test_df = dataframe[train_size:]

    train_dataset = CustomDataset(train_df, lr_transforms, hr_transforms, use_dummy=data_cfg['use_dummy_data'])
    test_dataset = CustomDataset(test_df, lr_transforms, hr_transforms, use_dummy=data_cfg['use_dummy_data'])

    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=data_cfg['num_workers'])
    print("Data loaded and prepared.")

    # --- Model, Loss, and Metric Initialization ---
    print("Initializing models, losses, and metrics...")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    gen_loss_fn = GeneratorLossFunction(device=device)
    dis_loss_fn = DiscriminatorLossFunction(device=device)
    print("Initialization complete.")

    # --- Trainer Initialization ---
    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        psnr=psnr_metric,
        ssim=ssim_metric,
        g_loss=gen_loss_fn,
        d_loss=dis_loss_fn,
        dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        model_save_dir=saving_cfg['model_save_dir'],
        sample_save_dir=saving_cfg['sample_save_dir']
    )

    # --- Early Stopping Setup ---
    best_model_path = os.path.join(saving_cfg['model_save_dir'], saving_cfg['best_model_filename'])
    early_stopping = EarlyStopping(
        patience=early_stop_cfg['patience'],
        min_delta=early_stop_cfg['min_delta'],
        verbose=True,
        path=best_model_path
    )

    # --- Training Loop ---
    print("Starting training...")
    all_losses_g, all_losses_d, all_real_scores, all_fake_scores, all_psnr_scores, all_ssim_scores = [], [], [], [], [], []

    for epoch in range(train_cfg['epochs']):
        print(f"\nEpoch {epoch + 1}/{train_cfg['epochs']}")
        epoch_loss_g, epoch_loss_d, epoch_real_scores, epoch_fake_scores, epoch_psnr, epoch_ssim = \
            trainer.train_one_epoch(
                epoch=epoch,
                learning_rate=optim_cfg['learning_rate'],
                betas=tuple(optim_cfg['betas']) # Convert list from YAML to tuple
            )

        # Aggregate metrics for visualization
        all_losses_g.extend(epoch_loss_g)
        all_losses_d.extend(epoch_loss_d)
        all_real_scores.extend(epoch_real_scores)
        all_fake_scores.extend(epoch_fake_scores)
        all_psnr_scores.extend(epoch_psnr)
        all_ssim_scores.extend(epoch_ssim)

        # Evaluate on test set
        test_psnr, test_ssim = trainer.evaluate()
        print(f"Test PSNR at epoch {epoch + 1}: {test_psnr:.4f}, Test SSIM: {test_ssim:.4f}")

        # Save sample images
        sample_lr_images, _ = next(iter(test_dataloader))
        trainer.save_samples(sample_lr_images, epoch + 1)

        # Save model checkpoint periodically
        if (epoch + 1) % saving_cfg['save_interval'] == 0:
            trainer.save_model(epoch + 1, empty_dir=False) # Set empty_dir to False to avoid deleting previous checkpoints

        # Check for early stopping
        early_stopping(test_psnr, generator, discriminator)
        if early_stopping.early_stop:
            print("Early stopping triggered. Restoring best model weights.")
            best_model_state = torch.load(best_model_path)
            generator.load_state_dict(best_model_state['generator_state_dict'])
            discriminator.load_state_dict(best_model_state['discriminator_state_dict'])
            break

    print("Training complete.")

    # --- Visualization ---
    print("Generating performance plots...")
    # Convert tensors to numpy for plotting
    plot_psnr_scores = [score.cpu().detach().numpy() if isinstance(score, torch.Tensor) else score for score in all_psnr_scores]
    plot_ssim_scores = [score.cpu().detach().numpy() if isinstance(score, torch.Tensor) else score for score in all_ssim_scores]
    plot_real_scores = [score.cpu().detach().numpy() if isinstance(score, torch.Tensor) else score for score in all_real_scores]
    plot_fake_scores = [score.cpu().detach().numpy() if isinstance(score, torch.Tensor) else score for score in all_fake_scores]

    visualize_results(all_losses_g, all_losses_d, plot_real_scores, plot_fake_scores, plot_psnr_scores, plot_ssim_scores)
    print("Performance plots saved.")

if __name__ == '__main__':
    main()
