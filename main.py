# main.py
import torch
import pandas as pd
import glob
import os
import warnings
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

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():
    # --- Configuration ---
    image_size = 256
    batch_size = 10
    learning_rate = 1e-5
    betas = (0.95, 0.999)
    epochs = 50  # Increased epochs for better training
    patience = 10 # Early stopping patience
    min_delta = 0.001 # Minimum change to qualify as an improvement
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Data Loading and Preparation ---
    print("Loading data...")
    # Assuming the data path is correctly set up for Kaggle or local execution
    # For local execution, you might need to adjust this path
    try:
        heart_data = glob.glob('dataset/*.png')
        dataframe = pd.DataFrame(heart_data, columns=['paths'])
    except Exception as e:
        print(f"Kaggle data loading failed: {e}. Using dummy data for demonstration.")
        dataframe = pd.DataFrame([{'paths': f'dummy_image_{i}.png'} for i in range(1000)])

    # Get transforms
    lr_transforms, hr_transforms = get_transforms(image_size)

    # Create datasets
    # In a real scenario, you would split your actual image paths into train/test
    # For this refactor, we will use the dummy image logic for CustomDataset
    # if actual image paths are not available or cause issues.
    if 'dummy_image' in dataframe['paths'].iloc[0]:
        train_dataset = CustomDataset(dataframe[:750], lr_transforms, hr_transforms, use_dummy=True)
        test_dataset = CustomDataset(dataframe[750:], lr_transforms, hr_transforms, use_dummy=True)
    else:
        train_dataset = CustomDataset(dataframe[:750], lr_transforms, hr_transforms, use_dummy=False)
        test_dataset = CustomDataset(dataframe[750:], lr_transforms, hr_transforms, use_dummy=False)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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
        device=device
    )

    # --- Early Stopping Setup ---
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, verbose=True, path=os.path.join(trainer.model_save_dir, 'best_model.pth'))

    # --- Training Loop ---
    print("Starting training...")
    all_losses_g, all_losses_d, all_real_scores, all_fake_scores, all_psnr_scores, all_ssim_scores = [], [], [], [], [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        # Train one epoch
        epoch_loss_g, epoch_loss_d, epoch_real_scores, epoch_fake_scores, epoch_psnr, epoch_ssim = \
            trainer.train_one_epoch(
                epoch=epoch,
                learning_rate=learning_rate,
                betas=betas
            )

        # Append epoch results to overall lists
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
        # Take a batch from the test dataloader for visualization
        sample_lr_images, _ = next(iter(test_dataloader))
        trainer.save_samples(sample_lr_images, epoch + 1)

        # Save models periodically
        if (epoch + 1) % 5 == 0:
            trainer.save_model(epoch + 1)

        # Early stopping check
        early_stopping(test_psnr, generator, discriminator)
        if early_stopping.early_stop:
            print("Early stopping triggered. Restoring best model weights.")
            # Load the best model saved by EarlyStopping
            best_model_state = torch.load(os.path.join(trainer.model_save_dir, 'best_model.pth'))
            generator.load_state_dict(best_model_state['generator_state_dict'])
            discriminator.load_state_dict(best_model_state['discriminator_state_dict'])
            break

    print("Training complete.")

    # --- Visualization ---
    print("Generating performance plots...")
    # Ensure scores are on CPU and detached for plotting
    plot_psnr_scores = [score.cpu().detach().numpy() for score in all_psnr_scores]
    plot_ssim_scores = [score.cpu().detach().numpy() for score in all_ssim_scores]
    plot_real_scores = [score.cpu().detach().numpy() for score in all_real_scores]
    plot_fake_scores = [score.cpu().detach().numpy() for score in all_fake_scores]

    visualize_results(all_losses_g, all_losses_d, plot_real_scores, plot_fake_scores, plot_psnr_scores, plot_ssim_scores)
    print("Performance plots saved.")

if __name__ == '__main__':
    main()
