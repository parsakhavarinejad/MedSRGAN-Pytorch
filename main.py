import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import pandas as pd
import glob
import os
import yaml
from sklearn.model_selection import train_test_split

# Import from project modules
from data.custom_dataset import CustomDataset
from utils.transforms import get_data_pipelines
from models.generator import Generator
from models.discriminator import Discriminator
from losses.generator_loss import GeneratorLossFunction
from trainer.trainer import Trainer
from utils.early_stopping import EarlyStopping
from utils.visualize import visualize_epoch_results

def main():
    # --- Load Configuration ---
    with open('config\config.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    # --- Device Configuration ---
    if cfg['training']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['training']['device'])
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading data...")
    if not cfg['data']['use_dummy_data']:
        try:
            train_paths = glob.glob(cfg['data']['train_path'])
            test_paths = glob.glob(cfg['data']['test_path'])
            all_paths = train_paths + test_paths
            if not all_paths:
                raise FileNotFoundError("No image files found in the specified paths.")
            dataframe = pd.DataFrame(all_paths, columns=['paths'])
        except FileNotFoundError as e:
            print(f"Error: {e} Falling back to dummy data.")
            cfg['data']['use_dummy_data'] = True

    if cfg['data']['use_dummy_data']:
        print("Using dummy data for demonstration.")
        dataframe = pd.DataFrame([{'paths': f'dummy_{i}.png'} for i in range(cfg['data']['dummy_data_size'])])

    train_df, test_df = train_test_split(dataframe, train_size=cfg['data']['train_split_ratio'], random_state=42)

    augmentations, hr_transforms, lr_transforms = get_data_pipelines(image_size=cfg['data']['image_size'])
    
    train_dataset = CustomDataset(
        dataframe=train_df,
        hr_transforms=hr_transforms,
        lr_transforms=lr_transforms,
        use_dummy=cfg['data']['use_dummy_data']
    )
    test_dataset = CustomDataset(
        dataframe=test_df,
        hr_transforms=hr_transforms,
        lr_transforms=lr_transforms,
        use_dummy=cfg['data']['use_dummy_data']
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    print(f"Data loaded: {len(train_dataset)} training images, {len(test_dataset)} testing images.")

    # --- Model, Loss, Optimizer Initialization ---
    print("Initializing models and optimizers...")
    generator = Generator(in_channels=1, out_channels=1).to(device)
    discriminator = Discriminator().to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    gen_loss_fn = GeneratorLossFunction(device=device)
    betas_as_floats = tuple(map(float, cfg['optimizer']['betas']))
    opt_g = optim.Adam(generator.parameters(), lr=cfg['optimizer']['learning_rate_g'], betas=betas_as_floats, weight_decay=cfg['optimizer']['weight_decay'])
    opt_d = optim.Adam(discriminator.parameters(), lr=cfg['optimizer']['learning_rate_d'], betas=betas_as_floats, weight_decay=cfg['optimizer']['weight_decay'])
    print("Initialization complete.")

    # --- Trainer Initialization ---
    trainer = Trainer(
        generator, discriminator, psnr_metric, ssim_metric, gen_loss_fn,
        train_dataloader, test_dataloader, opt_g, opt_d, device,
        model_save_dir=cfg['saving']['model_save_dir'],
        outputs_dir=cfg['saving']['sample_save_dir'],
        save_interval=cfg['saving']['save_interval']
    )
    # trainer.load_checkpoint(...) # Optional: specify a path to resume

    # --- Early Stopping Setup ---
    early_stopping = EarlyStopping(
        patience=cfg['early_stopping']['patience'],
        min_delta=cfg['early_stopping']['min_delta'],
        verbose=True,
        path=os.path.join(trainer.checkpoint_dir, cfg['saving']['best_model_filename'])
    )

    # --- Training Loop & Results Collection ---
    print(f"Starting training for {cfg['training']['epochs']} epochs...")
    training_results = {'loss_g': [], 'loss_d': [], 'real_scores': [], 'fake_scores': [], 'psnr': [], 'ssim': []}

    for epoch in range(trainer.start_epoch, cfg['training']['epochs']):
        avg_g_loss, avg_d_loss, avg_real_score, avg_fake_score = trainer.train_one_epoch(epoch)
        val_psnr, val_ssim = trainer.evaluate()

        # Store results
        training_results['loss_g'].append(avg_g_loss)
        training_results['loss_d'].append(avg_d_loss)
        training_results['real_scores'].append(avg_real_score)
        training_results['fake_scores'].append(avg_fake_score)
        training_results['psnr'].append(val_psnr)
        training_results['ssim'].append(val_ssim)

        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} -> G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f} | Val_PSNR: {val_psnr:.4f} dB, Val_SSIM: {val_ssim:.4f}")

        # Save samples and checkpoint
        sample_lr, sample_hr = next(iter(test_dataloader))
        trainer.save_samples(sample_lr[:4], sample_hr[:4], epoch + 1)
        
        if (epoch + 1) % trainer.save_interval == 0:
            trainer.save_checkpoint(epoch)

        early_stopping(val_psnr, generator, discriminator, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("Training complete.")
    plot_save_path = os.path.join(trainer.log_dir, 'performance_plots.png')
    visualize_epoch_results(training_results, plot_save_path)

if __name__ == '__main__':
    main()