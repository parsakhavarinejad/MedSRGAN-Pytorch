import torch
import matplotlib.pyplot as plt
import os

def visualize_epoch_results(training_results: dict, save_path: str):
    """Visualizes the training performance metrics."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))

    epochs = range(1, len(training_results['loss_g']) + 1)

    # Plot Losses
    axs[0].plot(epochs, training_results['loss_g'], 'b-o', label='Generator Loss')
    axs[0].plot(epochs, training_results['loss_d'], 'r-o', label='Discriminator Loss')
    axs[0].set_title('Training Losses')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Real vs Fake Scores
    axs[1].plot(epochs, training_results['real_scores'], 'g-o', label='Real Scores (D)')
    axs[1].plot(epochs, training_results['fake_scores'], 'm-o', label='Fake Scores (D)')
    axs[1].set_title('Discriminator Scores')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Probability')
    axs[1].legend()
    axs[1].grid(True)

    # Plot PSNR
    axs[2].plot(epochs, training_results['psnr'], 'c-o', label='PSNR (dB)')
    axs[2].set_title('Validation PSNR')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('PSNR (dB)')
    axs[2].legend()
    axs[2].grid(True)

    # Plot SSIM
    axs[3].plot(epochs, training_results['ssim'], 'y-o', label='SSIM')
    axs[3].set_title('Validation SSIM')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('SSIM')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Performance plots saved to {save_path}")