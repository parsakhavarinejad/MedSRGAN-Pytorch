import matplotlib.pyplot as plt
import numpy as np

def visualize_results(loss_g: list, loss_d: list, real_scores: list,
                      fake_scores: list, psnr_scores: list, ssim_scores: list,
                      save_path: str = 'performance_plots.png'):
    """
    Visualizes the training performance metrics: Generator Loss, Discriminator Loss,
    Real vs Fake Scores, PSNR, and SSIM.

    Args:
        loss_g (list): List of generator losses per epoch.
        loss_d (list): List of discriminator losses per epoch.
        real_scores (list): List of discriminator's real scores per epoch.
        fake_scores (list): List of discriminator's fake scores per epoch.
        psnr_scores (list): List of PSNR scores per epoch.
        ssim_scores (list): List of SSIM scores per epoch.
        save_path (str): Path to save the generated plot.
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Convert scores to numpy arrays if they are still tensors
    # Ensure they are detached from the graph and on CPU before converting to numpy
    real_scores_np = [score.item() if isinstance(score, torch.Tensor) else score for score in real_scores]
    fake_scores_np = [score.item() if isinstance(score, torch.Tensor) else score for score in fake_scores]
    psnr_scores_np = [score.item() if isinstance(score, torch.Tensor) else score for score in psnr_scores]
    ssim_scores_np = [score.item() if isinstance(score, torch.Tensor) else score for score in ssim_scores]


    # Plot Losses
    axs[0].plot(loss_g, label='Generator Loss')
    axs[0].plot(loss_d, label='Discriminator Loss')
    axs[0].set_title('Losses over Training')
    axs[0].set_xlabel('Batch Iterations') # Changed to batch iterations as losses are appended per batch
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Real vs Fake Scores
    axs[1].plot(real_scores_np, label='Real Scores')
    axs[1].plot(fake_scores_np, label='Fake Scores')
    axs[1].set_title('Discriminator Real vs Fake Scores')
    axs[1].set_xlabel('Batch Iterations')
    axs[1].set_ylabel('Score')
    axs[1].legend()
    axs[1].grid(True)

    # Plot PSNR and SSIM
    # These are typically per epoch, so adjust x-axis if needed
    axs[2].plot(psnr_scores_np, label='PSNR')
    axs[2].plot(ssim_scores_np, label='SSIM')
    axs[2].set_title('PSNR and SSIM over Training')
    axs[2].set_xlabel('Batch Iterations') # Or 'Epochs' if you average per epoch
    axs[2].set_ylabel('Metric Value')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"Performance plots saved to {save_path}")
