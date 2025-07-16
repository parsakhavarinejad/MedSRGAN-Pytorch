import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorLossFunction(nn.Module):
    """
    Calculates the total loss for the Discriminator.
    This typically involves a binary cross-entropy loss for real images (label 1)
    and fake images (label 0).
    """
    def __init__(self, device: str = 'cuda'):
        """
        Initializes the DiscriminatorLossFunction.

        Args:
            device (str): The device (e.g., 'cuda' or 'cpu') where tensors will reside.
        """
        super().__init__()
        self.device = device

    def forward(self, hr_preds: torch.Tensor, sr_preds: torch.Tensor) -> torch.Tensor:
        """
        Calculates the discriminator loss.

        Args:
            hr_preds (torch.Tensor): Discriminator's predictions for real high-resolution images.
            sr_preds (torch.Tensor): Discriminator's predictions for super-resolved (fake) images.

        Returns:
            torch.Tensor: The total discriminator loss.
        """
        # Loss for real images: Discriminator should predict 1 for real images
        # We use F.binary_cross_entropy_with_logits for numerical stability
        # if the input `hr_preds` are raw logits (before sigmoid).
        # However, the Discriminator's `final` layer already applies Sigmoid,
        # so we should use F.binary_cross_entropy directly.
        # Let's assume `hr_preds` and `sr_preds` are already sigmoid outputs (probabilities).
        real_loss = F.binary_cross_entropy(hr_preds, torch.ones_like(hr_preds).to(self.device))

        # Loss for fake images: Discriminator should predict 0 for fake images
        fake_loss = F.binary_cross_entropy(sr_preds, torch.zeros_like(sr_preds).to(self.device))

        # Total discriminator loss is the sum of real and fake losses
        total_loss = real_loss + fake_loss
        return total_loss
