import torch
import torch.nn as nn

class DiscriminatorLossFunction(nn.Module):
    """
    Calculates the total loss for the Discriminator with soft labels (0.9 for real, 0.1 for fake).
    Uses nn.BCELoss, appropriate for a discriminator with a Sigmoid output.
    """
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.bce_loss = nn.BCELoss()

    def forward(self, hr_preds: torch.Tensor, sr_preds: torch.Tensor) -> torch.Tensor:
        """
        Calculates the discriminator loss with soft labels.
        """
        # Soft labels: 0.9 for real, 0.1 for fake
        real_targets = torch.full_like(hr_preds, 0.9).to(self.device)
        fake_targets = torch.full_like(sr_preds, 0.1).to(self.device)

        # Loss for real images
        real_loss = self.bce_loss(hr_preds, real_targets)

        # Loss for fake images
        fake_loss = self.bce_loss(sr_preds, fake_targets)

        # Total discriminator loss is the average of real and fake losses
        total_loss = (real_loss + fake_loss) / 2
        return total_loss