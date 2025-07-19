import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
class GeneratorLossFunction(nn.Module):
    """
    Calculates the total loss for the Generator, combining content loss,
    adversarial loss, and adversarial feature loss.
    """
    def __init__(self, device: str = 'cuda', lambda1: float = 1.0, lambda2: float = 0.5,
                 vgg_layers: list[int] = [2, 7, 16, 25, 34],
                 vgg_weights: list[float] = [1/2, 1/4, 1/8, 1/16, 1/16]) -> None:
        """
        Initializes the GeneratorLossFunction.

        Args:
            device (str): The device (e.g., 'cuda' or 'cpu') where tensors will reside.
            lambda1 (float): Weight for the adversarial loss.
            lambda2 (float): Weight for the adversarial feature loss.
            vgg_layers (list[int]): List of VGG19 layer indices from which to extract features.
            vgg_weights (list[float]): Weights for the MSE loss on VGG features.
        """
        super().__init__()
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # Load pre-trained VGG19 model and use its features
        self.vgg = models.vgg19(pretrained=True).features[:max(vgg_layers)+1].to(device).eval()
        self.vgg_layers = vgg_layers
        self.vgg_weights = vgg_weights

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def vgg_extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extracts features from specified VGG19 layers.
        """
        layers_output = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.vgg_layers:
                layers_output.append(x)
        return layers_output

    def content_loss(self, hr_image: torch.Tensor, sr_image: torch.Tensor, lambda_l1: float = 0.2) -> torch.Tensor:
        """
        Calculates the content loss.
        """
        hr_features = self.vgg_extract(hr_image)
        sr_features = self.vgg_extract(sr_image)

        feature_loss = sum(
            self.vgg_weights[i] * self.mse_loss(sr_features[i], hr_features[i])
            for i in range(len(self.vgg_layers))
        )

        l1_loss = self.l1_loss(hr_image, sr_image)

        return lambda_l1 * l1_loss + feature_loss

    def adversarial_loss(self, discriminator: nn.Module, lr_image: torch.Tensor,
                         hr_image: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        """
        Calculates the adversarial loss for the Generator.
        """
        d_real = discriminator(lr_image, hr_image)[-1]
        d_fake = discriminator(lr_image, sr_image)[-1]
        
        adv_loss = - torch.log(d_real + 1e-8) - torch.log(1 - d_fake + 1e-8)
        return adv_loss.mean()

    def adversarial_feature_loss(self, discriminator: nn.Module, lr_image: torch.Tensor,
                                 hr_image: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        """
        Calculates the adversarial feature loss (feature matching loss).
        """
        d_real_features = discriminator(lr_image, hr_image)[:-1]
        d_fake_features = discriminator(lr_image, sr_image)[:-1]

        adv_feat_loss = sum(
            self.vgg_weights[i] * self.mse_loss(d_real_features[i], d_fake_features[i])
            for i in range(len(d_real_features))
        )
        return adv_feat_loss

    def forward(self, discriminator: nn.Module, lr_image: torch.Tensor,
                hr_image: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        """
        Calculates the total Generator loss.
        """
        total_g_loss = (
            1 * self.content_loss(hr_image, sr_image) +
            0.01 * self.lambda1 * self.adversarial_loss(discriminator, lr_image, hr_image, sr_image) +
            0.0001 * self.lambda2 * self.adversarial_feature_loss(discriminator, lr_image, hr_image, sr_image)
        )
        return total_g_loss