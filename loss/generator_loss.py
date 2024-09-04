import torch
from torch import nn
from torchvision import models


class GeneratorLossFunction(nn.Module):
    def __init__(self, device: str = 'cuda', lambda1: float = 4e-2, lambda2: float = 1e-4,
                 vgg_layers: list[int] = [2, 7, 16, 25, 34],
                 weights: list[float] = [1 / 2, 1 / 2, 1 / 4, 1 / 32, 1 / 64]) -> None:

        super().__init__()

        self.vgg = models.vgg19(pretrained=True).features.to(device).eval()

        self.layers = vgg_layers
        self.weights = weights
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, discriminator, LR, HR, SR):
        return self.content_loss(HR, SR) + self.lambda1 * self.adversarial_loss(discriminator, LR, HR,
                                                                                SR) + self.lambda2 * self.adversarial_feature_loss(
            discriminator, LR, HR, SR)

    def vgg_extract(self, x):
        layers_output = []
        for layer in self.vgg:
            x = layer(x)
            layers_output.append(x)

        features = [layers_output[layer] for layer in self.layers]
        return features

    def content_loss(self, HR, SR, lambda_l1=0.2):
        HR_features = self.vgg_extract(HR)
        SR_features = self.vgg_extract(SR)

        loss = 0.0
        for i in range(len(self.layers)):
            loss += self.weights[i] * self.mse_loss(SR_features[i], HR_features[i])

        l1_loss = self.l1_loss(HR, SR)
        content_loss = lambda_l1 * l1_loss + loss

        return content_loss

    def adversarial_loss(self, discriminator, lr, hr, sr):
        """
        Compute the adversarial loss for the generator.
        """
        d_real = discriminator(lr, hr)[-1]
        d_fake = discriminator(lr, sr)[-1]
        adv = -torch.log(1 - d_real) - torch.log(d_fake)
        return adv.mean()

    def adversarial_feature_loss(self, discriminator, lr, hr, sr):
        """
        Compute the adversarial feature loss for the generator.
        """

        weights = [1 / 2, 1 / 2, 1 / 4, 1 / 32, 1 / 64]

        d_real = discriminator(lr, hr)
        d_fake = discriminator(lr, sr)

        advfeat = 0
        for idx in range(len(weights)):
            advfeat += weights[idx] * nn.MSELoss()(d_real[idx], d_fake[idx])

        return advfeat.mean()