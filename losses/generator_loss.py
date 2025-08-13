import torch
import torch.nn as nn
import torchvision.models as models

class GeneratorLossFunction(nn.Module):
    def __init__(self, device, lambda0=1, lambda1=0.01, lambda2=0.01, vgg_layers=[2, 7, 16, 25, 34]):
        super().__init__()
        self.device = device
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.vgg = models.vgg19(weights='DEFAULT').features[:max(vgg_layers)+1].to(device).eval()
        self.vgg_layers = vgg_layers
        self.vgg_weights = [1.0/len(vgg_layers)] * len(vgg_layers)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.adversarial_bce_loss = nn.BCELoss()

    def vgg_extract(self, x):
        # VGG expects 3-channel input, so we replicate the grayscale channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        layers_output = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.vgg_layers:
                layers_output.append(x)
        return layers_output

    def content_loss(self, hr_image, sr_image):
        hr_features = self.vgg_extract(hr_image)
        sr_features = self.vgg_extract(sr_image)
        feature_loss = sum(
            self.vgg_weights[i] * self.mse_loss(sr_features[i], hr_features[i])
            for i in range(len(self.vgg_layers))
        )
        return self.l1_loss(hr_image, sr_image) + feature_loss

    def adversarial_loss(self, sr_preds_from_d):
        targets = torch.full_like(sr_preds_from_d, 0.9).to(self.device)
        return self.adversarial_bce_loss(sr_preds_from_d, targets)

    def adversarial_feature_loss(self, d_real_features, d_fake_features):
        adv_feat_loss = sum(
            self.mse_loss(d_fake_features[i], d_real_features[i])
            for i in range(len(d_real_features))
        )
        return adv_feat_loss

    def forward(self, discriminator, lr_image, hr_image, sr_image):
        d_real_features = discriminator(lr_image, hr_image)
        d_fake_features = discriminator(lr_image, sr_image)

        sr_preds_from_d = d_fake_features[-1]

        loss_content = self.content_loss(hr_image, sr_image)
        loss_adv = self.adversarial_loss(sr_preds_from_d)
        loss_adv_feat = self.adversarial_feature_loss(d_real_features[:-1], d_fake_features[:-1])

        return (self.lambda0 * loss_content +
                self.lambda1 * loss_adv +
                self.lambda2 * loss_adv_feat)