class Trainer(nn.Module):

    def __init__(self, generator: nn.Module = None, discriminator: nn.Module = None,
                 g_loss: nn.Module = None, d_loss: nn.Module = None,
                 batch_size: int = 4, dataloader: DataLoader = None,
                 mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
                 std: tuple[float, float, float] = (0.5, 0.5, 0.5),
                 device: str = 'cuda',
                 model_logs_dir: str = 'model_logs',
                 outputs_dir: str = 'output_images') -> None:
        super().__init__()

        now = datetime.now()
        output_name = f'Month{now.month}_Day{now.day}_Hour{now.hour}'

        self.model_save_dir = os.path.join(model_logs_dir, output_name)
        self.image_output_dir = os.path.join(outputs_dir, output_name)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)

        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.device = device

    def denorm(self, img_tensor: torch.Tensor) -> torch.Tensor:
        return img_tensor * self.mean + self.std

    def save_samples(self, low_res_images: torch.Tensor, index: int = 0) -> None:
        self.generator.eval()
        low_res_images = low_res_images[0].to(self.device)

        with torch.no_grad():
            super_res_images = self.generator(low_res_images)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(low_res_images.permute(1, 2, 0).cpu().numpy())
        ax[0].axis('off')
        ax[1].imshow(super_res_images.permute(1, 2, 0).cpu().numpy())
        ax[1].axis('off')
        plt.show()

        plt.savefig(os.path.join(self.image_output_dir, f'output_image_{index:04d}.png'), bbox_inches='tight',
                    pad_inches=0.1)
        print(f'Saving Sample Epoch: {index:04d}')

    def save_model(self, epoch):
        gen_path = os.path.join(self.model_save_dir, f'generator_epoch_{epoch}.pth')
        disc_path = os.path.join(self.model_save_dir, f'discriminator_epoch_{epoch}.pth')

        torch.save(self.generator, gen_path)

        torch.save(self.discriminator, disc_path)

        print(f"Models saved at epoch {epoch}")

    def train_discriminator(self, lr_image, hr_image, opt_d):
        opt_d.zero_grad()

        hr_preds = self.discriminator(lr_image, hr_image)[-1]
        hr_targets = torch.ones(lr_image.size(0), 1, device=self.device)
        real_loss = F.binary_cross_entropy(hr_preds, hr_targets)
        real_score = hr_preds.mean()

        sr_image = self.generator(lr_image)
        sr_targets = torch.zeros(lr_image.size(0), 1, device=self.device)
        sr_preds = self.discriminator(lr_image, sr_image)[-1]
        fake_loss = F.binary_cross_entropy(sr_preds, sr_targets)
        fake_score = sr_preds.mean()

        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()

        return loss.item(), real_score, fake_score

    def train_generator(self, lr_image, hr_image, opt_g):
        opt_g.zero_grad()

        sr_image = self.generator(lr_image)

        preds = self.discriminator(lr_image, sr_image)[-1]
        targets = torch.ones(lr_image.size(0), 1, device=self.device)
        loss = self.g_loss(self.discriminator, lr_image, hr_image, sr_image)

        loss.backward()
        opt_g.step()

        return loss.item()

    def fit(self, epochs: int = 10, learning_rate: float = 1e-4, beta: tuple[float, float] = (0.95, 0.999),
            start_idx=1):
        torch.cuda.empty_cache()

        losses_g = []
        losses_d = []
        real_scores = []
        fake_scores = []

        opt_d = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=beta)
        opt_g = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=beta)

        for epoch in range(epochs):
            for real_images in tqdm.tqdm(self.dataloader):
                lr_image, hr_image = real_images[0].to(self.device), real_images[1].to(self.device)
                loss_d, real_score, fake_score = self.train_discriminator(lr_image, hr_image, opt_d)
                loss_g = self.train_generator(lr_image, hr_image, opt_g)

            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)

            print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

            self.save_samples(lr_image, epoch + start_idx)

            if epoch % 5 == 0:
                self.save_model(epoch)

        return losses_g, losses_d, real_scores, fake_scores