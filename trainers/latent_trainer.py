from trainers.trainer import Trainer
import torch
import wandb
import torch.optim as optim


class LatentTrainer(Trainer):

    def __init__(self, cfg, model, train_dataset, test_dataset):
        super().__init__(cfg, model, train_dataset, test_dataset)
        self.optim = optim.Adam([
            {'params': model.decoder.parameters(), 'lr': 0.001},
            {'params': model.fc_decode.parameters(), 'lr': 0.001},
            {'params': model.fc_mu.parameters(), 'lr': 0.01},
            {'params': model.fc_logvar.parameters(), 'lr': 0.01}
        ])

    def train_step(self, features):
        inputs, labels, one_hots = features
        self.optim.zero_grad()
        recon_x, mu, logvar = self.model(one_hots)
        loss = self.loss(inputs, recon_x, mu, logvar)
        loss.backward()
        self.optim.step()
        return loss, recon_x

    def forward_and_loss(self, features):
        inputs, labels, one_hots = features
        self.optim.zero_grad()
        recon_x, mu, logvar = self.model(one_hots)
        loss = self.loss(inputs, recon_x, mu, logvar)
        return loss, recon_x

    def run_valid_loop(self):
        with torch.no_grad():
            self._sample_wandb()

    def _sample_wandb(self):
        recon_x, mu, logvar = self.model(self.train_one_hots)
        recon_train_images = [wandb.Image(image.squeeze(0).numpy(), caption=f"Recon Image {i + 1}") for i, image in
                              enumerate(recon_x)]

        wandb.log({"recon train images": recon_train_images}, step=self.step)

        # sample from latent space
        z = torch.randn(10, self.cfg.model.latent_dimension)
        samples = self.model.decode(z)
        sampled_images = [wandb.Image(image.squeeze(0).numpy(), caption=f"Sampled Image {i + 1}") for i, image in
                          enumerate(samples)]
        wandb.log({"sampled images": sampled_images}, step=self.step)
