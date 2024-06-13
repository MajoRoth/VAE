from trainers.trainer import Trainer
import numpy as np
import torch
from tqdm import tqdm
import wandb



class AmortizedTrainer(Trainer):

    def train_step(self, features):
        inputs, labels, one_hots = features
        self.optim.zero_grad()
        recon_x, mu, logvar = self.model(inputs)
        loss = self.loss(inputs, recon_x, mu, logvar)
        loss.backward()
        self.optim.step()
        return loss, recon_x

    def forward_and_loss(self, features):
        inputs, labels, one_hots = features
        self.optim.zero_grad()
        recon_x, mu, logvar = self.model(inputs)
        loss = self.loss(inputs, recon_x, mu, logvar)
        return loss, recon_x

    def run_valid_loop(self):
        losses = []

        with torch.no_grad():
            # validation loss
            for i, features in enumerate(tqdm(self.test_dataset, desc=f'Valid')):
                loss, outputs = self.forward_and_loss(features)

                losses.append(loss)

            valid_loss = np.mean(losses)
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.save_to_checkpoint()

            self._write_summary_valid(self.step, valid_loss)
            self._sample_wandb()

    def _sample_wandb(self):
        recon_x, mu, logvar = self.model(self.train_images)
        recon_train_images = [wandb.Image(image.squeeze(0).numpy(), caption=f"Recon Image {i + 1}") for i, image in
                              enumerate(recon_x)]

        recon_x, mu, logvar = self.model(self.validation_images)
        recon_valid_images = [wandb.Image(image.squeeze(0).numpy(), caption=f"Recon Image {i + 1}") for i, image in
                              enumerate(recon_x)]

        wandb.log({"recon train images": recon_train_images}, step=self.step)
        wandb.log({"recon valid images": recon_valid_images}, step=self.step)

        # sample from latent space
        z = torch.randn(10, self.cfg.model.latent_dimension)
        samples = self.model.decode(z)
        sampled_images = [wandb.Image(image.squeeze(0).numpy(), caption=f"Sampled Image {i + 1}") for i, image in
                          enumerate(samples)]
        wandb.log({"sampled images": sampled_images}, step=self.step)
