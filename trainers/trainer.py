import random
import numpy as np
import torch.nn as nn
import torch
import os
from tqdm import tqdm
import wandb
import torch.optim as optim
from abc import ABC, abstractmethod


##### Taken from https://github.com/microsoft/NeuralSpeech/tree/master/PriorGrad-vocoder #####
def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


##############################################################################################

def get_random_images_per_label(dataloader, one_hot=False):
    dataset = dataloader.dataset

    # Dictionary to store indices of images for each label
    label_indices = {}

    # Populate label_indices with indices of images for each label
    for idx, (image, label, one_hot) in enumerate(dataset):
        if label not in label_indices:
            label_indices[label] = []
        label_indices[label].append(idx)

    # Randomly select one image per label
    random_images = []
    random_one_hots = []
    for label, indices in sorted(label_indices.items()):
        random_idx = random.choice(indices)
        random_images.append(dataset[random_idx][0])
        random_one_hots.append(dataset[random_idx][2])# Only append the image, not the label

    # Stack the images along a new dimension to form a tensor of shape [10, 1, 28, 28]
    random_images = torch.stack(random_images)
    random_one_hots = torch.stack(random_one_hots)

    return random_images, random_one_hots


class Trainer(ABC):
    def __init__(self, cfg, model, train_dataset, test_dataset):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.optim = optim.Adam(self.model.parameters(), lr=cfg.trainer.lr)
        self.recon_loss = nn.MSELoss()

        self.best_valid_loss = 1000000
        self.step = 0
        self.epoch = -1

        # wandb
        self.train_images, self.train_one_hots = get_random_images_per_label(self.train_dataset)
        self.validation_images, _ = get_random_images_per_label(self.test_dataset)


        wandb.log({"input train images": [wandb.Image(image.squeeze(0).numpy(), caption=f"Input Image {i + 1}") for
                                          i, image in enumerate(self.train_images)]}, step=self.step)
        wandb.log({"input validation images": [wandb.Image(image.squeeze(0).numpy(), caption=f"Input Image {i + 1}") for
                                          i, image in enumerate(self.validation_images)]}, step=self.step)

    def train(self):
        self.step = 0

        for epoch in range(self.cfg.trainer.epochs):
            self.epoch = epoch
            total_loss = 0
            batch_step = 0
            for features in tqdm(self.train_dataset, desc=f'Epoch: {epoch}'):
                loss, outputs = self.train_step(features)
                batch_step += 1
                total_loss += loss.item()

                if self.step % 1000 == 0:
                    self._write_summary(self.step, total_loss / batch_step)

                self.step += 1

            self.run_valid_loop()

    def loss(self, inputs, recon_x, mu, logvar):
        # recon loss + KL loss
        recon_loss = self.recon_loss(recon_x, inputs)

        sigma = torch.exp(0.5 * logvar)
        kl_loss = mu.pow(2) + sigma.pow(2) - torch.log(sigma) - 1
        kl_loss = torch.mean(kl_loss)
        return recon_loss + kl_loss


    def save_to_checkpoint(self):
        save_name = f'{self.cfg.model.dir}/model_{self.epoch}.pt'
        torch.save(self.model.state_dict(), save_name)
        torch.save(self.train_dataset, f"{self.cfg.model.dir}/train.pt")
        torch.save(self.test_dataset, f"{self.cfg.model.dir}/test.pt")

    @abstractmethod
    def train_step(self, features):
        pass

    @abstractmethod
    def forward_and_loss(self, features):
        pass

    @abstractmethod
    def run_valid_loop(self):
        pass

    def _write_summary(self, step, loss):
        wandb.log({"train/loss": loss}, step=self.step)

    def _write_summary_valid(self, step, loss):
        wandb.log({"valid/loss": loss}, step=self.step)

