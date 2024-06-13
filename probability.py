import torch
import json
import argparse
import os
import random
import numpy as np
from tqdm import tqdm

import torchvision
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import wandb
import torchvision.transforms as transforms
from models.amortized_vae import AmortizedVAE


from trainers.trainer_getter import get_trainer
from models.model_getter import get_model
from trainer import Trainer
from confs.conf_getter import get_conf
from train import IndexedDataset

def get_random_images_per_label(dataloader, number_of_images=5):
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
    for label, indices in sorted(label_indices.items()):
        random_indexes = random.sample(indices, number_of_images)
        random_images_per_label = []
        for random_idx in random_indexes:
            random_images_per_label.append(dataset[random_idx][0])

        random_images.append(random_images_per_label)

    # Stack the images along a new dimension to form a tensor of shape [10, 1, 28, 28]
    concatenated_lists = [torch.cat(sublist, dim=0) for sublist in random_images]

    # Step 2: Concatenate all lists along the first dimension to form a single batch
    batch = torch.cat(concatenated_lists, dim=0)

    # Step 3: Reshape the batch to [50, 1, 28, 28]
    batch = batch.view(10 * number_of_images, 1, 28, 28)
    return batch




def log_probability(model, samples, sigma_p=0.4, M=1000):
    mu, logvar = model.encode(samples)
    sigma = torch.exp(0.5 * logvar)
    batch_size, latent_dim = mu.shape
    gausian_factor = -0.5 * latent_dim * np.log(2 * torch.pi)

    log_list = list()
    for i in tqdm(range(M)):
        z = model.reparameterize(mu, logvar)
        recon_imgs = model.decode(z)

        log_p_z = gausian_factor - torch.norm(z - mu, dim=1) ** 2 / 2
        log_q_z = gausian_factor - torch.log(torch.prod(sigma, 1)) - torch.sum((z - mu) ** 2 / sigma ** 2, dim=1) / 2
        log_p_x_z = gausian_factor - latent_dim * np.log(sigma_p) - torch.sum((samples - recon_imgs) ** 2, dim=(1, 2, 3)) / (2 * sigma_p ** 2)

        sum_log = log_p_z + log_p_x_z - log_q_z
        log_list.append(sum_log.detach().numpy())

    return torch.logsumexp(torch.tensor(log_list), dim=0) - np.log(M)


def prob(model, train_dataset, test_dataset):
    train_samples = get_random_images_per_label(train_dataset)
    test_samples = get_random_images_per_label(test_dataset)

    train_prob = log_probability(model, train_samples)
    test_prob = log_probability(model, test_samples)

    reshaped_images = train_samples.view(10, 5, 1, 28, 28)
    reshaped_train_probs = train_prob.view(10, 5)
    reshaped_test_probs = test_prob.view(10, 5)


    average_prob_train = reshaped_train_probs.mean(dim=1)
    average_prob_test = reshaped_test_probs.mean(dim=1)

    random_indices = torch.randint(5, (10,))
    sampled_images = reshaped_images[torch.arange(10), random_indices]
    sampled_probs = reshaped_train_probs[torch.arange(10), random_indices]
    sampled_images_wandb = [wandb.Image(image.squeeze(0).numpy(), caption=f"log probability {sampled_probs[i]:.2f}") for i, image in enumerate(sampled_images)]
    wandb.log({"train sampled images with log probabilities": sampled_images_wandb})

    table = wandb.Table(columns=[])
    table.add_column(name="digit", data=list(range(10)))
    table.add_column(name="random sample", data=sampled_images_wandb)
    table.add_column(name="log probability", data=[f"{i:.2f}" for i in sampled_probs])
    table.add_column(name="average log probability train", data=[f"{i:.2f}" for i in average_prob_train])
    table.add_column(name="average log probability test", data=[f"{i:.2f}" for i in average_prob_test])
    average_prob = [(x + y)/2 for x, y in zip(average_prob_test, average_prob_train)]
    table.add_column(name="total average log probability", data=[f"{i:.2f}" for i in average_prob])


    print(f"train mean: {average_prob_train.mean()} test mean: {average_prob_test.mean()}")

    wandb.log({"table": table})





def get_parser():
    parser = argparse.ArgumentParser(description='calculate log probability of a model')
    parser.add_argument('--ckpt', default="./checkpoints/amortized_vae/model_3.pt", type=str)
    parser.add_argument('--train', default="./checkpoints/amortized_vae/train.pt", type=str)
    parser.add_argument('--test', default="./checkpoints/amortized_vae/test.pt", type=str)

    parser.add_argument('--conf', default="amortized_vae", type=str)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    cfg = get_conf(args.conf)

    model = AmortizedVAE(latent_dim=cfg.model.latent_dimension)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    wandb.init(project="VAE", name="probability", resume="allow", notes=f"{cfg}")

    res = prob(model, torch.load(args.train), torch.load(args.test))


