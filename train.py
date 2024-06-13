import torch
import json
import argparse
import os

import torchvision
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import wandb
import torchvision.transforms as transforms


from trainers.trainer_getter import get_trainer
from models.model_getter import get_model
from confs.conf_getter import get_conf


class IndexedDataset(Dataset):
    def __init__(self, dataset, train):
        self.dataset = dataset
        self.train = train
        self.num_samples = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        target_one_hot = torch.zeros(self.num_samples)

        if self.train:
            target_one_hot[idx] = 1

        return image, label, target_one_hot



def train(args):
    cfg = get_conf(args.conf)
    print(f'ARGS: {args}')
    print(f'PARAMS: {cfg}')

    cfg.model.dir = f"./checkpoints/{args.conf}"

    wandb.init(project="VAE", name=args.conf, resume="allow", notes=f"{cfg}")

    ckpts_dir = "checkpoints"
    args.model_dir = f"{ckpts_dir}/{cfg.model.name}"
    os.makedirs(args.model_dir, exist_ok=True)

    # data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_targets = train_dataset.targets
    train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets)
    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    train_dataset = IndexedDataset(train_dataset, train=True)  # Wrap with IndexedDataset
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_dataset = IndexedDataset(test_dataset, train=False)  # Wrap with IndexedDataset
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = get_model(cfg)
    if torch.cuda.is_available():
        model = model.cuda()

    trainer = get_trainer(cfg)
    trainer(cfg=cfg, model=model, train_dataset=train_loader, test_dataset=test_loader).train()


def get_parser():
    parser = argparse.ArgumentParser(description='train an neural network')
    parser.add_argument('--conf', default="amortized_vae", type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    train(args)
