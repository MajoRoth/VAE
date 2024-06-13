from pathlib import Path
from trainers.amortized_trainer import AmortizedTrainer
from trainers.latent_trainer import LatentTrainer


def get_trainer(cfg):
    if cfg.model.name == "amortized_vae":
        return AmortizedTrainer
    elif cfg.model.name == "latent_vae":
        return LatentTrainer
    else:
        raise Exception(f"Model {cfg.model.name} not found")



