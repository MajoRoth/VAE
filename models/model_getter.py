
from pathlib import Path
from models.amortized_vae import AmortizedVAE
from models.latent_vae import LatentVAE


def get_model(cfg):
    conf_getter_path = Path(__file__)
    models = {p.stem: p for p in conf_getter_path.parent.glob("*.py")}
    print(models)
    if cfg.model.name == "amortized_vae":
        return AmortizedVAE(latent_dim=cfg.model.latent_dimension)
    elif cfg.model.name == "latent_vae":
        return LatentVAE(input_dim=cfg.model.input_dimension, latent_dim=cfg.model.latent_dimension)
    else:
        raise Exception(f"Model {cfg.model.name} not found")



