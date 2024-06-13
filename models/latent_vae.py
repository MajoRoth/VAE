import torch
from torch import nn
from torch.nn import init


class LatentVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=20):
        super(LatentVAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        # Latent space
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128)

        self.fc_mu.weight.data.fill_(0)
        self.fc_logvar.weight.data.fill_(0)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2),  # (batch_size, 128, 2, 2)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 1, 28, 28)
        )

    def reparameterize(self, mu, logvar):
        epsilon = torch.normal(0, 1, mu.shape)
        sigma = torch.exp(0.5 * logvar)
        z = mu + sigma * epsilon
        return z


    def encode(self, x):
        # x is now one hot
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 128, 1, 1)
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
