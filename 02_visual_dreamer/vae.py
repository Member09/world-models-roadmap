import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        # Flatten
        self.linear1 = nn.Linear(input_dim, 512)
        self.activation = nn.ReLU()

        # The "Two-Head" monster
        # Head 1: Where is the cloud center? (Mean)
        self.fc_mu = nn.Linear(512, latent_dim)

        # Head 2: How big is the cloud? (Log Variance)
        # here, we are predicting Log-Var, as neural networks hate to predict small positive numbers directly.
        # It is easier to predict -5.0 ( which becomes exp(-5) = small sigma )
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        hidden = self.activation(self.linear1(x))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=2, output_dim=784):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(latent_dim, 512)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(512, output_dim)
        self.sigmoid = nn.Sigmoid() # forces o/p to [0,1]

    def forward(self, z):
        hidden = self.activation(self.linear1(z))
        reconstruction = self.sigmoid(self.linear2(hidden))
        return reconstruction


class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        mu, logvar = mu, logvar

        # 1. Convert Log-Var to Standard Deviation
        # Research thought: Why 0.5? Because log(std^2) = 2 * log(std). 
        # So log(std) = 0.5 * log_var.
        std = torch.exp(0.5 * logvar)
        # 2. Sample Epsilon from N(0, 1)
        # "epsilon" is just a random noise tensor of the same shape as std
        epsilon = torch.randn_like(std) # --> rand : Unoform distribution [0,1) ; randn : Normal(Gaussian) distribution (bell curve).
        # 3. The Trick: z is deterministic wrt mu and std!
        z = mu + std*epsilon
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, mu, logvar
