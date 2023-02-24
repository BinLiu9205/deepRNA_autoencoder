import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dim = 50


class Encoder(nn.Module):
    def __init__(self, latent_dim, n_inputs):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(n_inputs, 1000)
        self.enc2 = nn.Linear(1000, 100)
        self.enc3 = nn.Linear(100, latent_dim)




    #def forward(self, x, prior_mu, prior_sigma, prior_sigma_square):

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        # reshape the latent layer
        x = self.enc3(x)

        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_inputs):#, n_features_path, n_features_trans):
        super(Decoder, self).__init__()
        self.dec1 = nn.Linear(latent_dim, 100)
        self.dec2 = nn.Linear(100, 1000)
        self.dec3 = nn.Linear(1000, n_inputs)
        # self.n_features_path = n_features_path
        # self.n_features_trans = n_features_trans

    def forward(self, x):
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        reconstruction = F.softplus(self.dec3(x))
        return reconstruction


## Model of the simple VAE
class simpleAE(nn.Module):
    def __init__(self, n_inputs):
        super(simpleAE, self).__init__()
        self.encoder = Encoder(latent_dim, n_inputs,)
        self.decoder = Decoder(latent_dim, n_inputs)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
