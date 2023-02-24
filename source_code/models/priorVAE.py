import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dim = 50


class Encoder(nn.Module):
    def __init__(self, latent_dim, n_inputs, n_features_path, n_features_trans):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(n_inputs, 1000)
        self.enc2 = nn.Linear(1000, 100)
        self.enc3 = nn.Linear(100, latent_dim*2)
        self.kl = 0
        self.n_features_path = n_features_path
        self.n_features_trans = n_features_trans

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std)
        return sample



    #def forward(self, x, prior_mu, prior_sigma, prior_sigma_square):

    def forward(self, x):
        x_total = x.clone().detach()
        x = F.leaky_relu(self.enc1(x_total[:, 0:self.n_features_trans]))
        x = F.leaky_relu(self.enc2(x))
        # reshape the latent layer
        x = self.enc3(x).view(-1, 2, latent_dim)

        mu = x[:, 0, :]  # first dimension for mu
        log_var = x[:, 1, :]  # second dimension for log_var
        z = self.reparameterization(mu, log_var)
        prior_log_var = x_total[:, (self.n_features_trans+2*self.n_features_path)
                                    :(self.n_features_trans+3*self.n_features_path)]
        prior_mu = x_total[:, (self.n_features_trans):(
            self.n_features_trans+self.n_features_path)]
        prior_var = x_total[:, (self.n_features_trans+self.n_features_path)
                                :(self.n_features_trans+2*self.n_features_path)]
        self.kl = torch.sum((1 + log_var - prior_log_var - ((mu-prior_mu).pow(2) + log_var.exp())/prior_var))

        self.kl = -0.5*self.kl

        return mu, log_var, z


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
class priorVAE(nn.Module):
    def __init__(self, n_inputs, n_features_path, n_features_trans):
        super(priorVAE, self).__init__()
        self.encoder = Encoder(latent_dim, n_inputs, n_features_path, n_features_trans)
        self.decoder = Decoder(latent_dim, n_inputs)

    def forward(self, x):
        z = self.encoder(x)[2]
        return self.decoder(z)
