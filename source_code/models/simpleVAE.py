import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dim = 50

class Encoder(nn.Module):
    def __init__(self, latent_dim, n_inputs):
        super(Encoder, self).__init__()
        self.enc1 = nn.Linear(n_inputs, 1000)
        self.enc2 = nn.Linear(1000, 100)
        self.enc3 = nn.Linear(100, latent_dim*2)
        self.kl = 0

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std)
        return sample

    def forward(self, x):
        x = F.leaky_relu(self.enc1(x))
        x = F.leaky_relu(self.enc2(x))
        # reshape the latent layer
        x = self.enc3(x).view(-1, 2, latent_dim)

        mu = x[:, 0, :]  # first dimension for mu
        log_var = x[:, 1, :]  # second dimension for log_var
        z = self.reparameterization(mu, log_var)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return mu, log_var, z
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_inputs):
        super(Decoder,self).__init__()
        self.dec1 = nn.Linear(latent_dim, 100)
        self.dec2 = nn.Linear(100, 1000)
        self.dec3 = nn.Linear(1000, n_inputs)
    
    def forward(self, x):
        x = F.leaky_relu(self.dec1(x))
        x = F.leaky_relu(self.dec2(x))
        reconstruction = F.softplus(self.dec3(x))
        return reconstruction


## Model of the simple VAE
class simpleVAE(nn.Module):
    def __init__(self, n_inputs):
        super(simpleVAE, self).__init__()
        self.encoder = Encoder(latent_dim, n_inputs)
        self.decoder = Decoder(latent_dim, n_inputs)

    def forward(self, x):
        z = self.encoder(x)[2]
        return self.decoder(z)





    

        

