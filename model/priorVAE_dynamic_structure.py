import torch
import torch.nn as nn
import torch.nn.functional as F

#latent_dim = 51 (?) or any number of the pathways in the dataset
#encoder_config is a list of node number for each layer: [1000, 100]
#n_input = shape of the input


class Encoder(nn.Module):
    def __init__(self, encoder_config, n_input, latent_dim, n_features_path, n_features_trans):
        super(Encoder, self).__init__()
        input_size = n_input
        encoder_layers = []
        for neurons in encoder_config:
            encoder_layers.append(nn.Linear(input_size, neurons))
            input_size = neurons
        encoder_layers.append(nn.Linear(input_size, latent_dim*2))
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.kl = 0
        self.n_features_path = n_features_path
        self.n_features_trans = n_features_trans
        self.latent_dim = latent_dim
        

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std)
        return sample



    #def forward(self, x, prior_mu, prior_sigma, prior_sigma_square):

    def forward(self, x):
        total_layers = len(self.encoder_layers)
        x_total = x.clone().detach()
        x = F.leaky_relu(self.encoder_layers[0](x_total[:, 0:self.n_features_trans]))
        for layer_index in range(1, total_layers - 1):
            x = F.leaky_relu(self.encoder_layers[layer_index](x))
        # reshape the latent layer
        x = self.encoder_layers[-1](x).view(-1, 2, self.latent_dim)
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
    def __init__(self, encoder_config, n_input, latent_dim):
        super(Decoder, self).__init__()
        layers = []
        decoder_config = encoder_config[::-1] 
        input_size = latent_dim
        for neurons in decoder_config[0:]:
            layers.append(nn.Linear(input_size, neurons))
            input_size = neurons
        self.decoder_layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(decoder_config[-1], n_input)
    def forward(self, x):
        for layer in self.decoder_layers:
            x = F.leaky_relu(layer(x))
        reconstruction = F.softplus(self.final_layer(x))
        return reconstruction


class priorVAE(nn.Module):
    def __init__(self, encoder_config, n_input, latent_dim, n_features_path, n_features_trans):
        super(priorVAE, self).__init__()
        self.encoder = Encoder(encoder_config, n_input, latent_dim, n_features_path, n_features_trans)
        self.decoder = Decoder(encoder_config, n_input, latent_dim)

    # We only use the z for the reconstruction on the decoder's side        
    def forward(self, x):
        z = self.encoder(x)[2]
        return self.decoder(z)

        
        

        
        