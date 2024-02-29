import torch
import pandas as pd
import sys
sys.path.insert(
    0, '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/Experimental_update_revision/Post-hoc_and_visualization/model')
from priorVAE_dynamic_structure import priorVAE
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb     
import yaml
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



my_parser = argparse.ArgumentParser()
my_parser.add_argument('-model_path', required=True,
                       help='Offer the full path of the saved model.')
my_parser.add_argument('-config_name', required=True,
                       help='Offer the name of the yaml file, they should be saved in conf folder in advance.')
my_parser.add_argument('-model_name', required=False, help="Define the name you use for saving the data, the default is the same as the config name")
my_parser.add_argument('-save_type', required=False, default='both', help = "Clarify which result to save, default as 'both' for latent_res + reconstructed, 'latent' for latent res only, 'recon' for reconstructed only" )
args = my_parser.parse_args()
res_save_path = "/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/"

os.makedirs(res_save_path+"latent_mu_nsclc/", exist_ok=True)
os.makedirs(res_save_path+"latent_sigma_nsclc/", exist_ok=True)
os.makedirs(res_save_path+"reconstructed_value_nsclc/", exist_ok=True)
os.makedirs(res_save_path+"resampled_dim_nsclc/", exist_ok=True)

os.makedirs(res_save_path+"latent_mu_sclc/", exist_ok=True)
os.makedirs(res_save_path+"latent_sigma_sclc/", exist_ok=True)
os.makedirs(res_save_path+"reconstructed_value_sclc/", exist_ok=True)
os.makedirs(res_save_path+"resampled_dim_sclc/", exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config_name = args.config_name

print('Submitting model for ' + str(config_name))
with open(config_name, 'r') as file:
    config = yaml.safe_load(file)
#print(str(config))
#print(str(config['test_set']))
encoder_config = config['encoder_config']
latent_dim = config['latent_dim']
n_features_path = latent_dim
batch_size = config['batch_size']


#X_test = pd.read_pickle(config['test_set'])
X_train_tem = config['test_set']
directory, filename = X_train_tem.rsplit('/', 1)
new_filename = filename.replace("test", "nsclc", 1)
new_filename_2 = filename.replace("test", "sclc", 1)
X_train_path = f"{directory}/{new_filename}"
X_train_path_2 = f"{directory}/{new_filename_2}"
X_train = pd.read_pickle(X_train_path)
X_train_2 = pd.read_pickle(X_train_path_2)

n_features_trans = X_train.shape[1]
n_input = X_train.shape[1]

mu_prior = pd.read_csv(config['mu'], index_col=0)
sigma_prior = pd.read_csv(config['sigma'], index_col=0)
log_sigma_prior = np.log(sigma_prior)
mu_prior = mu_prior.add_suffix('_mu')
sigma_prior = sigma_prior.add_suffix('_sigma')
log_sigma_prior = log_sigma_prior.add_suffix('_logSigma')
#X_test = X_test.join(mu_prior)
#X_test = X_test.join(sigma_prior)
#X_test = X_test.join(log_sigma_prior)
X_train = X_train.join(mu_prior)
X_train = X_train.join(sigma_prior)
X_train = X_train.join(log_sigma_prior)
n_features_path = mu_prior.shape[1]

X_train_2 = X_train_2.join(mu_prior)
X_train_2 = X_train_2.join(sigma_prior)
X_train_2 = X_train_2.join(log_sigma_prior)

#print(X_test.head(2))

#X_test = X_test.to_numpy()
X_train = X_train.to_numpy()
X_train_2 = X_train_2.to_numpy()

#test_set = torch.Tensor(X_test)
#test_set = TensorDataset(test_set, test_set)
#test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

train_set = torch.Tensor(X_train)
train_set = TensorDataset(train_set, train_set)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

train_set_2 = torch.Tensor(X_train_2)
train_set_2 = TensorDataset(train_set_2, train_set_2)
train_loader_2 = DataLoader(train_set_2, batch_size=batch_size, shuffle=False)

def generate_latent_results_and_reconstruction(train_loader_name, model_path):
    model = priorVAE(encoder_config=encoder_config, n_input=n_input, latent_dim =latent_dim, n_features_path = n_features_path, n_features_trans=n_features_trans).to(device)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device(device)))
    model.eval()  # Set the model to inference mode
    model.to(device)
    reconstruction_all = []
    mu_all = []
    z_all = []
    sigma_all = []
    
    for batch in train_loader_name:
        input_tensor = batch[0].to(device)
    # Pass the data through the encoder
        with torch.no_grad():  # Ensure gradients are not computed for inference
            mu, sigma, z = model.encoder(input_tensor)
            reconstruction = model(input_tensor)
        if device.type == 'cuda':
            mu_all.append(mu.cpu())
            sigma_all.append(sigma.cpu())
            z_all.append(z.cpu())
            reconstruction_all.append(reconstruction.cpu())
        else:
            mu_all.append(mu)
            sigma_all.append(sigma)
            z_all.append(z)
            reconstruction_all.append(reconstruction)
            
    stacked_reconstructions = torch.cat(reconstruction_all, dim = 0)
    stacked_mu = torch.cat(mu_all, dim = 0)
    stacked_sigma = torch.cat(sigma_all, dim = 0)
    stacked_z = torch.cat(z_all, dim = 0)
            
    if device.type == 'cuda':
        reconstruction_np = stacked_reconstructions.cpu().numpy()
        mu_np = stacked_mu.cpu().numpy()
        sigma_np = stacked_sigma.cpu().numpy()
        z_np = stacked_z.cpu().numpy()
    else:
        reconstruction_np = stacked_reconstructions.numpy()
        mu_np = stacked_mu.numpy()
        sigma_np = stacked_sigma.numpy()
        z_np = stacked_z.numpy()
        
    return mu_np, sigma_np, z_np, reconstruction_np


def get_root_name(path):
    # Extract the base filename from the path
    base_name = os.path.basename(path)
    # Split the base name and the extension and return the name part
    root_name, _ = os.path.splitext(base_name)
    return root_name

if args.model_name is not None:
    save_name = args.model_name
else:
    save_name = get_root_name(config_name)



mu_res, sigma_res, z_res, recon_res  = generate_latent_results_and_reconstruction(train_loader, args.model_path)
np.save(res_save_path+"latent_mu_nsclc/"+save_name+"_latent_result.npy", mu_res)
np.save(res_save_path+"latent_sigma_nsclc/"+save_name+"_latent_sigma_result.npy", sigma_res)
np.save(res_save_path+"reconstructed_value_nsclc/"+save_name+"_reconstructed_result.npy", recon_res)
np.save(res_save_path+"resampled_dim_nsclc/"+save_name+"_z_result.npy", z_res)


mu_res, sigma_res, z_res, recon_res  = generate_latent_results_and_reconstruction(train_loader_2, args.model_path)
np.save(res_save_path+"latent_mu_sclc/"+save_name+"_latent_result.npy", mu_res)
np.save(res_save_path+"latent_sigma_sclc/"+save_name+"_latent_sigma_result.npy", sigma_res)
np.save(res_save_path+"reconstructed_value_sclc/"+save_name+"_reconstructed_result.npy", recon_res)
np.save(res_save_path+"resampled_dim_sclc/"+save_name+"_z_result.npy", z_res)
