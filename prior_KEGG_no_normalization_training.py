import yaml
import sys
sys.path.insert(
    0, '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/Experimental_update_revision/Hyperparameter_search_for_structure/model')
import argparse
#from simpleVAE import simpleVAE
#from simpleAE import simpleAE
from priorVAE_dynamic_structure import priorVAE
#from priorVAE import priorVAE
from utils import EarlyStopping, LRScheduler
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score
import scipy.stats as stats
from torchsummary import summary


import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime

## Setting up hydra and wandb

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb        

#with open('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Hyperparameter_search_for_structure/training/conf/kegg_training_trial1.yaml', 'r') as file:
#    sweep_config = yaml.safe_load(file) 
np.random.seed(42)

@hydra.main(config_path="conf", config_name="model_prior_kegg_no_normalization_static")
def main(cfg: DictConfig) -> None:
    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project='deepRNA', config=flat_cfg)
    #OmegaConf.set_struct(cfg, False)  # Allow changes to the config
    #cfg.merge_with_dotlist(
    #    [f"{k}={v}" for k, v in wandb.config.as_dict().items()])
    OmegaConf.set_struct(cfg, True)
    # 
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    beta = cfg.training.beta
    criterion = nn.MSELoss(reduction='sum')
    encoder_config = cfg.model.encoder_config
    latent_dim = cfg.model.latent_dim
    n_features_path =cfg.model.n_features_path
      
    
    
    def fit(model, dataloader):
        model.train()
        running_loss = 0.0
        mse_running = 0.0
        kl_running  = 0.0
        for i, data in enumerate(dataloader):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            data_predict = model(data)
            mse_loss = criterion(data_predict, data[:, 0:n_features_trans])
            #print(str(mse_loss))
            loss = mse_loss + beta*model.encoder.kl
            #print(str(model.encoder.kl))
            mse_running  += mse_loss.item()
            kl_running += model.encoder.kl.item()
            running_loss += loss.item()
            loss.backward()
            # Check gradients of the model parameters
            #for name, param in model.named_parameters():
            #    if not torch.isnan(param.grad).any() and not torch.isinf(param.grad).any():
            #        print(f'Everything is fine in {name}')
            optimizer.step()
        train_loss = running_loss/len(dataloader.dataset)
        mse_loss_all = mse_running/len(dataloader.dataset)
        kl_loss_all = kl_running/len(dataloader.dataset) 
        #print(len(dataloader.dataset))

        return train_loss, mse_loss_all, kl_loss_all

    def validate(model, dataloader):
        model.eval()
        running_loss = 0.0
        mse_running = 0.0
        kl_running  = 0.0
        for i, data in enumerate(dataloader):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            data_predict = model(data)
            mse_loss = criterion(data_predict, data[:, 0:n_features_trans])
            loss = mse_loss + beta*model.encoder.kl
            mse_running  += mse_loss.item()
            kl_running += model.encoder.kl.item()
            running_loss += loss.item()
        val_loss = running_loss/len(dataloader.dataset)
        mse_loss_all = mse_running/len(dataloader.dataset)
        kl_loss_all = kl_running/len(dataloader.dataset) 
        return val_loss, mse_loss_all, kl_loss_all


    inputdata_train = cfg.datasets.train_set
    inputdata_test = cfg.datasets.test_set
    prior_file_mu = cfg.gene_set_definition.mu
    prior_file_sigma = cfg.gene_set_definition.sigma
    
    X_train = pd.read_pickle(inputdata_train)
    X_test = pd.read_pickle(inputdata_test)
    
    n_features_trans = X_train.shape[1]
    n_input = X_train.shape[1]

    mu_prior = pd.read_csv(prior_file_mu, index_col=0)
    sigma_prior = pd.read_csv(prior_file_sigma, index_col=0)
    log_sigma_prior = np.log(sigma_prior)
    mu_prior = mu_prior.add_suffix('_mu')
    sigma_prior = sigma_prior.add_suffix('_sigma')
    log_sigma_prior = log_sigma_prior.add_suffix('_logSigma')
    X_train = X_train.join(mu_prior)
    X_train = X_train.join(sigma_prior)
    X_train = X_train.join(log_sigma_prior)
    X_test = X_test.join(mu_prior)
    X_test = X_test.join(sigma_prior)
    X_test = X_test.join(log_sigma_prior)
    n_features_path = mu_prior.shape[1]
    X_train.head(1)
    X_test.head(1)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_set = torch.Tensor(X_train)
    test_set = torch.Tensor(X_test)
    train_set = TensorDataset(train_set, train_set)
    test_set = TensorDataset(test_set, test_set)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    print('After the dataloader ' +
            str(n_input) + ' ' + str(X_test.shape))
    print(str(encoder_config)+ str(n_input) + str(latent_dim) + str(n_features_path) + str(n_features_trans))
    model = priorVAE(encoder_config=encoder_config, n_input=n_input, latent_dim =latent_dim, n_features_path = n_features_path, n_features_trans=n_features_trans).to(device)
    #model = priorVAE(n_inputs=n_input, n_features_trans=n_features_trans,
                     #n_features_path=n_features_path).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = LRScheduler(optimizer)
    print('Before summarizing up the info ' +
            str(n_input) + ' ' + str(X_test.shape))

    summary(model, (n_input+3*n_features_path,))

    print('After summarizing up the info ' +
            str(n_input) + ' ' + str(X_test.shape))



    train_loss = []
    val_loss = []
    mse_train_loss = []
    mse_test_loss = []
    kl_train_loss = []
    kl_test_loss = []
    start_time = time.time()


    for epoch in range(epochs):
        epochStartTime = time.time()
        res_train = fit(model, train_loader)
        res_test = validate(model, test_loader)
        train_epoch_loss = res_train[0]
        test_epoch_loss = res_test[0]
        reconstruct_epoch_train = res_train[1]
        reconstruct_epoch_test = res_test[1]
        kl_epoch_train = res_train[2]
        kl_epoch_test = res_test[2]
        lr_scheduler(test_epoch_loss)
        epochEndTime = time.time()
        #print(f"Train Loss: {train_epoch_loss:.4f}")
        #print(f"Val Loss: {val_epoch_loss:.4f}")
        #print(f"Epoch time: {epochEndTime-epochStartTime:.2f} sec")
        epoch_run_time = time.time() - epochStartTime

        wandb.log({"epoch_train_loss": train_epoch_loss, "epoch_test_loss": test_epoch_loss,
                "epoch_kl_train_loss": kl_epoch_train, "epoch_kl_test_loss": kl_epoch_test,
                "epoch_recon_train_loss": reconstruct_epoch_train, "epoch_recon_test_loss": reconstruct_epoch_test,
                "epoch_running_time": epoch_run_time})
        train_loss.append(train_epoch_loss)
        val_loss.append(test_epoch_loss)

    #wandb.log({"test_loss": val_loss, "test_KL": kl_test_loss})  
    run_id = wandb.run.id
    model_path = f"priorVAE_dynamic_model_{run_id}_kegg_no_normalization.pth"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)
    
    
#def run_sweep():
    #sweep_id = wandb.sweep(sweep=sweep_config, project="deepRNA")
#    wandb.agent(sweep_id, function=main, count = 25) #count=10)


if __name__ == "__main__":
    main()
    #run_sweep()

