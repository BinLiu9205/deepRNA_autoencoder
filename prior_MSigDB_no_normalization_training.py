#import sys
#sys.path.insert(
#    0, '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models')
import argparse
#from simpleVAE import simpleVAE
#from simpleAE import simpleAE
from priorVAE import priorVAE
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

import wandb



run = wandb.init(
    project = "deepRNA", 
    config = {
        "epochs" : 500,
        "architeture" : "priorVAE",
        #"beta" : 50,
        #"beta" :  250,
        "beta" : 1,
        "data" : "baseline_MSigDB_traditional_parameter_setting", 
        "batch_size" : 128, 
        "lr" : 0.0001,
        "pathway_definition" : "Hallmark", 
    }
)


beta = run.config.beta
print(beta)

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
        loss = mse_loss + beta*model.encoder.kl
        mse_running  += mse_loss.item()
        kl_running += model.encoder.kl.item()
        running_loss += loss.item()
        loss.backward()
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


## The training begins afterwards 

criterion = nn.MSELoss(reduction='sum')


inputdata_train = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl'
inputdata_test = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl'
prior_file_mu = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv'
prior_file_sigma = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv'

X_train = pd.read_pickle(inputdata_train)
X_test = pd.read_pickle(inputdata_test)
batch_size = run.config.batch_size

n_features_trans = X_train.shape[1]
n_inputs = X_train.shape[1]

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

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#lr = 0.0001
lr = run.config.lr
epochs = run.config.epochs

train_set = torch.Tensor(X_train)
test_set = torch.Tensor(X_test)
train_set = TensorDataset(train_set, train_set)
test_set = TensorDataset(test_set, test_set)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False)
print('After the dataloader ' +
        str(n_inputs) + ' ' + str(X_test.shape))
model = priorVAE(n_inputs=n_inputs, n_features_trans=n_features_trans,
                    n_features_path=n_features_path).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = LRScheduler(optimizer)
print('Before summarizing up the info ' +
        str(n_inputs) + ' ' + str(X_test.shape))

summary(model, (n_inputs+3*n_features_path,))

print('After summarizing up the info ' +
        str(n_inputs) + ' ' + str(X_test.shape))


#epochs = 100

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

    wandb.log({"epoch_train_loss": train_epoch_loss, "epoch_test_loss" : test_epoch_loss,
               "epoch_kl_train_loss": kl_epoch_train, "epoch_kl_test_loss": kl_epoch_test, 
               "epoch_recon_train_loss": reconstruct_epoch_train, "epoch_recon_test_loss": reconstruct_epoch_test,
               "epoch_running_time": epoch_run_time})
    train_loss.append(train_epoch_loss)
    val_loss.append(test_epoch_loss)
    

wandb.log({"test_loss": val_loss, "test_KL": kl_test_loss})


## Todo: The config can be different from the real values used in training -- one motivation to use hydra
#model.save(os.path.join(wandb.run.dir, "demo_priorVAE_gene_level_model.h5"))
run_id = wandb.run.id
model_path = f"priorVAE_dynamic_model_{run_id}.pth"
torch.save(model.state_dict(), model_path)
wandb.save(model_path)


#print(wandb.run.dir)

wandb.finish()
