import sys
sys.path.insert(
    0, '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models')
import argparse
from simpleVAE import simpleVAE
from simpleAE import simpleAE
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
class ModelNotAvailableError(Exception):

    def __init__(self, modelname, message="Current model is not in simpleAE, simpleVAE, priorVAE, beta_simpleVAE or beta_priorVAE"):
        self.modelname = modelname
        self.message = message
        super().__init__(self.message)


class MissingParticularInfoError(Exception):

    def __init__(self, missingvalue, message="Required files missing, please check the command line"):
        self.missingvalue = missingvalue
        self.message = message
        super().__init__(self.message)


class DataFormError(Exception):

    def __init__(self, wrongdataform, message="Wrong data form for given input"):
        self.wrongdataform = wrongdataform
        self.message = message
        super().__init__(self.message)


## The command line should include 
#  1. a list of models of interest  
#  2. training set
#  3. test set
#  4. other scores (e.g. permutation, mask e.g.)
#  5. hypterparameters
#  get by sys.argv[i]

my_parser = argparse.ArgumentParser()

my_parser.add_argument('requiredmodels',
                       help='Define the models you want to work on in a list\n \
                    supportive models currently include: simpleAE, simpleVAE, priorVAE, beta_simpleVAE, beta_priorVAE')
my_parser.add_argument('-inputdata_train', type=str,
                       required=True, default='', help='Define the data used for training')
my_parser.add_argument('-inputdata_test', type=str,
                       required=True, default='', help='Define the data used for test')
my_parser.add_argument('-outputpath', '-O', type=str, default='',
                       required=True, help='Define the directory to save the results')
my_parser.add_argument('-name_suffix', default='', required=False, help="Add_a_suffix_to_the_results_for_saving_them")


my_parser.add_argument('-modified_beta', required=False, help= 'Beta coeffecient for KL divergence, default as 250', default= 250)
my_parser.add_argument('-batchsize', required=False,
                       help='Batch size used for the training, default as 128', default=128)
my_parser.add_argument('-epochs', required=False,
                       help='Epoch number of the training, default as 200', default=200)
my_parser.add_argument('-latentdimension', required=False,
                       help='Latent dimension of the bottleneck, default as 50', default=50)
my_parser.add_argument('-learningrate', required=False,
                       help='Learning rate for starting point of the lr_scheduler, default as 0.0001', default=0.0001)
my_parser.add_argument('-random_seed', required=False,
                       help='Choose your lucky number of random seeds, default as 42', default=42)
my_parser.add_argument('-prior_file_mu', required=False,
                       help='Offer the prior Mu for prior models. Required when priorVAE or bete_priorVAE involved in the model list.', 
                       default="/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/pathway_scoring/prior_distribution/bootstrap_mean_MsigDB.csv")
my_parser.add_argument('-prior_file_sigma', required=False,
                       help='Offer the prior sigma for prior models. Required when priorVAE or bete_priorVAE involved in the model list.',
                       default="/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/pathway_scoring/prior_distribution/bootstrap_sigma_MsigDB.csv")
my_parser.add_argument('-data_transformation', required=False, default=True, help="Whether to apply MinMaxScaler transformation on the data, default as True")
my_parser.add_argument('-downstream_analysis', required=False, default=True,
                       help='Whether to run cluster and visualization based on the trained model, default to be True')
args = my_parser.parse_args()


torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

args.requiredmodels = args.requiredmodels.strip('[]').split(",")


merge_prior = False
for i in range(len(args.requiredmodels)):
    sg_model =args.requiredmodels[i]
    if sg_model not in ['simpleAE', 'simpleVAE', 'priorVAE', 'beta_simpleVAE', 'beta_priorVAE']:
        raise ModelNotAvailableError(sg_model)
    elif sg_model == 'priorVAE' or sg_model == 'beta_priorVAE':
        merge_prior = True

os.makedirs(os.path.dirname(args.outputpath +
            "/autoencoder_models/"), exist_ok=True)
os.makedirs(os.path.dirname(args.outputpath +
            "/encoder_models/"), exist_ok=True)
os.makedirs(os.path.dirname(args.outputpath +
            "/latent_results/"), exist_ok=True)
os.makedirs(os.path.dirname(args.outputpath +
            "/sigma_mu_VAE/"), exist_ok=True)
os.makedirs(os.path.dirname(args.outputpath +
            "/sigma_mu_VAE/mean/"), exist_ok=True)
os.makedirs(os.path.dirname(args.outputpath +
            "/sigma_mu_VAE/var/"), exist_ok=True)

if args.inputdata_train.find("csv") != -1:
    X_train = pd.read_csv(args.inputdata_train, index_col=0)
    X_test = pd.read_csv(args.inputdata_test, index_col=0)
elif args.inputdata_train.find("pkl") != -1:
    X_train = pd.read_pickle(args.inputdata_train)
    X_test = pd.read_pickle(args.inputdata_test)
else:
    print("Please offer with a correct format")
print('I am using the data from ' + args.inputdata_train)
print('My original training data has the dimension of ' + str(X_train.shape))
print('My original test data has the dimension of ' + str(X_test.shape))
colnames_df = X_train.columns
index_df_train = X_train.index
index_df_test = X_test.index
print('Resetting the index, my shape is ' + str(X_test.shape))
if args.data_transformation is True:
    print('Transforming data')
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train.columns = colnames_df
    X_train.set_index(index_df_train, inplace=True)
    X_test.columns = colnames_df
    X_test.set_index(index_df_test, inplace=True)
    

n_features_trans = X_train.shape[1]
n_inputs = X_train.shape[1]

print("This is the file for the prior " + args.prior_file_mu)

print('Before merging + changing to numpy ' + str(n_inputs) + ' ' + str(X_test.shape))
if merge_prior is True:
    if len(args.prior_file_sigma) > 0 and len(args.prior_file_mu) > 0 is None:
        raise MissingParticularInfoError('prior distribution')
    else:
        mu_prior = pd.read_csv(args.prior_file_mu, index_col=0)
        sigma_prior = pd.read_csv(args.prior_file_sigma, index_col=0)
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

print('After merging, before changing to numpy ' + str(n_inputs) + ' ' + str(X_test.shape))

if isinstance(X_train, pd.DataFrame):
    X_train = X_train.to_numpy()
    print('Convert X_train')
if isinstance(X_test, pd.DataFrame):
    X_test = X_test.to_numpy()
    print('Convert X_test')
if X_test.shape[1] != X_train.shape[1]:
    raise DataFormError('X_train and X_test must have the same shape')
print('After merging, after changing to numpy ' +
      str(n_inputs) + ' ' + str(X_test.shape))

for i in range(len(args.requiredmodels)):
    sg_model = args.requiredmodels[i]
    batch_size = int(args.batchsize)
    epochs = int(args.epochs)
    #epochs = 20
    latent_dim = int(args.latentdimension)
    #lr = 0.0001
    lr = float(args.learningrate)
    #early_stopping = EarlyStopping()
    criterion = nn.MSELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Processing model" + sg_model)
    print('Before start the data loader ' +
          str(n_inputs) + ' ' + str(X_test.shape))

    if sg_model == "priorVAE" or sg_model == "beta_priorVAE":
        print("I am running priorVAE derived models")
        if sg_model == "priorVAE":
            beta = 1
        else: 
            beta = int(args.modified_beta)

        train_set = torch.Tensor(X_train)
        test_set = torch.Tensor(X_test)
        train_set = TensorDataset(train_set, train_set)
        test_set = TensorDataset(test_set, test_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        print('After the dataloader ' +
              str(n_inputs) + ' ' + str(X_test.shape))
        model = priorVAE(n_inputs=n_inputs, n_features_trans=n_features_trans,n_features_path=n_features_path).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = LRScheduler(optimizer)
        print('Before summarizing up the info ' +
              str(n_inputs) + ' ' + str(X_test.shape))
        
        summary(model, (n_inputs+3*n_features_path,))
        
        print('After summarizing up the info ' +
              str(n_inputs) + ' ' + str(X_test.shape))

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

    elif sg_model == "simpleVAE" or sg_model == "beta_simpleVAE":
        print("I am running simple VAE derived models")
        if sg_model == "simpleVAE":
            beta = 1
        else:
            beta = int(args.modified_beta)

        train_set = torch.Tensor(X_train[:,0:n_features_trans])
        test_set = torch.Tensor(X_test[:, 0:n_features_trans])
        train_set = TensorDataset(train_set, train_set)
        test_set = TensorDataset(test_set, test_set)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False)
        model = simpleVAE(n_inputs=n_inputs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = LRScheduler(optimizer)
        print('my_xtest' + str(X_test.shape))


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
                mse_loss = criterion(data_predict, data)
                loss = mse_loss + beta*model.encoder.kl
                running_loss += loss.item()
                mse_running  += mse_loss.item()
                kl_running += model.encoder.kl.item()
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
                mse_loss = criterion(data_predict, data)
                loss = mse_loss + beta*model.encoder.kl
                running_loss += loss.item()
                mse_running  += mse_loss.item()
                kl_running += model.encoder.kl.item()
            val_loss = running_loss/len(dataloader.dataset)
            mse_loss_all = mse_running/len(dataloader.dataset)
            kl_loss_all = kl_running/len(dataloader.dataset) 
            return val_loss, mse_loss_all, kl_loss_all

    else:
        print("I am running simpleAE")
        train_set = torch.Tensor(X_train[:, 0:n_features_trans])
        test_set = torch.Tensor(X_test[:, 0:n_features_trans])
        train_set = TensorDataset(train_set, train_set)
        test_set = TensorDataset(test_set, test_set)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
                test_set, batch_size=batch_size, shuffle=False)
        model = simpleAE(n_inputs=n_inputs).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = LRScheduler(optimizer)
        summary(model, (n_inputs,))
        print('my_xtest' + str(X_test.shape))

        def fit(model, dataloader):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                data, _ = data
                data = data.to(device)
                data = data.view(data.size(0), -1)
                optimizer.zero_grad()
                data_predict = model(data)
                mse_loss = criterion(data_predict, data)
                loss = mse_loss
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_loss = running_loss/len(dataloader.dataset)
            #print(len(dataloader.dataset))
            return train_loss


        def validate(model, dataloader):
            model.eval()
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                data, _ = data
                data = data.to(device)
                data = data.view(data.size(0), -1)
                optimizer.zero_grad()
                data_predict = model(data)
                mse_loss = criterion(data_predict, data)
                loss = mse_loss
                running_loss += loss.item()

            val_loss = running_loss/len(dataloader.dataset)
            return val_loss
        

    train_loss = []
    val_loss = []
    mse_train_loss = []
    mse_test_loss = []
    kl_train_loss = []
    kl_test_loss = [] 
    start_time = time.time()
    for epoch in range(epochs):
        epochStartTime = time.time()
        print(f"Epoch {epoch+1} of {epochs}")
        if sg_model != "simpleAE":
            res_train = fit(model, train_loader)
            res_test = validate(model, test_loader)
            train_epoch_loss = res_train[0]
            val_epoch_loss = res_test[0]
            reconstruct_epoch_train = res_train[1]
            reconstruct_epoch_test = res_test[1]
            kl_epoch_train = res_train[2]
            kl_epoch_test = res_test[2]
            mse_train_loss.append(reconstruct_epoch_train)
            mse_test_loss.append(reconstruct_epoch_test)
            kl_train_loss.append(kl_epoch_train)
            kl_test_loss.append(kl_epoch_test)
        else:
            train_epoch_loss = fit(model, train_loader)
            val_epoch_loss = validate(model, test_loader)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        lr_scheduler(val_epoch_loss)
        epochEndTime = time.time()
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")
        print(f"Epoch time: {epochEndTime-epochStartTime:.2f} sec")
    run_time = time.time() - start_time
    print(f"Run time: {(run_time/60):.2f} mins")

    torch.save(model.state_dict(), args.outputpath +
               "/autoencoder_models/" + sg_model + ".pth")
    torch.save(model.encoder.state_dict(), args.outputpath +
               "/encoder_models/" + sg_model + "_encoder.pth")

    os.makedirs(os.path.dirname(args.outputpath +
                                "/" + sg_model + "_results/"), exist_ok=True)
    
    train_test_dict = {}
    train_test_dict['train_loss'] = train_loss
    train_test_dict['test_loss'] = val_loss
    
    detailed_loss = {}
    detailed_loss['recon_train'] = mse_train_loss
    detailed_loss['recon_test'] = mse_test_loss
    detailed_loss['kl_train'] = kl_train_loss
    detailed_loss['kl_test'] = kl_test_loss
    
    df_loss = pd.DataFrame(train_test_dict)
    df_detailed = pd.DataFrame(detailed_loss)   
    
    df_loss.to_csv(args.outputpath +
                "/" + sg_model + "_results/training_test_loss_" + sg_model + '.csv')
    df_detailed.to_csv(args.outputpath +
            "/" + sg_model + "_results/recon_kl_loss_" + sg_model + '.csv')
    
    # Loss plot
    fig = plt.figure()
    ax = plt.axes()
    #x_axis = np.linspace(1, epochs, num = epochs)
    x_axis = np.linspace(1, epoch+1, num=epoch+1)
    ax.set(xlabel='Epoch', ylabel="Loss")
    plt.xticks(np.arange(1, epoch+1, step=5))
    plt.plot(x_axis, train_loss, color='cyan', label="Train loss")
    plt.plot(x_axis, val_loss, color='orange', label="Validate loss")
    plt.legend()
    plt.savefig(args.outputpath +
                "/" + sg_model + "_results/loss_to_epoch_" + sg_model + '.png')
    
    


    if args.downstream_analysis is True:

        # Latent Dimension
        print('my_xtest' + str(X_test.shape))
        sample_val, _ = X_test.shape
        print('My sample size is something like ' + str(sample_val))
        latent_results = []
        mean_results = []
        var_results = []
        print(sg_model + " is the current model we are working on")
        # print(str(sample_val))
        # print(str(test_loader.dataset[1][0]))
        # print(str(test_loader.dataset[1][0].reshape(1, -1)))

        if "VAE" in sg_model:
            print("This is VAE related")
            for i in range(sample_val):
                data_tem = test_loader.dataset[i][0]
                output = model.encoder(data_tem.reshape(1, -1).to(device))
                output_new = output[2]
                mean_res = output[0]
                var_res = output[1]
                latent_results.append(output_new.cpu().detach().numpy())
                mean_results.append(mean_res.cpu().detach().numpy())
                var_results.append(var_res.cpu().detach().numpy())
                
                
        else:
            print("This is a simple model")
            for i in range(sample_val):
                data_tem = test_loader.dataset[i][0]
                output = model.encoder(data_tem.reshape(1, -1).to(device))
                latent_results.append(output.cpu().detach().numpy())
        
        latent_results = np.array(latent_results)
        latent_results = latent_results.reshape(sample_val, latent_dim)
        print('My latent result is something like ' + str(latent_results.shape))
        np.save(args.outputpath +
                "/latent_results/"+sg_model+".npy", latent_results)
        
        if len(mean_results) > 0:
            mean_results = np.array(mean_results)
            var_results = np.array(var_results)
            mean_results = mean_results.reshape(sample_val, latent_dim)
            var_results = var_results.reshape(sample_val, latent_dim)
            np.save(args.outputpath +
                "/sigma_mu_VAE/mean/"+sg_model+"mean_results_vae.npy", mean_results)
            np.save(args.outputpath +
                "/sigma_mu_VAE/var/"+sg_model+"var_results_vae.npy", var_results)

        # Clustering
        latent_tsne = TSNE(n_components=2).fit_transform(latent_results)
        silhouette_avg_kmeans = []
        silhouette_avg_gm = []
        

        for k in range(2, 15):
            #Initialize the class object
            kmeans = KMeans(n_clusters=k)
            gmModel = GaussianMixture(n_components=k)
        #predict the labels of clusters.
            label1 = kmeans.fit(latent_tsne).predict(latent_tsne)
            label2 = gmModel.fit(latent_tsne).predict(latent_tsne)
            silhouette_avg_kmeans.append(silhouette_score(latent_tsne, label1))
            silhouette_avg_gm.append(silhouette_score(latent_tsne, label2))
            #Getting unique labels

            u_labels1 = np.unique(label1)
            u_labels2 = np.unique(label2)

            #plotting the results:
            fig = plt.figure()
            for i in u_labels1:
                plt.scatter(latent_tsne[label1 == i, 0],
                            latent_tsne[label1 == i, 1], label=i)

            plt.savefig(args.outputpath +
                        "/" + sg_model + "_results/" + "kmeans_for_" + sg_model + "_" +
                        str(len(u_labels1))+"_clusters.png")

            fig = plt.figure()
            for i in u_labels2:
                plt.scatter(latent_tsne[label2 == i, 0],
                            latent_tsne[label2 == i, 1], label=i)

            plt.savefig(args.outputpath +
                        "/" + sg_model + "_results/" + "GMM_for_" + sg_model + "_" +
                        str(len(u_labels1))+"_clusters.png")


            tsne_res = pd.DataFrame(latent_tsne, columns=['X', 'Y'])
            tsne_res['kmean'] = pd.Series(label1, index=tsne_res.index)
            tsne_res['GMM'] = pd.Series(label2, index=tsne_res.index)
            tsne_res['sampleName'] = pd.Series(
                X_train.tolist().index, index=tsne_res.index)
            tsne_res.to_csv(args.outputpath +
                            "/" + sg_model + "_results/" + "clustering_result_for_" + sg_model + "_" +
                            str(k) + "_clusters.csv")

        fig = plt.figure()
        plt.plot(range(2, 15), silhouette_avg_kmeans)
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette score kmeans')
        plt.title('Silhouette analysis For Optimal k in kmeans')
        plt.savefig(args.outputpath +
                    "/" + sg_model + "_results/"+'Silhouette analysis For Optimal k in kmeans.png')

        fig = plt.figure()
        plt.plot(range(2, 15), silhouette_avg_gm)
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette score gmm')
        plt.title('Silhouette analysis For Optimal k in gmm')
        plt.savefig(args.outputpath +
                    "/" + sg_model + "_results/" + 'Silhouette analysis For Optimal k in gmm.png')

        silhouette_kmeans = max(silhouette_avg_kmeans)
        silhouette_gmm = max(silhouette_avg_gm)
        if silhouette_gmm >= silhouette_kmeans:
            print("Use the Gaussian Mixture Model")
            best_n = silhouette_avg_gm.index(silhouette_gmm) + 2
            print("Number of clusters: " + str(best_n))
        else:
            print("Use the Kmeans")
            best_n = silhouette_avg_kmeans.index(silhouette_kmeans) + 2
            print("Number of clusters: " + str(best_n))

print("The training process was completed successfully")

fout = args.outputpath + "/" + str(datetime.datetime.now()) + "_logdata.txt"
fo = open(fout, "w")
args_dict = dict(vars(args))
for k, v in args_dict.items():
    fo.write(str(k) + ' : ' + str(v) + '\n\n')
fo.close()
