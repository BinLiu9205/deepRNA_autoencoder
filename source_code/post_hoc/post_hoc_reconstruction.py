import sys
sys.path.insert(
    0, '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models')
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import priorVAE
import simpleAE
import simpleVAE
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
torch.manual_seed(42)
seed = 42
np.random.seed(seed)



f_dir1 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/autoencoder_models")
f_dir2 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/autoencoder_models")
f_dir3 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/autoencoder_models")


dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/autoencoder_models/"
dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/autoencoder_models/"
dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/autoencoder_models/"

dir1_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/"
dir2_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/"
dir3_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/"

add_f1 = "community level + "
add_f2 = "gene level + "
add_f3 = "transcript level + "

all_res = [dir1 + i for i in f_dir1]
all_res = (*all_res, *[dir2 + i for i in f_dir2])
all_res = (*all_res, *
                  [dir3 + i for i in f_dir3])

name_list = [add_f1 + i[0:-4] for i in f_dir1]
name_list = (*name_list, *[add_f2 + i[0:-4] for i in f_dir2])
name_list = (*name_list, *
             [add_f3 + i[0:-4] for i in f_dir3])

model_list = [i[0:-4] for i in f_dir1]
model_list = (*model_list, *[i[0:-4] for i in f_dir2])
model_list = (*model_list, *
             [i[0:-4] for i in f_dir3])


anno1 = 'gene'
anno2 = 'gene'
anno3 = 'transcript'

annoName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno1 + '_level.csv'
annoName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno2 + '_level.csv'
annoName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno3 + '_level.csv'
annoList = [annoName1 for i in range(len(f_dir1))]
annoList = (*annoList, *[annoName2 for i in range(len(f_dir2))])
annoList = (*annoList, *[annoName3 for i in range(len(f_dir3))])

annoName1Sig = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_' + anno1 + '_level.csv'
annoName2Sig = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_' + anno2 + '_level.csv'
annoName3Sig = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_' + anno3 + '_level.csv'
annoListSig = [annoName1Sig for i in range(len(f_dir1))]
annoListSig = (*annoListSig, *[annoName2Sig for i in range(len(f_dir2))])
annoListSig = (*annoListSig, *[annoName3Sig for i in range(len(f_dir3))])

test1 = 'community'
test2 = 'gene'
test3 = 'transcript'

trainName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test1 + '_level_train_test/all_samples_' + test1 + '_level_train.pkl'
trainName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test2 + '_level_train_test/all_samples_' + test2 + '_level_train.pkl'
trainName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test3 + '_level_train_test/all_samples_' + test3 + '_level_train.pkl'
trainList = [trainName1 for i in range(len(f_dir1))]
trainList = (*trainList, *[trainName2 for i in range(len(f_dir2))])
trainList = (*trainList, *[trainName3 for i in range(len(f_dir3))])

testName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test1 + '_level_train_test/all_samples_' + test1 + '_level_test.pkl'
testName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test2 + '_level_train_test/all_samples_' + test2 + '_level_test.pkl'
testName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test3 + '_level_train_test/all_samples_' + test3 + '_level_test.pkl'
testList = [testName1 for i in range(len(f_dir1))]
testList = (*testList, *[testName2 for i in range(len(f_dir2))])
testList = (*testList, *[testName3 for i in range(len(f_dir3))])

# Change this part for saving the plots and numeric results 
main_dir = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/post_hoc_all_samples/no_early_stopping_500_epochs/'
numpy_dir = 'reconstruction'

os.makedirs(os.path.join(dir1_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir2_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir3_root, numpy_dir), exist_ok=True)


save_dir = [(dir1_root + numpy_dir + '/') for i in range(len(f_dir1))]
save_dir = (*save_dir, *[(dir2_root + numpy_dir + '/') for i in range(len(f_dir2))])
save_dir = (*save_dir, *[(dir3_root + numpy_dir + '/') for i in range(len(f_dir3))])

batch_size = 128

counter = 0
current_train = 'init'
current_test = 'init'
current_anno = 'init'
current_annoSig = 'init'
n_features_path = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Initialize the models based on the type of .pth files
for model_name in all_res:
    train_data = trainList[counter]
    test_data = testList[counter]
    anno_file = annoList[counter]
    anno_fileSig = annoListSig[counter]
    if train_data != current_train:
        current_train = train_data
        X_train_ori = pd.read_pickle(train_data)
    if test_data != current_test:
        current_test = test_data
        X_test_ori = pd.read_pickle(test_data)
    if anno_file != current_anno:
        current_anno = anno_file
        anno = pd.read_csv(anno_file, index_col=0)
    if anno_fileSig != current_annoSig:
        current_annoSig = anno_fileSig
        annoSig = pd.read_csv(anno_fileSig, index_col=0)
    
    model_short = model_list[counter]
    n_inputs = X_train_ori.shape[1]
    n_features_trans = X_train_ori.shape[1]
    sample_val, _ = X_test_ori.shape
    colnames_df = X_train_ori.columns
    index_df_train = X_train_ori.index
    index_df_test = X_test_ori.index
    
    print(X_train_ori.shape)
    
    if model_short != 'beta_priorVAE' and model_short != 'priorVAE':
        X_train = X_train_ori.to_numpy()
        X_test = X_test_ori.to_numpy()
        train_set = torch.Tensor(X_train)
        test_set = torch.Tensor(X_test)
        train_set = TensorDataset(train_set, train_set)
        test_set = TensorDataset(test_set, test_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        if model_short == 'simpleAE':
            model = simpleAE.simpleAE(n_inputs = n_inputs).to(device)
            model.load_state_dict(torch.load(model_name), strict=False)
            reconstruct_results = []
            for i in range(sample_val):
                data_tem = test_loader.dataset[i][0]
                output = model(data_tem.reshape(1, -1).to(device))
                reconstruct_results.append(output.cpu().detach().numpy())
            reconstruct_results = np.array(reconstruct_results)
            reconstruct_results = reconstruct_results.reshape(sample_val, n_inputs)
            print("saving result for" + name_list[counter])
            print(reconstruct_results.shape)
            np.save(save_dir[counter] + name_list[counter] + '_reconstruction_results'
             + ".npy", reconstruct_results)


        elif model_short == 'beta_simpleVAE' or model_short == 'simpleVAE':
            model = simpleVAE.simpleVAE(n_inputs=n_inputs).to(device)
            model.load_state_dict(torch.load(model_name), strict=False)
            reconstruct_results = []
            for i in range(sample_val):
                data_tem = test_loader.dataset[i][0]
                output = model(data_tem.reshape(1, -1).to(device))
                reconstruct_results.append(output.cpu().detach().numpy())
            reconstruct_results = np.array(reconstruct_results)
            reconstruct_results = reconstruct_results.reshape(sample_val, n_inputs)
            print("saving result for" + name_list[counter])
            print(reconstruct_results.shape)
            np.save(save_dir[counter] + name_list[counter] + '_reconstruction_results'
             + ".npy", reconstruct_results)
    
    else:
        X_train = X_train_ori
        X_test = X_test_ori
        X_train.columns = colnames_df
        X_train.set_index(index_df_train, inplace=True)
        X_test.columns = colnames_df
        X_test.set_index(index_df_test, inplace=True)
        mu_prior = anno
        sigma_prior = annoSig
        sigma_prior = sigma_prior + 1
        log_sigma_prior = np.log(sigma_prior)
        mu_prior = mu_prior.add_suffix('_mu')
        sigma_prior = sigma_prior.add_suffix('_sigma')
        log_sigma_prior = log_sigma_prior.add_suffix('_logSigma')
        X_train = X_train.join(mu_prior)
        X_train = X_train.join(sigma_prior)
        X_train = X_train.join(log_sigma_prior)
        n_features_path = mu_prior.shape[1]
        X_test = X_test.join(mu_prior)
        X_test = X_test.join(sigma_prior)
        X_test = X_test.join(log_sigma_prior)
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        train_set = torch.Tensor(X_train)
        test_set = torch.Tensor(X_test)
        train_set = TensorDataset(train_set, train_set)
        test_set = TensorDataset(test_set, test_set)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False)
        model = priorVAE.priorVAE(n_inputs=n_inputs, n_features_trans=n_features_trans,
                          n_features_path=n_features_path).to(device)
        model.load_state_dict(torch.load(model_name), strict=False)
        reconstruct_results = []

        for i in range(sample_val):
            data_tem = test_loader.dataset[i][0]
            output = model(data_tem.reshape(1, -1).to(device))
            reconstruct_results.append(output.cpu().detach().numpy())
        reconstruct_results = np.array(reconstruct_results)
        reconstruct_results = reconstruct_results.reshape(sample_val, n_inputs)
        print("saving result for" + name_list[counter])
        print(reconstruct_results.shape)
        np.save(save_dir[counter] + name_list[counter] + '_reconstruction_results'
             + ".npy", reconstruct_results)


    counter +=1
