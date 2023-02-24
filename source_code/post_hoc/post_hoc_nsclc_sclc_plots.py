import sys
sys.path.insert(
    0, '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models')
import priorVAE
import simpleAE
import simpleVAE
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
torch.manual_seed(42)
seed = 42
np.random.seed(seed)
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from pca import pca



annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
annotate_tem_train = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl')
X_train = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_train.pkl')
annotate_tem = annotate_tem.set_index('Source Name')
annotate_tem_train = annotate_tem_train.set_index('Source Name')
annotate_test = annotate_tem.reindex(index=X_test.index)
annotate_train = annotate_tem_train.reindex(index=X_train.index)


nsclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
nsclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
sclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'small cell lung cancer cell line']
sclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'small cell lung cancer cell line']


nsclc_test_loc = [annotate_test.index.get_loc(idx) for idx in nsclc_test]
nsclc_train_loc = [annotate_train.index.get_loc(idx) for idx in nsclc_train]
sclc_test_loc = [annotate_test.index.get_loc(idx) for idx in sclc_test]
sclc_train_loc = [annotate_train.index.get_loc(idx) for idx in sclc_train]


print(len(nsclc_train_loc), len(nsclc_test_loc),len(sclc_train_loc), len(sclc_test_loc))


f_dir1 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/autoencoder_models")
f_dir2 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/autoencoder_models")
f_dir3 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/autoencoder_models")


dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/autoencoder_models/"
dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/autoencoder_models/"
dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/autoencoder_models/"

dir1_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir2_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir3_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/"

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
main_dir = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/post_hoc_all_samples/no_early_stopping_500_epochs_new_sigma_mu/'
numpy_dir = 'lung_cancer'
plot_dir = 'lung_cancer_visualization'
pdf_dir = 'ms_lung_cancer_visualization_pdf'

path = os.path.join(main_dir, plot_dir)
os.makedirs(os.path.join(dir1_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir2_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir3_root, numpy_dir), exist_ok=True)
os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(main_dir,pdf_dir), exist_ok=True)


save_dir = [(dir1_root + numpy_dir + '/') for i in range(len(f_dir1))]
save_dir = (*save_dir, *[(dir2_root + numpy_dir + '/') for i in range(len(f_dir2))])
save_dir = (*save_dir, *[(dir3_root + numpy_dir + '/') for i in range(len(f_dir3))])

batch_size = 128

current_train = 'init'
current_test = 'init'
current_anno = 'init'
current_annoSig = 'init'
n_features_path = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


list_num = [i for i in range(1, 51)]
dimension_label = list(map(lambda x: "dimension_"+str(x), list_num))
anno = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv", index_col=0)
anno_name = list(anno.columns)
anno_name = [single[9:] for single in anno_name]


nsclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
nsclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
sclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'small cell lung cancer cell line']
sclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'small cell lung cancer cell line']


nsclc_test_loc = [annotate_test.index.get_loc(idx) for idx in nsclc_test]
nsclc_train_loc = [annotate_train.index.get_loc(idx) for idx in nsclc_train]
sclc_test_loc = [annotate_test.index.get_loc(idx) for idx in sclc_test]
sclc_train_loc = [annotate_train.index.get_loc(idx) for idx in sclc_train]


n = 20  

index_nsclc = np.random.choice((len(nsclc_test_loc) + len(nsclc_train_loc)), 50, replace=False)  
index_nsclc = index_nsclc[15:35]
# There is always one NSCLC in SCLC, so try another random selection
index_sclc = np.random.choice((len(sclc_test_loc) + len(sclc_train_loc)), n, replace=False)  
health_list = ["adenocarcinoma (NSCLC)" for i in range(20)]
health_list = [*health_list, *["SCLC" for i in range(20)]]


print(str(health_list))


interested_organs = ["adenocarcinoma (NSCLC)", "SCLC"]
color_pa = ["blue", "red"]
color_bar = {"adenocarcinoma (NSCLC)": "blue", "SCLC": "red"}
#plt.rc('figure', figsize=(3.34, 2.36))

## Initialize the models based on the type of .pth files
for counter in range(len(all_res)):
    model_name = all_res[counter]
    print("Model number" + str(counter))
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
    colnames_df = X_train_ori.columns
    index_df_train = X_train_ori.index
    index_df_test = X_test_ori.index
    print(model_short)
    
        
        
    if model_short.find('priorVAE') == -1:
        X_train = X_train_ori.to_numpy()
        X_test = X_test_ori.to_numpy()
        nsclc_all = np.vstack((X_train[nsclc_train_loc, :], X_test[nsclc_test_loc, :]))
        sclc_all = np.vstack((X_train[sclc_train_loc, : ], X_test[sclc_test_loc, :]))
        nsclc_all_set = torch.Tensor(nsclc_all)
        sclc_all_set = torch.Tensor(sclc_all)
        nsclc_all_set = TensorDataset(nsclc_all_set, nsclc_all_set)
        sclc_all_set = TensorDataset(sclc_all_set, sclc_all_set)
        nsclc_loader = DataLoader(
            nsclc_all_set, batch_size=batch_size, shuffle=False)
        sclc_loader = DataLoader(
            sclc_all_set, batch_size=batch_size, shuffle=False)
        print('starting non-prior models')
        if model_short.count('simpleAE') > 0:
            model = simpleAE.simpleAE(n_inputs=n_inputs).to(device)
            model.load_state_dict(torch.load(model_name), strict=False)
            print("Loading my model" + model_name)
            nsclc_mu_results = []
            for i in range(nsclc_all.shape[0]):
                data_tem = nsclc_loader.dataset[i][0]
                output = model.encoder(data_tem.reshape(1, -1).to(device))
                nsclc_mu_results.append(output.cpu().detach().numpy())
            nsclc_mu_results = np.array(nsclc_mu_results)
            nsclc_mu_results = nsclc_mu_results.reshape(nsclc_all.shape[0], 50)
            print("Saving results")
            np.save(save_dir[counter] + name_list[counter] + '_mean_results_nsclc_vae.npy', nsclc_mu_results)
            
            sclc_mu_results = []
            for i in range(sclc_all.shape[0]):
                data_tem = sclc_loader.dataset[i][0]
                output = model.encoder(data_tem.reshape(1, -1).to(device))
                sclc_mu_results.append(output.cpu().detach().numpy())
            sclc_mu_results = np.array(sclc_mu_results)
            sclc_mu_results = sclc_mu_results.reshape(sclc_all.shape[0], 50)
            print("Saving results")
            np.save(save_dir[counter] + name_list[counter] + '_mean_results_sclc_vae.npy', sclc_mu_results)
            

        else:
            model = simpleVAE.simpleVAE(n_inputs=n_inputs).to(device)
            model.load_state_dict(torch.load(model_name), strict=False)
            print("Loading my model" + model_name)
            
            nsclc_mu_results = []
            for i in range(nsclc_all.shape[0]):
                data_tem = nsclc_loader.dataset[i][0]
                output = model.encoder(data_tem.reshape(1, -1).to(device))
                nsclc_mu_results.append(output[0].cpu().detach().numpy())
            nsclc_mu_results = np.array(nsclc_mu_results)
            nsclc_mu_results = nsclc_mu_results.reshape(nsclc_all.shape[0], 50)
            print("Saving results")
            np.save(save_dir[counter] + name_list[counter] + '_mean_results_nsclc_vae.npy', nsclc_mu_results)
            
            sclc_mu_results = []
            for i in range(sclc_all.shape[0]):
                data_tem = sclc_loader.dataset[i][0]
                output = model.encoder(data_tem.reshape(1, -1).to(device))
                sclc_mu_results.append(output[0].cpu().detach().numpy())
            sclc_mu_results = np.array(sclc_mu_results)
            sclc_mu_results = sclc_mu_results.reshape(sclc_all.shape[0], 50)
            print("Saving results")
            np.save(save_dir[counter] + name_list[counter] + '_mean_results_sclc_vae.npy', sclc_mu_results)


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
        nsclc_all = np.vstack((X_train[nsclc_train_loc, :], X_test[nsclc_test_loc, :]))
        sclc_all = np.vstack((X_train[sclc_train_loc, : ], X_test[sclc_test_loc, :]))
        nsclc_all_set = torch.Tensor(nsclc_all)
        sclc_all_set = torch.Tensor(sclc_all)
        nsclc_all_set = TensorDataset(nsclc_all_set, nsclc_all_set)
        sclc_all_set = TensorDataset(sclc_all_set, sclc_all_set)
        nsclc_loader = DataLoader(
            nsclc_all_set, batch_size=batch_size, shuffle=False)
        sclc_loader = DataLoader(
            sclc_all_set, batch_size=batch_size, shuffle=False)
        model = priorVAE.priorVAE(n_inputs=n_inputs, n_features_trans=n_features_trans,
                                  n_features_path=n_features_path).to(device)
        model.load_state_dict(torch.load(model_name), strict=False)
        
        nsclc_mu_results = []
        for i in range(nsclc_all.shape[0]):
            data_tem = nsclc_loader.dataset[i][0]
            output = model.encoder(data_tem.reshape(1, -1).to(device))
            nsclc_mu_results.append(output[0].cpu().detach().numpy())
        nsclc_mu_results = np.array(nsclc_mu_results)
        nsclc_mu_results = nsclc_mu_results.reshape(nsclc_all.shape[0], 50)
        print("Saving results")
        np.save(save_dir[counter] + name_list[counter] + '_mean_results_nsclc_vae.npy', nsclc_mu_results)
        
        sclc_mu_results = []
        for i in range(sclc_all.shape[0]):
            data_tem = sclc_loader.dataset[i][0]
            output = model.encoder(data_tem.reshape(1, -1).to(device))
            sclc_mu_results.append(output[0].cpu().detach().numpy())
        sclc_mu_results = np.array(sclc_mu_results)
        sclc_mu_results = sclc_mu_results.reshape(sclc_all.shape[0], 50)
        print("Saving results")
        np.save(save_dir[counter] + name_list[counter] + '_mean_results_sclc_vae.npy', sclc_mu_results)
        
        
    ## Biplots for 20 NSCLC and 20 SCLC samples 


    nsclc_random = nsclc_mu_results[index_nsclc]
    sclc_random = sclc_mu_results[index_sclc]
    res_selected_all = np.vstack((nsclc_random, sclc_random))
    model_short = name_list[counter]
    print('2 component PCA for ' + model_short)
    pca_model = pca(n_components=2)
    if model_short.find("prior") != -1:
        pca_latent_selected = pca_model.fit_transform(
            res_selected_all, row_labels=health_list, col_labels=anno_name)
    else:
        pca_latent_selected = pca_model.fit_transform(
            res_selected_all, row_labels=health_list, col_labels=dimension_label)
    fig, ax = pca_model.scatter()
    plt.gcf().set_size_inches(6, 5.6)
    plt.tight_layout()
    plt.savefig(main_dir + plot_dir + '/' +
                name_list[counter] + '_pca_scatter_NSCLC_SCLC')
    plt.savefig(main_dir + pdf_dir + '/' +
                name_list[counter] + '_pca_scatter_NSCLC_SCLC.svg')
    plt.close()
    
    fig, ax = pca_model.biplot(n_feat=10)
    plt.gcf().set_size_inches(6, 5.6)
    plt.tight_layout()
    plt.savefig(main_dir + plot_dir +'/' +
                name_list[counter] + "_NSCLC_SCLC_biplots")
    plt.savefig(main_dir + pdf_dir + '/' +
                name_list[counter] + "_NSCLC_SCLC_biplots.svg")
    plt.close()
    
    
    ## PCA plot

    pca_model1 = PCA(n_components=2)
    pca_latent_selected = pca_model1.fit_transform(res_selected_all)
    finalDf = pd.DataFrame(data=pca_latent_selected)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=13)
    ax.set_ylabel('Principal Component 2', fontsize=13)
    ax.set_title('2 component PCA for ' + model_short, fontsize=15)

    for target, color in zip(interested_organs, color_pa):
        indicesToKeep = [index for index in range(
            len(health_list)) if health_list[index] == target]
        plt.scatter(finalDf.loc[indicesToKeep, 0],
                   finalDf.loc[indicesToKeep, 1], c=color, s=25)
    
    ax.grid()
    plt.gcf().set_size_inches(4.5, 3.75)
    plt.tight_layout()
    plt.savefig(main_dir + plot_dir +'/'  +
                name_list[counter] + '_PCA_plots_NSCLC_SCLC')
    print(model_short +": "+ str(pca_model1.explained_variance_ratio_))

    

    if model_short.find("prior") != -1:
        sns.clustermap(np.transpose(res_selected_all), standard_scale=0, col_colors=pd.Series(health_list).map(color_bar).to_numpy(), yticklabels=anno_name, row_cluster=False, col_cluster=False,
                    cmap="RdBu_r")
        plt.gcf().set_size_inches(8, 8.4)
        plt.tight_layout()
        plt.savefig(main_dir + plot_dir +'/' +
                name_list[counter] + "_heatmap_not_clustered")
        plt.savefig(main_dir + pdf_dir + '/' +
                    name_list[counter] + "_heatmap_not_clustered.svg")
        plt.close()

        sns.clustermap(np.transpose(res_selected_all), standard_scale=0, col_colors=pd.Series(health_list).map(color_bar).to_numpy(), yticklabels=anno_name,
                       cmap="RdBu_r")
        plt.gcf().set_size_inches(8, 8.4)
        plt.tight_layout()
        plt.savefig(main_dir + plot_dir +'/' +
                name_list[counter] + "_heatmap_clustered")
        plt.savefig(main_dir + pdf_dir + '/' +
                    name_list[counter] + "_heatmap_clustered.svg")
        plt.close()

    else:
        sns.clustermap(np.transpose(res_selected_all), standard_scale=0, col_colors=pd.Series(health_list).map(color_bar).to_numpy(), yticklabels=dimension_label, row_cluster=False, col_cluster=False,
                    cmap="RdBu_r")
        plt.gcf().set_size_inches(8, 8.4)
        plt.tight_layout()
        plt.savefig(main_dir + plot_dir +'/' +
                name_list[counter] + "_heatmap_not_clustered")
        plt.savefig(main_dir + pdf_dir + '/' +
                    name_list[counter] + "_heatmap_not_clustered.svg")
        plt.close()

        sns.clustermap(np.transpose(res_selected_all), standard_scale=0, col_colors=pd.Series(health_list).map(color_bar).to_numpy(), yticklabels=dimension_label,
                       cmap="RdBu_r")
        plt.gcf().set_size_inches(8, 8.4)
        plt.tight_layout()
        plt.savefig(main_dir + plot_dir +'/' +
                name_list[counter] + "_heatmap_clustered")
        plt.savefig(main_dir + pdf_dir + '/' +
                    name_list[counter] + "_heatmap_clustered.svg")
        plt.close()
    
