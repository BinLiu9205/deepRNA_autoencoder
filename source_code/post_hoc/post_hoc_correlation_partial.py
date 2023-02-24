import seaborn as sns
import scipy.stats
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
seed = 42
np.random.seed(seed)


f_dir1 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/reconstruction")
f_dir2 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/reconstruction")
f_dir3 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/reconstruction")


dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/reconstruction/"
dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/reconstruction/"
dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/reconstruction/"

dir1_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/"
dir2_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/"
dir3_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/"


all_res = [dir1 + i for i in f_dir1]
all_res = (*all_res, *[dir2 + i for i in f_dir2])
all_res = (*all_res, *
                  [dir3 + i for i in f_dir3])

name_list = [ i[0:-27] for i in f_dir1]
name_list = (*name_list, *[i[0:-27] for i in f_dir2])
name_list = (*name_list, *
             [i[0:-27] for i in f_dir3])



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
numpy_dir = 'correlation'
plot_dir = 'correlation_map'
pdf_dir = 'correlation_map_pdf_to_save'

path = os.path.join(main_dir, plot_dir)
os.makedirs(path, exist_ok=True)
path_new= os.path.join(main_dir, pdf_dir)
os.makedirs(path_new, exist_ok=True)
os.makedirs(os.path.join(dir1_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir2_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir3_root, numpy_dir), exist_ok=True)

save_dir = [(dir1_root + numpy_dir + '/') for i in range(len(f_dir1))]
save_dir = (*save_dir, *[(dir2_root + numpy_dir + '/') for i in range(len(f_dir2))])
save_dir = (*save_dir, *[(dir3_root + numpy_dir + '/') for i in range(len(f_dir3))])



model_name = []
input_input_cor = []
input_recons_cor = []
recons_recons_cor = []
input_recons_pair_cor = []
length_trial = 20
train_samples = ['_'.join([str(i+1), 'T']) for i in range(length_trial)]
recon_samples = ['_'.join([str(i+1), 'R']) for i in range(length_trial)]
all_samples = [*train_samples, *recon_samples]

counter = 0
current_test = 'init'
for f in all_res:    
    test_data = testList[counter]
    if test_data != current_test:
        current_test = test_data
        X_test = pd.read_pickle(test_data)
        X_test = X_test.to_numpy()

    
    res = np.load(f)
    model_short = f[0:-len('reconstruction_results.npy')]
    merged_res = np.concatenate((X_test[0:length_trial,:], res[0:length_trial,:]), axis=0)
    cor_res = []

    
    for i in range(2*length_trial):
        for k in range(2*length_trial):
            cor_res.append(scipy.stats.pearsonr(
                merged_res[i, :], merged_res[k, :])[0])
    cor_res_reshape = np.array(cor_res).reshape(2*length_trial, 2*length_trial)
    
    np.save(save_dir[counter] + name_list[counter] + '_correlation_partial'
            + ".npy", cor_res_reshape)


    plt.figure(figsize=(15, 17))
    sns.clustermap(np.transpose(cor_res_reshape),
                   xticklabels=all_samples, yticklabels=all_samples, cmap='YlGnBu')
    plt.title(name_list[counter]+"_scaled")
    plt.savefig(main_dir + plot_dir +'/' + name_list[counter] + '_correlation_sample_partial.png')
    plt.close()
    #plt.figure(figsize=(15, 17))
    sns.clustermap(np.transpose(cor_res_reshape),
                   xticklabels=all_samples, yticklabels=all_samples, cmap='YlGnBu')
    plt.gcf().set_size_inches(8.25,8.15)
    plt.tight_layout()
    plt.savefig(main_dir + pdf_dir + '/' +
                name_list[counter] + '_correlation_sample_partial.pdf')
    plt.close() 


    counter +=1
