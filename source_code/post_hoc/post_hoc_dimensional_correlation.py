import json
import os
import pandas as pd
import numpy as np
from functools import reduce
import re
from collections import Counter
from statsmodels.stats.weightstats import ztest as ztest
from scipy.stats import pearsonr as pearsonr
import math
import matplotlib.pyplot as plt
seed = 42
np.random.seed(seed)


prior_list = ['priorVAE', 'beta_priorVAE']


f_dir1_mu = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/mean")
f_dir2_mu = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/mean")
f_dir3_mu = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/mean")


dir1_mu = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/mean/"
dir2_mu = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/mean/"
dir3_mu = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/mean/"


f_dir1_var = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/var")
f_dir2_var = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/var")
f_dir3_var = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/var")


dir1_var = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/var/"
dir2_var = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/var/"
dir3_var = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/sigma_mu_VAE/var/"


dir1_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir2_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir3_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/"

add_f1 = "community level + "
add_f2 = "gene level + "
add_f3 = "transcript level + "

all_res_mu = [dir1_mu + i for i in f_dir1_mu]
all_res_mu = (*all_res_mu, *[dir2_mu + i for i in f_dir2_mu])
all_res_mu = (*all_res_mu, *
                  [dir3_mu + i for i in f_dir3_mu])

all_res_var = [dir1_var + i for i in f_dir1_var]
all_res_var = (*all_res_var, *[dir2_var + i for i in f_dir2_var])
all_res_var = (*all_res_var, *
                  [dir3_var + i for i in f_dir3_var])

name_list = [add_f1 + i[0:-len('mean_results_vae.npy')] for i in f_dir1_mu]
name_list = (*name_list, *[add_f2 + i[0:-len('mean_results_vae.npy')] for i in f_dir2_mu])
name_list = (*name_list, *
             [add_f3 + i[0:-len('mean_results_vae.npy')] for i in f_dir3_mu])


anno1 = 'gene'
anno2 = 'gene'
anno3 = 'transcript'

annoName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno1 + '_level.csv'
annoName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno2 + '_level.csv'
annoName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno3 + '_level.csv'
annoList = [annoName1 for i in range(len(f_dir1_mu))]
annoList = (*annoList, *[annoName2 for i in range(len(f_dir2_mu))])
annoList = (*annoList, *[annoName3 for i in range(len(f_dir3_mu))])

annoName1Sig = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_' + anno1 + '_level.csv'
annoName2Sig = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_' + anno2 + '_level.csv'
annoName3Sig = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_' + anno3 + '_level.csv'
annoListSig = [annoName1Sig for i in range(len(f_dir1_mu))]
annoListSig = (*annoListSig, *[annoName2Sig for i in range(len(f_dir2_mu))])
annoListSig = (*annoListSig, *[annoName3Sig for i in range(len(f_dir3_mu))])

test1 = 'community'
test2 = 'gene'
test3 = 'transcript'

trainName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test1 + '_level_train_test/all_samples_' + test1 + '_level_train.pkl'
trainName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test2 + '_level_train_test/all_samples_' + test2 + '_level_train.pkl'
trainName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test3 + '_level_train_test/all_samples_' + test3 + '_level_train.pkl'
trainList = [trainName1 for i in range(len(f_dir1_mu))]
trainList = (*trainList, *[trainName2 for i in range(len(f_dir2_mu))])
trainList = (*trainList, *[trainName3 for i in range(len(f_dir3_mu))])

testName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test1 + '_level_train_test/all_samples_' + test1 + '_level_test.pkl'
testName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test2 + '_level_train_test/all_samples_' + test2 + '_level_test.pkl'
testName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + test3 + '_level_train_test/all_samples_' + test3 + '_level_test.pkl'
testList = [testName1 for i in range(len(f_dir1_mu))]
testList = (*testList, *[testName2 for i in range(len(f_dir2_mu))])
testList = (*testList, *[testName3 for i in range(len(f_dir3_mu))])

# Change this part for saving the plots and numeric results 
main_dir = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/post_hoc_all_samples/no_early_stopping_500_epochs_new_sigma_mu/'
#numpy_dir = ['mean_correlation/pvalue', 'mean_correlation/corr', 'sigma_correlation/pvalue' , 'sigma_correlation/corr']
#description = ['mean_correlation_pval', 'mean_correlation_corr', 'sigma_correlation_pval', 'sigma_correlation_corr']
numpy_dir = "dimensional_correlation"
plot_dir = 'prior_and_real_mu_dimensional_correlation'
pdf_dir = 'ms_prior_real_mu_dimensional_correlation_pdf'
path = os.path.join(main_dir, plot_dir)
os.makedirs(path, exist_ok=True)
path_new = os.path.join(main_dir, pdf_dir)
os.makedirs(path_new, exist_ok=True)

os.makedirs(os.path.join(dir1_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir2_root, numpy_dir), exist_ok=True)
os.makedirs(os.path.join(dir3_root, numpy_dir), exist_ok=True)

save_dir = [dir1_root + numpy_dir + '/' for i in range(len(f_dir1_mu))]
save_dir = (*save_dir, *[dir2_root + numpy_dir +
            '/' for i in range(len(f_dir2_mu))])
save_dir = (*save_dir, *[dir3_root + numpy_dir +
            '/' for i in range(len(f_dir3_mu))])



anno = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv", index_col=0)
anno_name = list(anno.columns)
anno_name = [single[9:] for single in anno_name]


for counter in range(len(all_res_mu)):
    latent_mean = np.load(all_res_mu[counter])
    latent_var = np.load(all_res_var[counter])
    test_shape = latent_mean.shape[0]

    mu_prior = pd.read_csv(annoList[counter], index_col=0)
    var_prior = pd.read_csv(
        annoListSig[counter], index_col=0)

    X_test = pd.read_pickle(testList[counter])
    index_df_test = X_test.index
    transcripts_nr = X_test.shape[1]
    colnames_df = X_test.columns
    X_test.columns = colnames_df
    X_test.set_index(index_df_test, inplace=True)
    pathway_nr = mu_prior.shape[1]
    mu_prior = mu_prior.add_suffix('_mu')
    var_prior = var_prior.add_suffix('_sigma')
    X_test = X_test.join(mu_prior)
    X_test = X_test.join(var_prior)

    mu_prior_test = X_test.iloc[:, transcripts_nr:(transcripts_nr+pathway_nr)]
    var_prior_test = X_test.iloc[:, (transcripts_nr+pathway_nr):]

    # print(mu_prior_test.shape)
    # print(latent_mean.shape)

    final_corr = []
    final_corr_pval = []
    final_corr_sigma = []
    final_corr_sigma_pval = []
    
    save_all_res_mean_correlation = {}
    save_all_res_mean_correlation_pvalue = {}
    save_all_res_sigma_correlation = {}
    save_all_res_sigma_correlation_pvalue = {}


    if all_res_mu[counter].find('prior') != -1:

        for j in range(mu_prior_test.shape[1]):
            corr, corr_pval = pearsonr(latent_mean[:, j], mu_prior_test.iloc[:, j])
            corr_sigma, corr_sigma_pval = pearsonr(latent_var[:, j], var_prior_test.iloc[:, j])

            final_corr.append(corr)
            final_corr_pval.append(corr_pval)
            final_corr_sigma.append(corr_sigma)
            final_corr_sigma_pval.append(corr_sigma_pval)
            
        save_all_res_mean_correlation[name_list[counter]] = final_corr
        save_all_res_mean_correlation_pvalue[name_list[counter]] = final_corr_pval
        save_all_res_sigma_correlation[name_list[counter]] = final_corr_sigma
        save_all_res_sigma_correlation_pvalue[name_list[counter]] = final_corr_sigma_pval

    # Scatter plot x-prior means, y-latent means, color-z statistics (original value of z-score)
    
        for i in range(len(final_corr)):
            unsorted_list = [(pval, path_name) for pval, path_name in
                            zip(final_corr, anno_name)]
            sorted_list = sorted(unsorted_list)


            pvalue_sorted = []
            path_sorted = []

            for j in sorted_list:
                pvalue_sorted += [j[0]]
                path_sorted += [j[1]]
                
        fig = plt.figure(figsize=(25, 10.5))

        plt.title("Mu highly correlated pathways " + name_list[counter], fontsize=15)
        plt.barh(path_sorted, pvalue_sorted)
        #print(result)
        plt.xticks(rotation = 270)
        plt.savefig(main_dir + plot_dir + '/' +
        name_list[counter] + 'mu_correlation.png')
        plt.close()
        
        #fig = plt.figure(figsize=(25, 10.5))
        plt.barh(path_sorted, pvalue_sorted)
        #print(result)
        plt.gcf().set_size_inches(5.65, 6.85 )
        plt.xticks(rotation=270)
        plt.tight_layout()
        plt.savefig(main_dir + pdf_dir + '/' +
                    name_list[counter] + 'mu_correlation.pdf')
        plt.close()
        
        plt.bar(path_sorted, pvalue_sorted)
        #print(result)
        plt.gcf().set_size_inches(6.5 , 5.55)
        plt.xticks(rotation=270)
        plt.tight_layout()
        plt.savefig(main_dir + pdf_dir + '/' +
                    name_list[counter] + 'mu_correlation_vertical.pdf')
        plt.close()
        
        
        print(name_list[counter] + ":")
        print(sorted_list)
        
        for i in range(len(final_corr_sigma)):
            unsorted_list = [(pval, path_name) for pval, path_name in
                            zip(final_corr_sigma, anno_name)]
            sorted_list = sorted(unsorted_list)


            pvalue_sorted = []
            path_sorted = []

            for j in sorted_list:
                pvalue_sorted += [j[0]]
                path_sorted += [j[1]]
                
        fig = plt.figure(figsize=(25, 10.5))
        

        plt.title("Sigma highly correlated pathways " + name_list[counter], fontsize=15)

        plt.barh(path_sorted, pvalue_sorted)
        #print(result)
        plt.xticks(rotation = 270)
        plt.savefig(main_dir + plot_dir + '/' +
        name_list[counter] + 'sigma_correlation.png')
        plt.close()
        
        fig = plt.figure(figsize=(25, 10.5))
        plt.barh(path_sorted, pvalue_sorted)
        #print(result)
        plt.xticks(rotation=270)
        plt.savefig(main_dir + pdf_dir + '/' +
                    name_list[counter] + 'sigma_correlation.pdf')
        plt.close()
        
        
    else:
        continue
    # We can't do the correlation to the unit Gaussian simple VAE since the sd is zero
    
    all_corr_mean = np.array(save_all_res_mean_correlation)
    all_corr_pval_mean = np.array(save_all_res_mean_correlation_pvalue)
    all_corr_mu = np.array(save_all_res_sigma_correlation)
    all_corr_pval_mu = np.array(save_all_res_sigma_correlation_pvalue)
    
    np.save(save_dir[counter] + name_list[counter] +
            "_mean_correlation.npy", final_corr)
    np.save(save_dir[counter] + name_list[counter] +
            "_mean_correlation_pvalue.npy", final_corr_pval)
    np.save(save_dir[counter] + name_list[counter] +
            "_sigma_correlation.npy", final_corr_sigma)
    np.save(save_dir[counter] + name_list[counter] +
            "_sigma_correlation_pvalue.npy", final_corr_sigma_pval)
