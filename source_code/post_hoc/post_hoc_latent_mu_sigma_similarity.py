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

add_f1 = "community + "
add_f2 = "gene + "
add_f3 = "transcript + "

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
print(len(name_list))


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
numpy_dir = ['z_test/p_values', 'z_test/z_score', 'correlation/correlation_val' , 'correlation/correlation_pvalue']
description = ['z_score_pval', 'z_score_score', 'correlation_pval', 'correlation_score']
plot_dir = 'prior_and_real_mu'
path = os.path.join(main_dir, plot_dir)
os.makedirs(path, exist_ok=True)

for num in numpy_dir:
    os.makedirs(os.path.join(dir1_root, num), exist_ok=True)
    os.makedirs(os.path.join(dir2_root, num), exist_ok=True)
    os.makedirs(os.path.join(dir3_root, num), exist_ok=True)

save_dir = {}
for num in range(len(numpy_dir)):
    save_dir[description[num]] = [(dir1_root + numpy_dir[num] + '/')
                           for i in range(len(f_dir1_mu))]
    save_dir[description[num]] = (
        *save_dir[description[num]], *[(dir2_root + numpy_dir[num] + '/') for i in range(len(f_dir2_mu))])
    save_dir[description[num]] = (
        *save_dir[description[num]], *[(dir3_root + numpy_dir[num] + '/') for i in range(len(f_dir3_mu))])


anno = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv", index_col=0)
anno_name = list(anno.columns)
anno_name = [single[9:] for single in anno_name]


def Z_test(mu1, mu2,var1, var2):
    # Simulation to get 1000 samples with the known distrubition
    s1 = np.random.normal(mu1, abs(var1), 3000)
    s2 = np.random.normal(mu2, math.sqrt(var2), 3000)
    # Run z test: an example here https://www.statology.org/z-test-python/
    teststat, pval = ztest(s1, s2, value=0)
    return teststat, pval


def correlation_var(mu1, mu2, var1, var2):
    # Simulation to get 15000 samples with the known distrubition
    # Mu1, Var1 -> tested_result
    # Mu2, Var2 -> prior
    s1 = np.random.normal(mu1, abs(var1), 3000)
    s2 = np.random.normal(mu2, math.sqrt(var2), 3000)
    corr, pval_cor = pearsonr(s1, s2)
    return corr, pval_cor


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
    final_teststat = []
    final_pval = []


    if all_res_mu[counter].find('prior') != -1:
        for i in range(mu_prior_test.shape[0]):
            row_corr = []
            row_corr_pval = []
            row_teststat = []
            row_pval = []
            for j in range(mu_prior_test.shape[1]):
                corr, corr_pval = correlation_var(latent_mean[i, j], mu_prior_test.iloc[i, j],
                                                latent_var[i, j], var_prior_test.iloc[i, j])
                row_corr.append(corr)
                row_corr_pval.append(corr_pval)
                teststat, pval = Z_test(latent_mean[i, j], mu_prior_test.iloc[i, j],
                            latent_var[i, j], var_prior_test.iloc[i, j])
                row_teststat.append(teststat)
                row_pval.append(pval)
            final_teststat.append(row_teststat)
            final_pval.append(row_pval)
            final_corr.append(row_corr)
            final_corr_pval.append(row_corr_pval)

        final_corr = np.array(final_corr)
        final_corr_pval = np.array(final_corr_pval)
        final_teststat = np.array(final_teststat)
        final_pval = np.array(final_pval)

        np.save(save_dir[description[3]][counter] + name_list[counter] + "_correlation.npy", final_corr)
        np.save(save_dir[description[2]][counter] + name_list[counter] +
                "_correlation.npy" , final_corr_pval)
        np.save(save_dir[description[1]][counter] +
                name_list[counter] + "_teststat.npy", final_teststat)
        np.save(save_dir[description[0]][counter] +
                name_list[counter] + "_pval.npy", final_pval)
        
        fig_col = 5
        fig_row = 10
        fig = plt.figure(figsize=(25, 18))
        plt.rcParams.update({'font.size': 8})
        plt.subplots_adjust(top=0.8, bottom=0.4)

    # Scatter plot x-prior means, y-latent means, color-z statistics (original value of z-score)
        for i in range(mu_prior_test.shape[1]):
            fig.add_subplot(fig_row, fig_col, (i+1))
            plt.scatter(mu_prior_test.iloc[:, i], latent_mean[:, i],
                        c=final_teststat[:, i], cmap="plasma", s=15, alpha = 0.35)
            plt.title(anno_name[i])

        fig.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                    name_list[counter] + 'z_score_plot.png')
        plt.clf()
        
        fig_col = 5
        fig_row = 10
        fig = plt.figure(figsize=(25, 18))
        plt.rcParams.update({'font.size': 8})
        plt.subplots_adjust(top=0.8, bottom=0.4)

        # Scatter plot x-prior means, y-latent means, color-correlation
        for i in range(mu_prior_test.shape[1]):
            fig.add_subplot(fig_row, fig_col, (i+1))
            plt.scatter(mu_prior_test.iloc[:, i], latent_mean[:, i],
                        c=final_corr[:, i], cmap="plasma", s=15, alpha=0.35)
            plt.title(anno_name[i])

        fig.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                    name_list[counter] + 'correlation_plot.png')
        plt.clf()
        
        for i in range(mu_prior_test.shape[1]):
            fig.add_subplot(fig_row, fig_col, (i+1))
            plt.scatter(mu_prior_test.iloc[:, i], latent_mean[:, i],
                        c="#E64B35", s=12, alpha=0.25)
            plt.title(anno_name[i])
        plt.gcf().set_size_inches(16, 14)
        fig.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                    name_list[counter] + 'overall_correlation_plot_no_mapping.pdf')
        plt.clf()
        
    else:
        for i in range(mu_prior_test.shape[0]):
            row_corr = []
            row_corr_pval = []
            row_teststat = []
            row_pval = []
            for j in range(mu_prior_test.shape[1]):
                corr, corr_pval = correlation_var(latent_mean[i, j], 0,
                                                latent_var[i, j], 1)
                row_corr.append(corr)
                row_corr_pval.append(corr_pval)
                teststat, pval = Z_test(latent_mean[i, j], 0,
                            latent_var[i, j], 1)
                row_teststat.append(teststat)
                row_pval.append(pval)
            final_teststat.append(row_teststat)
            final_pval.append(row_pval)
            final_corr.append(row_corr)
            final_corr_pval.append(row_corr_pval)

        final_corr = np.array(final_corr)
        final_corr_pval = np.array(final_corr_pval)
        final_teststat = np.array(final_teststat)
        final_pval = np.array(final_pval)

        np.save(save_dir[description[3]][counter] + name_list[counter] + "_correlation.npy", final_corr)
        np.save(save_dir[description[2]][counter] + name_list[counter] +
                "_correlation.npy" , final_corr_pval)
        np.save(save_dir[description[1]][counter] +
                name_list[counter] + "_teststat.npy", final_teststat)
        np.save(save_dir[description[0]][counter] +
                name_list[counter] + "_pval.npy", final_pval)


           # Scatter plot x-prior means, y-latent means, color-z statistics (original value of z-score)
        fig_col = 5
        fig_row = 10
        fig = plt.figure(figsize=(25, 18))
        plt.rcParams.update({'font.size': 8})
        plt.subplots_adjust(top=0.8, bottom=0.4)
        for i in range(latent_mean.shape[1]):
            fig.add_subplot(fig_row, fig_col, (i+1))
            plt.hist(latent_mean[:, i])
        fig.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                    name_list[counter] + 'latent_mean_hist.png')
        plt.clf()

        fig_col = 5
        fig_row = 10
        fig = plt.figure(figsize=(25, 18))
        plt.rcParams.update({'font.size': 8})
        plt.subplots_adjust(top=0.8, bottom=0.4)
        for i in range(latent_mean.shape[1]):
            fig.add_subplot(fig_row, fig_col, (i+1))
            plt.hist(final_teststat[:, i])
        fig.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                    name_list[counter] + 'teststat_hist.png')
        plt.clf()

        fig_col = 5
        fig_row = 10
        fig = plt.figure(figsize=(25, 18))
        plt.rcParams.update({'font.size': 8})
        plt.subplots_adjust(top=0.8, bottom=0.4)
        for i in range(latent_mean.shape[1]):
            fig.add_subplot(fig_row, fig_col, (i+1))
            plt.hist(final_corr[:, i])
        fig.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                    name_list[counter] + 'correlation_hist.png')
        plt.clf()
