from bootstrapping import BootstrappingMSigDB
import numpy as np
import os
import pandas as pd
import sys


pathwayList_all = ["/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/pathway_libraries/pathway_definitions/HALLMARK_MSigDB_50_all/",
                   "/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/pathway_libraries/pathway_definitions/KEGG_Human_2021_149_selected/"]

targetList_all = ['/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test_zscore/all_samples_transcript_level_train_zscore.pkl',
                  '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test_zscore/all_samples_transcript_level_test_zscore.pkl',
                  '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_train_zscore.pkl',
                  '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_test_zscore.pkl']

savedir_all = ['/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/transcript_level/',
               '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/transcript_level/',
               '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/',
               '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/',
               '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_zscore/transcript_level/',
               '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_zscore/transcript_level/',
               '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_zscore/gene_level/',
               '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_zscore/gene_level/']
savename_all = ['msigDB_train_transcript', 'msigDB_test_transcript',
                'msigDB_train_gene', 'msigDB_test_gene',
                'KEGG_train_transcript', 'KEGG_test_transcript',
                'KEGG_train_gene', 'KEGG_test_gene']

counter = 0
print('starting bootstrapping')

for pathway in pathwayList_all:
    filesList = os.listdir(pathway)
    for targets in targetList_all:
        targetData = pd.read_pickle(targets)
        targetData = targetData.astype('float', copy=False)
        df_mean_direct = pd.DataFrame(columns=targetData.index)
        df_sigma_direct = pd.DataFrame(columns=targetData.index)
        df_mean_bootstrap = pd.DataFrame(columns=targetData.index)
        df_sigma_bootstrap = pd.DataFrame(columns=targetData.index)
        df_pvalues_mean = pd.DataFrame(columns=targetData.index)

        print("I am about to do the loop",  file=sys.stderr)
        pathwayList = list()
        for f in filesList:
            abs_f = os.path.join(pathway, f)
            bootstrapSample = BootstrappingMSigDB(
                sigData=abs_f,
                targetData=targetData)
            sigName, geneList = bootstrapSample.read_signature()
            pathwayList.append(sigName)
            if 'transcript' in savename_all[counter]:
                interest_locs = bootstrapSample.zScorePath(
                sigName=sigName, geneList=geneList)
                resTotal = bootstrapSample.bootstrappingProcess(geneList, interest_locs)
                resTotal.index = targetData.index
            elif 'gene' in savename_all[counter]:
                interest_locs = bootstrapSample.zScorePathGeneVersion(
                    sigName=sigName, geneList=geneList)
                resTotal = bootstrapSample.bootstrappingProcess_pd(geneList, interest_locs)
                resTotal.index = targetData.index
            else: 
                print("Wrong input level")

            df_mean_direct = df_mean_direct.append(
                resTotal.iloc[:, 0].to_frame().T, ignore_index=True)
            df_sigma_direct = df_sigma_direct.append(
                resTotal.iloc[:, 1].to_frame().T, ignore_index=True)
            df_mean_bootstrap = df_mean_bootstrap.append(
                resTotal.iloc[:, 2].to_frame().T, ignore_index=True)
            df_sigma_bootstrap = df_sigma_bootstrap.append(
                resTotal.iloc[:, 3].to_frame().T, ignore_index=True)
            df_pvalues_mean = df_pvalues_mean.append(
                resTotal.iloc[:, 4].to_frame().T, ignore_index=True)

        df_mean_direct.index = pathwayList
        df_sigma_direct.index = pathwayList
        df_mean_bootstrap.index = pathwayList
        df_sigma_bootstrap.index = pathwayList
        df_pvalues_mean.index = pathwayList

        df_mean_direct.transpose().to_csv(
            savedir_all[counter] + 'direct_mean_' + savename_all[counter] + '_level_zscore.csv')
        df_sigma_direct.transpose().to_csv(
            savedir_all[counter] + 'direct_sigma_' + savename_all[counter] + '_level_zscore.csv')
        df_mean_bootstrap.transpose().to_csv(
            savedir_all[counter] + 'bootstrap_mean_' + savename_all[counter] + '_level_zscore.csv')
        df_sigma_bootstrap.transpose().to_csv(
            savedir_all[counter] + 'bootstrap_sigma_' + savename_all[counter] + '_level_zscore.csv')
        df_pvalues_mean.transpose().to_csv(
            savedir_all[counter] + 'bootstrap_pval_' + savename_all[counter] + '_level_zscore.csv')

        counter += 1
