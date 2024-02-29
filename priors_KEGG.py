from bootstrapping import BootstrappingMSigDB
import numpy as np
import os
import pandas as pd
import sys

print('starting bootstrapping for the transcript level')

filesList = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/pathway_libraries/pathway_definitions/KEGG_Human_2021_149_selected/")
targetData = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/all_samples_transcript_level.pkl')
targetData = targetData.astype('float', copy=False)
#filesList = os.listdir('./trial_sigFolder/')
#targetData = pd.read_csv('./trial_sample.csv', index_col=0)
df_mean_direct = pd.DataFrame(columns=targetData.index)
df_sigma_direct = pd.DataFrame(columns=targetData.index)
df_mean_bootstrap = pd.DataFrame(columns=targetData.index)
df_sigma_bootstrap = pd.DataFrame(columns=targetData.index)
df_pvalues_mean = pd.DataFrame(columns=targetData.index)

print("I am about to do the loop",  file=sys.stderr)
pathwayList = list()
for f in filesList:
    abs_f = os.path.join(
        "/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/pathway_libraries/pathway_definitions/KEGG_Human_2021_149_selected/", f)
    #abs_f = os.path.join('./trial_sigFolder/', f)
    #print("Starting my class",  file=sys.stderr)
    bootstrapSample = BootstrappingMSigDB(
        sigData=abs_f,
        targetData=targetData)
    #print("Finish reading the CSV...",  file=sys.stderr)
    sigName, geneList = bootstrapSample.read_signature()
    pathwayList.append(sigName)
    interest_locs = bootstrapSample.zScorePath(
        sigName=sigName, geneList=geneList)
    resTotal = bootstrapSample.bootstrappingProcess(geneList, interest_locs)
    resTotal.index = targetData.index

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
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/transcript_level/direct_mean_KEGG_transcript_level_no_normalization.csv")
df_sigma_direct.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/transcript_level/direct_sigma_KEGG_transcript_level_no_normalization.csv")
df_mean_bootstrap.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/transcript_level/bootstrap_mean_KEGG_transcript_level_no_normalization.csv")
df_sigma_bootstrap.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/transcript_level/bootstrap_sigma_KEGG_transcript_level_no_normalization.csv")
df_pvalues_mean.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/transcript_level/bootstrap_pval_transcript_level_no_normalization.csv")

print('bootstrapping for the transcript level ends here')
print('=============================================================')
print('starting bootstrapping for the gene level')


targetData = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/all_samples_gene_level.pkl')
targetData = targetData.astype('float', copy=False)


df_mean_direct = pd.DataFrame(columns=targetData.index)
df_sigma_direct = pd.DataFrame(columns=targetData.index)
df_mean_bootstrap = pd.DataFrame(columns=targetData.index)
df_sigma_bootstrap = pd.DataFrame(columns=targetData.index)
df_pvalues_mean = pd.DataFrame(columns=targetData.index)

print("I am about to do the loop",  file=sys.stderr)
pathwayList = list()
for f in filesList:
    abs_f = os.path.join(
        "/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/pathway_libraries/pathway_definitions/KEGG_Human_2021_149_selected/", f)
    #abs_f = os.path.join('./trial_sigFolder/', f)
    #print("Starting my class",  file=sys.stderr)
    bootstrapSample = BootstrappingMSigDB(
        sigData=abs_f,
        targetData=targetData)
    #print("Finish reading the CSV...",  file=sys.stderr)
    
    sigName, geneList = bootstrapSample.read_signature()
    pathwayList.append(sigName)
    interest_locs = bootstrapSample.zScorePathGeneVersion(
        sigName=sigName, geneList=geneList)
    resTotal = bootstrapSample.bootstrappingProcess_pd(geneList, interest_locs)
    resTotal.index = targetData.index

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
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/direct_mean_KEGG_gene_level_no_normalization.csv")
df_sigma_direct.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/direct_sigma_KEGG_gene_level_no_normalization.csv")
df_mean_bootstrap.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization.csv")
df_sigma_bootstrap.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_sigma_KEGG_gene_level_no_normalization.csv")
df_pvalues_mean.transpose().to_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_pval_gene_level_no_normalization.csv")


print('bootstrapping for the gene level ends here')
