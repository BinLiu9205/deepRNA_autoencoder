from bootstrapping import BootstrappingMSigDB
import numpy as np
import os
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler


filesList = os.listdir(
    "/mnt/dzl_bioinf/binliu/jupyter/important_supportive_files/HALLMARK/")
trainFiles = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/Training_set_lung_related_filtered_characteristics_transcripts_to_genes.csv", index_col = 0 )
testFiles = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/Testing_set_lung_related_filtered_characteristics_transcripts_to_genes.csv", index_col  = 0)
#t = MinMaxScaler()
#t.fit(trainFiles)
#trainFiles = pd.DataFrame(t.transform(trainFiles.values), columns=trainFiles.columns, index=trainFiles.index)
#testFiles = pd.DataFrame(t.transform(testFiles.values), columns=testFiles.columns, index=testFiles.index)
targetData = pd.concat([trainFiles, testFiles])

df_mean_direct = pd.DataFrame(columns=targetData.index)
df_sigma_direct = pd.DataFrame(columns=targetData.index)
df_mean_bootstrap = pd.DataFrame(columns=targetData.index)
df_sigma_bootstrap = pd.DataFrame(columns=targetData.index)
df_pvalues_mean = pd.DataFrame(columns=targetData.index)

print("I am about to do the loop",  file=sys.stderr)
pathwayList = list()
for f in filesList:
    abs_f = os.path.join(
        "/mnt/dzl_bioinf/binliu/jupyter/important_supportive_files/HALLMARK/", f)
    #abs_f = os.path.join('./trial_sigFolder/', f)
    print("Starting my class",  file=sys.stderr)
    bootstrapSample = BootstrappingMSigDB(
        sigData=abs_f,
        targetData=targetData)
    print("Finish reading the CSV...",  file=sys.stderr)
    sigName, geneList = bootstrapSample.read_signature()
    pathwayList.append(sigName)
    interest_locs = bootstrapSample.zScorePathGeneVersion(
        sigName=sigName, geneList=geneList)
    resTotal = bootstrapSample.bootstrappingProcess_pd(geneList, interest_locs)
    resTotal.index = targetData.index


    df_mean_direct = df_mean_direct.append(resTotal.iloc[:, 0].to_frame().T, ignore_index=True)
    df_sigma_direct = df_sigma_direct.append(resTotal.iloc[:, 1].to_frame().T, ignore_index=True)
    df_mean_bootstrap = df_mean_bootstrap.append(
        resTotal.iloc[:, 2].to_frame().T, ignore_index=True)
    df_sigma_bootstrap = df_sigma_bootstrap.append(
        resTotal.iloc[:, 3].to_frame().T, ignore_index=True)
    df_pvalues_mean = df_pvalues_mean.append(resTotal.iloc[:, 4].to_frame().T, ignore_index=True)

df_mean_direct.index = pathwayList
df_sigma_direct.index = pathwayList
df_mean_bootstrap.index = pathwayList
df_sigma_bootstrap.index = pathwayList
df_pvalues_mean.index = pathwayList    


df_mean_direct.transpose().to_csv("/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/pathway_scoring/prior_distribution_gene/direct_mean_MSigDB.csv")
df_sigma_direct.transpose().to_csv("/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/pathway_scoring/prior_distribution_gene/direct_sigma_MSigDB.csv")
df_mean_bootstrap.transpose().to_csv("/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/pathway_scoring/prior_distribution_gene/bootstrap_mean_MsigDB.csv")
df_sigma_bootstrap.transpose().to_csv("/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/pathway_scoring/prior_distribution_gene/bootstrap_sigma_MsigDB.csv")
df_pvalues_mean.transpose().to_csv("/mnt/dzl_bioinf/binliu/jupyter/newTrial_June/pathway_scoring/prior_distribution_gene/bootstrap_mean_pvals.csv")
