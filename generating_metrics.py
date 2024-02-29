import json
import os
import pandas as pd
import numpy as np
from functools import reduce
import re
from collections import Counter
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
import seaborn as sns
import copy
from statistics import mean
seed = 42
np.random.seed(seed)

annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")


X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')
sample_nr = X_test.index
anno_trim = annotate_tem[annotate_tem['Source Name'].isin(sample_nr)]
anno_trim = anno_trim.set_index('Source Name')
anno_trim = anno_trim.loc[sample_nr]
anno_trim = anno_trim.reset_index()


score_metrics = ['accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'neg_log_loss']
file_shortened = os.listdir('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu')




## Group1

normal_index = anno_trim.index[(anno_trim['Characteristics[disease]'] == 'normal') & (
    anno_trim['Characteristics[organism part]'] == 'lung')]
adenocarcinoma_index = anno_trim.index[anno_trim['Characteristics[cell type]']
                                       == 'lung adenocarcinoma cell line']



overall_metrics = {}
overall_metrics['model_name'] = []
for score_metric in score_metrics:
    overall_metrics[score_metric+'_sd'] = []
    overall_metrics[score_metric+'_mean'] = [] 


for file_name in file_shortened:
    res = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu', file_name)
    shortened = file_name[0:-len('_latent_result.npy')]
    overall_metrics['model_name'].append(shortened)
    res = np.load(res)
    res_ordered_cancer = res[adenocarcinoma_index]
    res_ordered_normal = res[normal_index]
    X = np.vstack([res_ordered_cancer, res_ordered_normal])
    y1 = [1] * res_ordered_cancer.shape[0]
    y2 = [0] * res_ordered_normal.shape[0]    
    y = (*y1, *y2)

    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

    scores = cross_validate(logreg, X, y, cv=5, scoring=(
        'accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'neg_log_loss'), return_train_score=True)

    for score_metric in score_metrics:
        overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
        overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())
        

metric_res = pd.DataFrame(overall_metrics)

metric_res.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/lung_cancer_healthy.csv', index=False)

    

## Group2

breast_index = anno_trim.index[(anno_trim['Factor Value[cell type]'] == 'breast cancer cell line') & (anno_trim['Characteristics[disease]'] == 'breast adenocarcinoma')]
lung_index = anno_trim.index[anno_trim['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']


overall_metrics = {}
overall_metrics['model_name'] = []
for score_metric in score_metrics:
    overall_metrics[score_metric+'_sd'] = []
    overall_metrics[score_metric+'_mean'] = [] 


for file_name in file_shortened:
    res = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu', file_name)
    shortened = file_name[0:-len('_latent_result.npy')]
    overall_metrics['model_name'].append(shortened)
    res = np.load(res)
    res_ordered_cancer = res[breast_index]
    res_ordered_normal = res[lung_index]
    X = np.vstack([res_ordered_cancer, res_ordered_normal])
    y1 = [1] * res_ordered_cancer.shape[0]
    y2 = [0] * res_ordered_normal.shape[0]    
    y = (*y1, *y2)

    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

    scores = cross_validate(logreg, X, y, cv=5, scoring=(
        'accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'neg_log_loss'), return_train_score=True)

    for score_metric in score_metrics:
        overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
        overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())
        

metric_res = pd.DataFrame(overall_metrics)

metric_res.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/lung_breast_cancer_phenotype.csv', index=False)

## Group3

cancer_index = anno_trim.index[(anno_trim['Characteristics[organism part]'] == 'bone marrow') & (anno_trim['Characteristics[disease]'] == 'acute myeloid leukaemia')]
healthy_index = anno_trim.index[(anno_trim['Characteristics[organism part]'] == 'bone marrow') & (anno_trim['Characteristics[disease]'] == 'normal')]

overall_metrics = {}
overall_metrics['model_name'] = []
for score_metric in score_metrics:
    overall_metrics[score_metric+'_sd'] = []
    overall_metrics[score_metric+'_mean'] = [] 


for file_name in file_shortened:
    res = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu', file_name)
    shortened = file_name[0:-len('_latent_result.npy')]
    overall_metrics['model_name'].append(shortened)
    res = np.load(res)
    res_ordered_cancer = res[cancer_index]
    res_ordered_normal = res[healthy_index]
    X = np.vstack([res_ordered_cancer, res_ordered_normal])
    y1 = [1] * res_ordered_cancer.shape[0]
    y2 = [0] * res_ordered_normal.shape[0]    
    y = (*y1, *y2)

    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

    scores = cross_validate(logreg, X, y, cv=5, scoring=(
        'accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'neg_log_loss'), return_train_score=True)

    for score_metric in score_metrics:
        overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
        overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())
        

metric_res = pd.DataFrame(overall_metrics)

metric_res.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/leukemia_normal_bone_marrow.csv', index=False)



## Group4

annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl')
annotate_tem = annotate_tem.set_index('Source Name')
annotate_test = annotate_tem.reindex(index=X_test.index)
annotate_test['normalStatus'] = (
    annotate_test['Characteristics[disease]'] == 'normal')
organism = pd.unique(annotate_test['Characteristics[organism part]'])
organism_dis = annotate_test['Characteristics[organism part]'].value_counts()
organism_counts = pd.DataFrame(organism_dis)
organism_counts = organism_counts.reset_index()
organism_counts.columns = ['unique_organs',
                           'counts for sample']  # change column names
interested_organs = organism_counts.iloc[1:9, 0]
organism = pd.unique(annotate_test['Characteristics[organism part]'])
organism_dis = annotate_test['Characteristics[organism part]'].value_counts()
organism_counts = pd.DataFrame(organism_dis)
organism_counts = organism_counts.reset_index()
organism_counts.columns = ['unique_organs',
                           'counts for sample']  # change column names
interested_organs = organism_counts.iloc[1:9, 0]
idxs = []
organ_list = []
for i in interested_organs:
    tem_idx = annotate_test['Characteristics[organism part]'] == i
    tem_anno = annotate_test.index[tem_idx]
    idxs = [*idxs, *tem_anno]
    organ_list_tem = [i] * len(tem_anno)
    #print(organ_list_tem)
    organ_list = [*organ_list, *organ_list_tem]
    
idxs_loc = [annotate_test.index.get_loc(idx) for idx in idxs]

from sklearn.metrics import precision_score, make_scorer
custom_precision_scorer = make_scorer(precision_score, zero_division=0, average='binary')


overall_metrics = {}
overall_metrics['model_name'] = []
for score_metric in score_metrics:
    overall_metrics[score_metric+'_sd'] = []
    overall_metrics[score_metric+'_mean'] = [] 
    
for file_name in file_shortened:
    res = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu', file_name)
    shortened = file_name[0:-len('_latent_result.npy')]
    overall_metrics['model_name'].append(shortened)
    res = np.load(res)
    X = res[idxs_loc, :]
    y_tem = organ_list
    single_accu = []
    single_accu_sd = []
    single_ave_pre = []
    single_ave_pre_sd = []
    single_metric_tem = {}
    for score_metric in score_metrics:
        single_metric_tem[score_metric+'_sd']  = []
        single_metric_tem[score_metric+'_mean']  = []
        
    for k in interested_organs:
        y = copy.deepcopy(y_tem)
        y = [1 if x==k else 0 for x in y_tem]
        logreg = LogisticRegression(solver='lbfgs', max_iter=8000)

        scores = cross_validate(logreg, X, y, cv=5, 
                        scoring={'precision': custom_precision_scorer,
                                 'accuracy': 'accuracy',
                                 'f1': 'f1',
                                 'roc_auc': 'roc_auc',
                                 'average_precision': 'average_precision',
                                 'recall': 'recall', 
                                 'neg_log_loss': 'neg_log_loss'}, return_train_score=True)
        # DO something to iterate over 8 different organs 
        single_accu.append(scores['test_accuracy'].mean())
        single_accu_sd.append(scores['test_accuracy'].std())
        single_ave_pre.append(scores['test_average_precision'].mean())
        single_ave_pre_sd.append(scores['test_average_precision'].std())
        for score_metric in score_metrics:
            single_metric_tem[score_metric+'_sd'].append(scores['test_' + score_metric].std())
            single_metric_tem[score_metric+'_mean'].append(scores['test_' + score_metric].mean())


    for score_metric in score_metrics:
        overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
        overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())
        

metric_res = pd.DataFrame(overall_metrics)

metric_res.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/tissue_information.csv', index=False)


## Group5

overall_metrics = {}
overall_metrics['model_name'] = []
for score_metric in score_metrics:
    overall_metrics[score_metric+'_sd'] = []
    overall_metrics[score_metric+'_mean'] = [] 
    
for file_name in file_shortened:
    #res = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu', file_name)
    shortened = file_name[0:-len('_latent_result.npy')]
    res1 = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu_nsclc', file_name)
    res2 = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu_sclc', file_name)
    overall_metrics['model_name'].append(shortened)
    res_ordered_cancer = np.load(res1)
    res_ordered_normal = np.load(res2)
    X = np.vstack([res_ordered_cancer, res_ordered_normal])
    y1 = [1] * res_ordered_cancer.shape[0]
    y2 = [0] * res_ordered_normal.shape[0]    
    y = (*y1, *y2)

    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

    scores = cross_validate(logreg, X, y, cv=5, scoring=(
        'accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'neg_log_loss'), return_train_score=True)

    for score_metric in score_metrics:
        overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
        overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())
        

metric_res = pd.DataFrame(overall_metrics)

metric_res.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/nsclc_sclc_classifier.csv', index=False)