# %%
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


# %%
annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
#f_dir1 = os.listdir(
#    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")
#f_dir2 = os.listdir(
#    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")
#f_dir3 = os.listdir(
#    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")

f_dir1 = ['simpleAE.npy', 'simpleVAE.npy',
          'priorVAE.npy', 'beta_simpleVAE.npy',  'beta_priorVAE.npy']
f_dir2 = f_dir1
f_dir3 = f_dir1

dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"
dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"
dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"

dir1_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir2_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir3_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/"

level_name = ['community level', 'gene level', 'transcript level']


# %%
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

anno1 = 'gene'
anno2 = 'gene'
anno3 = 'transcript'

model_list = [i[0:-4] for i in f_dir1]
model_list = (*model_list, *[i[0:-4] for i in f_dir2])
model_list = (*model_list, *
              [i[0:-4] for i in f_dir3])

annoName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno1 + '_level.csv'
annoName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno2 + '_level.csv'
annoName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno3 + '_level.csv'
annoList = [annoName1 for i in range(len(f_dir1))]
annoList = (*annoList, *[annoName2 for i in range(len(f_dir2))])
annoList = (*annoList, *[annoName3 for i in range(len(f_dir3))])

test1 = 'community'
test2 = 'gene'
test3 = 'transcript'

trainName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test1 + '_level_train_test/all_samples_' + test1 + '_level_train.pkl'
trainName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test2 + '_level_train_test/all_samples_' + test2 + '_level_train.pkl'
trainName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test3 + '_level_train_test/all_samples_' + test3 + '_level_train.pkl'
trainList = [trainName1 for i in range(len(f_dir1))]
trainList = (*trainList, *[trainName2 for i in range(len(f_dir2))])
trainList = (*trainList, *[trainName3 for i in range(len(f_dir3))])

testName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test1 + '_level_train_test/all_samples_' + test1 + '_level_test.pkl'
testName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test2 + '_level_train_test/all_samples_' + test2 + '_level_test.pkl'
testName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test3 + '_level_train_test/all_samples_' + test3 + '_level_test.pkl'
testList = [testName1 for i in range(len(f_dir1))]
testList = (*testList, *[testName2 for i in range(len(f_dir2))])
testList = (*testList, *[testName3 for i in range(len(f_dir3))])

# Change this part for saving the plots and numeric results
main_dir = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/post_hoc_all_samples/no_early_stopping_500_epochs_new_sigma_mu/'
plot_dir = 'cancer_accuracy'
pdf_dir = 'criterion_pdf'

path = os.path.join(main_dir, plot_dir)
os.makedirs(path, exist_ok=True)

path_new = os.path.join(main_dir, pdf_dir)
os.makedirs(path_new, exist_ok=True)

X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')
sample_nr = X_test.index
anno_trim = annotate_tem[annotate_tem['Source Name'].isin(sample_nr)]
anno_trim = anno_trim.set_index('Source Name')
anno_trim = anno_trim.loc[sample_nr]
anno_trim = anno_trim.reset_index()
normal_index = anno_trim.index[(anno_trim['Characteristics[disease]'] == 'normal') & (
    anno_trim['Characteristics[organism part]'] == 'lung')]
adenocarcinoma_index = anno_trim.index[anno_trim['Characteristics[cell type]']
                                       == 'lung adenocarcinoma cell line']


list_num = [i for i in range(1, 51)]
dimension_label = list(map(lambda x: "dimension_"+str(x), list_num))
anno = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv", index_col=0)
anno_name = list(anno.columns)
anno_name = [single[9:] for single in anno_name]


# %%
model_list_all = ['simpleAE', 'simpleVAE',
                  'priorVAE', 'beta_simpleVAE',  'beta_priorVAE']


# %%
latent_fig_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics'
os.makedirs(latent_fig_save_path_base, exist_ok=True)
latent_fig_save_path_adeno_health = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/adenocarcinoma_vs_health'
os.makedirs(latent_fig_save_path_adeno_health, exist_ok=True)
latent_fig_save_path_organ = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/organ'
os.makedirs(latent_fig_save_path_organ, exist_ok=True)
latent_fig_save_path_adeno_sclc = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/adeno_vs_sclc'
os.makedirs(latent_fig_save_path_adeno_sclc, exist_ok=True)
latent_fig_save_path_lung_breast = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/adeno_lung_breast'
os.makedirs(latent_fig_save_path_lung_breast, exist_ok=True)
latent_fig_save_path_leukeamia_health = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/leukeamia_health'
os.makedirs(latent_fig_save_path_leukeamia_health, exist_ok=True)

# %%
score_metrics = ['accuracy', 'f1', 'roc_auc', 'average_precision', 'precision', 'recall', 'neg_log_loss']

# %%
def plot_metrics(metrics_array_main, metrics_array_errbar, model_array, label, metric_name, save_path, index1, index2, width = 10, height = 8, ylim_lower = 0.8, ylim_upper = 1):
      plt.figure(figsize=(width, height))
      plt.bar(model_array[index1:index2],
            metrics_array_main[index1:index2])
      plt.xlabel('Model Name')
      plt.ylabel(metric_name)
      plt.errorbar(model_array[index1:index2],
            metrics_array_main[index1:index2], metrics_array_errbar[index1:index2], fmt='.', color='Black', elinewidth=2,
                  capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
      plt.ylim(ylim_lower, ylim_upper)
      plt.xticks(rotation=270)
      plt.gcf().set_size_inches(4.2, 3.8)
      plt.tight_layout()
      plt.savefig(os.path.join(save_path,
                  label + '_' + metric_name + '.eps'))
      plt.show()
      plt.close()

# %% [markdown]
# Adenocarcinoma vs healthy

# %%
short_model = []
stat_res = []
pval_res = []
counter = 0
len_model = 5
i = 0
all_accu = []
all_accu_sd = []
ave_pre = []
ave_pre_sd = []
overall_metrics = {}
for score_metric in score_metrics:
      overall_metrics[score_metric+'_sd'] = []
      overall_metrics[score_metric+'_mean'] = [] 


for res in all_res:
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

      # print("Train Accuracy: %0.2f (+/- %0.2f)" %
      #       (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
      # print("Test Accuracy: %0.2f (+/- %0.2f)" %
      #       (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))

      # print("Train Average Precision: %0.2f (+/- %0.2f)" %
      #       (scores['train_average_precision'].mean(), scores['train_average_precision'].std() * 2))
      # print("Test Average Precision: %0.2f (+/- %0.2f)" %
      #       (scores['test_average_precision'].mean(), scores['test_average_precision'].std() * 2))
      
      for score_metric in score_metrics:
            overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
            overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())

      short_model.append(model_list[counter])
      all_accu.append(scores['test_accuracy'].mean())
      all_accu_sd.append(scores['test_accuracy'].std())
      ave_pre.append(scores['test_average_precision'].mean())
      ave_pre_sd.append(scores['test_average_precision'].std())
      
      counter +=1 
      
      
for i in range(3):
      width = 10
      height = 8
      plt.figure(figsize=(width, height))
      plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
            all_accu[len_model*i + 0:len_model*(i+1)])
      plt.xlabel('Model Name')
      plt.ylabel('Accuracy')
      plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], all_accu[len_model*i + 0:len_model*(i+1)], all_accu_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                  capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
      plt.ylim(0.8, 1)
      plt.xticks(rotation=270)
      plt.gcf().set_size_inches(4.2, 3.8)
      plt.tight_layout()
      plt.savefig(main_dir + plot_dir + '/' +
                  level_name[i] + 'accuracy_plot.png')
      plt.savefig(main_dir + pdf_dir + '/' +
                  level_name[i] + 'accuracy_plot.pdf')
      plt.show()


      width = 10
      height = 8
      plt.figure(figsize=(width, height))
      plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
              ave_pre[len_model*i + 0:len_model*(i+1)])
      plt.xlabel('Model Name')
      plt.ylabel('Average Precision Score')
      plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], ave_pre[len_model*i + 0:len_model*(i+1)], ave_pre_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                   capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
      plt.ylim(0.8, 1)
      plt.xticks(rotation=270)
      plt.gcf().set_size_inches(4.2, 3.8)
      plt.tight_layout()
      plt.savefig(main_dir + plot_dir + '/' +
                  level_name[i] + 'average_precision_plot.png')
      plt.savefig(main_dir + pdf_dir + '/' +
                  level_name[i] + 'average_precision_plot.pdf')
      plt.show()

      print("Models:")
      print(str(short_model[len_model*i + 0:len_model*(i+1)]))
      print("Accuracy mean:")
      print(str(all_accu[len_model*i + 0:len_model*(i+1)]))
      print("Accuracy sd:")
      print(str(all_accu_sd[len_model*i + 0:len_model*(i+1)]))
      print("Average precision mean:")
      print(str(ave_pre[len_model*i + 0:len_model*(i+1)]))
      print("Average precision sd:")
      print(str(ave_pre_sd[len_model*i + 0:len_model*(i+1)]))
      
      
for i in range(3):
      for score_metric in score_metrics:
            if min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])< 0.8:
                  ylim_lower = min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)]) - 0.1
            else:
                  ylim_lower = 0.8
                  
            if max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])> 1:
                  ylim_upper = max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])  + 0.1
                  
            else: 
                  ylim_upper = 1
                
            plot_metrics(metrics_array_main = overall_metrics[score_metric+'_mean'], metrics_array_errbar = overall_metrics[score_metric+'_sd'], model_array = short_model, label = level_name[i], metric_name = score_metric, save_path = latent_fig_save_path_adeno_health, index1 = len_model*i, index2 = len_model*(i+1), width = 10, height = 8, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
            


# %% [markdown]
# Organ

# %%
annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl')
annotate_tem = annotate_tem.set_index('Source Name')
annotate_test = annotate_tem.reindex(index=X_test.index)
annotate_test['normalStatus'] = (
    annotate_test['Characteristics[disease]'] == 'normal')
colors = {True: 'blue', False: 'red'}


# %%
organism = pd.unique(annotate_test['Characteristics[organism part]'])
organism_dis = annotate_test['Characteristics[organism part]'].value_counts()
organism_counts = pd.DataFrame(organism_dis)
organism_counts = organism_counts.reset_index()
organism_counts.columns = ['unique_organs',
                           'counts for sample']  # change column names
interested_organs = organism_counts.iloc[1:9, 0]


# %%
organism = pd.unique(annotate_test['Characteristics[organism part]'])
organism_dis = annotate_test['Characteristics[organism part]'].value_counts()
organism_counts = pd.DataFrame(organism_dis)
organism_counts = organism_counts.reset_index()
organism_counts.columns = ['unique_organs',
                           'counts for sample']  # change column names
interested_organs = organism_counts.iloc[1:9, 0]


# %%
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


# %%
short_model = []
stat_res = []
pval_res = []
counter = 0
len_model = 5
i = 0
all_accu = []
all_accu_sd = []
ave_pre = []
ave_pre_sd = []
overall_metrics = {}
for score_metric in score_metrics:
      overall_metrics[score_metric+'_sd'] = []
      overall_metrics[score_metric+'_mean'] = [] 


from sklearn.metrics import precision_score, make_scorer
custom_precision_scorer = make_scorer(precision_score, zero_division=0, average='binary')



for latent in all_res:
    res = np.load(latent)
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

    short_model.append(model_list[counter])
    all_accu.append(mean(single_accu))
    all_accu_sd.append(mean(single_accu_sd))
    ave_pre.append(mean(single_ave_pre))
    ave_pre_sd.append(mean(single_ave_pre_sd))

    for score_metric in score_metrics:
        overall_metrics[score_metric+'_sd'].append(mean(single_metric_tem[score_metric+'_sd']))
        overall_metrics[score_metric+'_mean'].append(mean(single_metric_tem[score_metric+'_mean']))
    

    counter += 1


# %%
for i in range(3):
        width = 10
        height = 8
        plt.figure(figsize=(width, height))
        plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
                all_accu[len_model*i + 0:len_model*(i+1)])
        plt.xlabel('Model Name')
        plt.ylabel('Accuracy')
        plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], all_accu[len_model*i + 0:len_model*(i+1)], all_accu_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                        capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
        #plt.ylim(0.8, 1)
        plt.xticks(rotation=270)
        plt.gcf().set_size_inches(4.2, 3.8)
        plt.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                level_name[i] + 'accuracy_plot_organs.png')
        plt.savefig(main_dir + pdf_dir + '/' +
                    level_name[i] + 'accuracy_plot_organs.pdf')
        plt.show()

        width = 10
        height = 8
        plt.figure(figsize=(width, height))
        plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
                ave_pre[len_model*i + 0:len_model*(i+1)])
        plt.xlabel('Model Name')
        plt.ylabel('Average Precision Score')
        plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], ave_pre[len_model*i + 0:len_model*(i+1)], ave_pre_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                        capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
        #plt.ylim(0.8, 1)
        plt.xticks(rotation=270)
        plt.gcf().set_size_inches(4.2, 3.8)
        plt.tight_layout()
        plt.savefig(main_dir + plot_dir + '/' +
                level_name[i] + 'average_precision_plot_organs.png')
        plt.savefig(main_dir + pdf_dir + '/' +
                    level_name[i] + 'average_precision_plot_organs.pdf')
        plt.show()
    
        print("Models:")
        print(str(short_model[len_model*i + 0:len_model*(i+1)]))
        print("Accuracy mean:")
        print(str(all_accu[len_model*i + 0:len_model*(i+1)]))
        print("Accuracy sd:")
        print(str(all_accu_sd[len_model*i + 0:len_model*(i+1)]))
        print("Average precision mean:")
        print(str(ave_pre[len_model*i + 0:len_model*(i+1)]))
        print("Average precision sd:")
        print(str(ave_pre_sd[len_model*i + 0:len_model*(i+1)]))
        

for i in range(3):
    for score_metric in score_metrics:
        if min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])< 0.8:
                ylim_lower = min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)]) - 0.1
        else:
                ylim_lower = 0.8
                
        if max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])> 1:
                ylim_upper = max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)]) + 0.1
                
        else: 
                ylim_upper = 1
                  
        plot_metrics(metrics_array_main = overall_metrics[score_metric+'_mean'], metrics_array_errbar = overall_metrics[score_metric+'_sd'], model_array = short_model, label = level_name[i], metric_name = score_metric, save_path = latent_fig_save_path_organ, index1 = len_model*i, index2 = len_model*(i+1), width = 10, height = 8, ylim_lower=ylim_lower, ylim_upper=ylim_upper)


# %% [markdown]
# NSCLC vs SCLC

# %%
dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/lung_cancer/"
dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/lung_cancer/"
dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/lung_cancer/"

add_f1 = "community level + "
add_f2 = "gene level + "
add_f3 = "transcript level + "

model_list_all = ['simpleAE', 'simpleVAE',
                  'priorVAE', 'beta_simpleVAE',  'beta_priorVAE']

all_res_1 = [dir1 + add_f1 + i + "_mean_results_nsclc_vae.npy" for i in model_list_all]
all_res_1 = (*all_res_1, *[dir2 + add_f2 + i +
             "_mean_results_nsclc_vae.npy" for i in model_list_all])
all_res_1 = (*all_res_1, *
             [dir3 + add_f3 + i +
              "_mean_results_nsclc_vae.npy" for i in model_list_all])

all_res_2 = [dir1 + add_f1 + i + "_mean_results_sclc_vae.npy" for i in model_list_all]
all_res_2 = (*all_res_2, *[dir2 + add_f2 + i +
             "_mean_results_sclc_vae.npy" for i in model_list_all])
all_res_2 = (*all_res_2, *
             [dir3 + add_f3 + i +
              "_mean_results_sclc_vae.npy" for i in model_list_all])


# %%

short_model = []
stat_res = []
pval_res = []
counter = 0
len_model = 5
i = 0
all_accu = []
all_accu_sd = []
ave_pre = []
ave_pre_sd = []
for score_metric in score_metrics:
      overall_metrics[score_metric+'_sd'] = []
      overall_metrics[score_metric+'_mean'] = [] 

for counter in range(len(all_res_1)):
    res_ordered_sclc = np.load(all_res_2[counter])
    res_ordered_nsclc = np.load(all_res_1[counter])
    X = np.vstack([res_ordered_sclc, res_ordered_nsclc])
    y1 = [1] * res_ordered_sclc.shape[0]
    y2 = [0] * res_ordered_nsclc.shape[0]
    y = (*y1, *y2)

    logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

    scores = cross_validate(logreg, X, y, cv=5, scoring={'precision': custom_precision_scorer,
                                 'accuracy': 'accuracy',
                                 'f1': 'f1',
                                 'roc_auc': 'roc_auc',
                                 'average_precision': 'average_precision',
                                 'recall': 'recall', 
                                 'neg_log_loss': 'neg_log_loss'}, return_train_score=True)

    # print("Train Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
    # print("Test Accuracy: %0.2f (+/- %0.2f)" %
    #       (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))

    # print("Train Average Precision: %0.2f (+/- %0.2f)" %
    #       (scores['train_average_precision'].mean(), scores['train_average_precision'].std() * 2))
    # print("Test Average Precision: %0.2f (+/- %0.2f)" %
    #       (scores['test_average_precision'].mean(), scores['test_average_precision'].std() * 2))

    short_model.append(model_list[counter])
    all_accu.append(scores['test_accuracy'].mean())
    all_accu_sd.append(scores['test_accuracy'].std())
    ave_pre.append(scores['test_average_precision'].mean())
    ave_pre_sd.append(scores['test_average_precision'].std())
    for score_metric in score_metrics:
        overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
        overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())
    counter += 1


for i in range(3):
    width = 10
    height = 8
    plt.figure(figsize=(width, height))
    plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
            all_accu[len_model*i + 0:len_model*(i+1)])
    plt.xlabel('Model Name')
    plt.ylabel('Accuracy')
    plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], all_accu[len_model*i + 0:len_model*(i+1)], all_accu_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                 capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
    plt.xticks(rotation=270)
    plt.gcf().set_size_inches(4.2, 3.8)
    plt.tight_layout()
    plt.savefig(main_dir + plot_dir + '/' +
                level_name[i] + 'accuracy_plot_sclc_nsclc.png')
    plt.savefig(main_dir + pdf_dir + '/' +
                level_name[i] + 'accuracy_plot_sclc_nsclc.pdf')
    plt.show()

    width = 10
    height = 8
    plt.figure(figsize=(width, height))
    plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
            ave_pre[len_model*i + 0:len_model*(i+1)])
    plt.xlabel('Model Name')
    plt.ylabel('Average Precision Score')
    plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], ave_pre[len_model*i + 0:len_model*(i+1)], ave_pre_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                 capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
    plt.xticks(rotation=270)
    plt.gcf().set_size_inches(4.2, 3.8)
    plt.tight_layout()
    plt.savefig(main_dir + plot_dir + '/' +
                level_name[i] + 'average_precision_plot_sclc_nsclc.png')
    plt.savefig(main_dir + pdf_dir + '/' +
                level_name[i] + 'average_precision_plot_sclc_nsclc.pdf')
    plt.show()

    print("Models:")
    print(str(short_model[len_model*i + 0:len_model*(i+1)]))
    print("Accuracy mean:")
    print(str(all_accu[len_model*i + 0:len_model*(i+1)]))
    print("Accuracy sd:")
    print(str(all_accu_sd[len_model*i + 0:len_model*(i+1)]))
    print("Average precision mean:")
    print(str(ave_pre[len_model*i + 0:len_model*(i+1)]))
    print("Average precision sd:")
    print(str(ave_pre_sd[len_model*i + 0:len_model*(i+1)]))
    
    
for i in range(3):
    for score_metric in score_metrics:
        if min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])< 0.8:
                ylim_lower = min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)]) - 0.1
        else:
                ylim_lower = 0.8
                
        if max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])> 1:
                ylim_upper = max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)]) + 0.1
                
        else: 
                ylim_upper = 1
                
        plot_metrics(metrics_array_main = overall_metrics[score_metric+'_mean'], metrics_array_errbar = overall_metrics[score_metric+'_sd'], model_array = short_model, label = level_name[i], metric_name = score_metric, save_path = latent_fig_save_path_adeno_sclc, index1 = len_model*i, index2 = len_model*(i+1), width = 10, height = 8, ylim_lower=ylim_lower, ylim_upper=ylim_upper)



# %% [markdown]
# Lung vs breast Cancer

# %%
breast_index = anno_trim.index[(anno_trim['Factor Value[cell type]'] == 'breast cancer cell line') & (anno_trim['Characteristics[disease]'] == 'breast adenocarcinoma')]
lung_index = anno_trim.index[anno_trim['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']

# %%
short_model = []
stat_res = []
pval_res = []
counter = 0
len_model = 5
i = 0
all_accu = []
all_accu_sd = []
ave_pre = []
ave_pre_sd = []
overall_metrics = {}
for score_metric in score_metrics:
      overall_metrics[score_metric+'_sd'] = []
      overall_metrics[score_metric+'_mean'] = [] 


for res in all_res:
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

      # print("Train Accuracy: %0.2f (+/- %0.2f)" %
      #       (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
      # print("Test Accuracy: %0.2f (+/- %0.2f)" %
      #       (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))

      # print("Train Average Precision: %0.2f (+/- %0.2f)" %
      #       (scores['train_average_precision'].mean(), scores['train_average_precision'].std() * 2))
      # print("Test Average Precision: %0.2f (+/- %0.2f)" %
      #       (scores['test_average_precision'].mean(), scores['test_average_precision'].std() * 2))
      
      for score_metric in score_metrics:
            overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
            overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())

      short_model.append(model_list[counter])
      all_accu.append(scores['test_accuracy'].mean())
      all_accu_sd.append(scores['test_accuracy'].std())
      ave_pre.append(scores['test_average_precision'].mean())
      ave_pre_sd.append(scores['test_average_precision'].std())
      
      counter +=1 
      
      
for i in range(3):
      width = 10
      height = 8
      plt.figure(figsize=(width, height))
      plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
            all_accu[len_model*i + 0:len_model*(i+1)])
      plt.xlabel('Model Name')
      plt.ylabel('Accuracy')
      plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], all_accu[len_model*i + 0:len_model*(i+1)], all_accu_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                  capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
      plt.ylim(0.8, 1)
      plt.xticks(rotation=270)
      plt.gcf().set_size_inches(4.2, 3.8)
      plt.tight_layout()
      plt.savefig(main_dir + plot_dir + '/' +
                  level_name[i] + 'accuracy_plot_lung_breast.png')
      plt.savefig(main_dir + pdf_dir + '/' +
                  level_name[i] + 'accuracy_plot_lung_breast.pdf')
      plt.show()


      width = 10
      height = 8
      plt.figure(figsize=(width, height))
      plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
              ave_pre[len_model*i + 0:len_model*(i+1)])
      plt.xlabel('Model Name')
      plt.ylabel('Average Precision Score')
      plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], ave_pre[len_model*i + 0:len_model*(i+1)], ave_pre_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                   capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
      plt.ylim(0.8, 1)
      plt.xticks(rotation=270)
      plt.gcf().set_size_inches(4.2, 3.8)
      plt.tight_layout()
      plt.savefig(main_dir + plot_dir + '/' +
                  level_name[i] + 'average_precision_plot_lung_breast.png')
      plt.savefig(main_dir + pdf_dir + '/' +
                  level_name[i] + 'average_precision_plot_lung_breast.pdf')
      plt.show()

      print("Models:")
      print(str(short_model[len_model*i + 0:len_model*(i+1)]))
      print("Accuracy mean:")
      print(str(all_accu[len_model*i + 0:len_model*(i+1)]))
      print("Accuracy sd:")
      print(str(all_accu_sd[len_model*i + 0:len_model*(i+1)]))
      print("Average precision mean:")
      print(str(ave_pre[len_model*i + 0:len_model*(i+1)]))
      print("Average precision sd:")
      print(str(ave_pre_sd[len_model*i + 0:len_model*(i+1)]))
      
      
for i in range(3):
      for score_metric in score_metrics:
            if min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])< 0.8:
                  ylim_lower = min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)]) - 0.1
            else:
                  ylim_lower = 0.8
                  
            if max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])> 1:
                  ylim_upper = max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])  + 0.1
                  
            else: 
                  ylim_upper = 1
                
            plot_metrics(metrics_array_main = overall_metrics[score_metric+'_mean'], metrics_array_errbar = overall_metrics[score_metric+'_sd'], model_array = short_model, label = level_name[i], metric_name = score_metric, save_path = latent_fig_save_path_lung_breast, index1 = len_model*i, index2 = len_model*(i+1), width = 10, height = 8, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
            


# %% [markdown]
# Leukaemia vs normal

# %%
cancer_index = anno_trim.index[(anno_trim['Characteristics[organism part]'] == 'bone marrow') & (anno_trim['Characteristics[disease]'] == 'acute myeloid leukaemia')]
healthy_index = anno_trim.index[(anno_trim['Characteristics[organism part]'] == 'bone marrow') & (anno_trim['Characteristics[disease]'] == 'normal')]

# %%
short_model = []
stat_res = []
pval_res = []
counter = 0
len_model = 5
i = 0
all_accu = []
all_accu_sd = []
ave_pre = []
ave_pre_sd = []
overall_metrics = {}
for score_metric in score_metrics:
      overall_metrics[score_metric+'_sd'] = []
      overall_metrics[score_metric+'_mean'] = [] 


for res in all_res:
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

      # print("Train Accuracy: %0.2f (+/- %0.2f)" %
      #       (scores['train_accuracy'].mean(), scores['train_accuracy'].std() * 2))
      # print("Test Accuracy: %0.2f (+/- %0.2f)" %
      #       (scores['test_accuracy'].mean(), scores['test_accuracy'].std() * 2))

      # print("Train Average Precision: %0.2f (+/- %0.2f)" %
      #       (scores['train_average_precision'].mean(), scores['train_average_precision'].std() * 2))
      # print("Test Average Precision: %0.2f (+/- %0.2f)" %
      #       (scores['test_average_precision'].mean(), scores['test_average_precision'].std() * 2))
      
      for score_metric in score_metrics:
            overall_metrics[score_metric+'_sd'].append(scores['test_' + score_metric].std())
            overall_metrics[score_metric+'_mean'].append(scores['test_' + score_metric].mean())

      short_model.append(model_list[counter])
      all_accu.append(scores['test_accuracy'].mean())
      all_accu_sd.append(scores['test_accuracy'].std())
      ave_pre.append(scores['test_average_precision'].mean())
      ave_pre_sd.append(scores['test_average_precision'].std())
      
      counter +=1 
      
      
for i in range(3):
      width = 10
      height = 8
      plt.figure(figsize=(width, height))
      plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
            all_accu[len_model*i + 0:len_model*(i+1)])
      plt.xlabel('Model Name')
      plt.ylabel('Accuracy')
      plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], all_accu[len_model*i + 0:len_model*(i+1)], all_accu_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                  capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
      plt.ylim(0.8, 1)
      plt.xticks(rotation=270)
      plt.gcf().set_size_inches(4.2, 3.8)
      plt.tight_layout()
      plt.savefig(main_dir + plot_dir + '/' +
                  level_name[i] + 'accuracy_plot_leukaemia_health.png')
      plt.savefig(main_dir + pdf_dir + '/' +
                  level_name[i] + 'accuracy_plot_leukaemia_health.pdf')
      plt.show()


      width = 10
      height = 8
      plt.figure(figsize=(width, height))
      plt.bar(short_model[len_model*i + 0:len_model*(i+1)],
              ave_pre[len_model*i + 0:len_model*(i+1)])
      plt.xlabel('Model Name')
      plt.ylabel('Average Precision Score')
      plt.errorbar(short_model[len_model*i + 0:len_model*(i+1)], ave_pre[len_model*i + 0:len_model*(i+1)], ave_pre_sd[len_model*i + 0:len_model*(i+1)], fmt='.', color='Black', elinewidth=2,
                   capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2.5)
      plt.ylim(0.8, 1)
      plt.xticks(rotation=270)
      plt.gcf().set_size_inches(4.2, 3.8)
      plt.tight_layout()
      plt.savefig(main_dir + plot_dir + '/' +
                  level_name[i] + 'average_precision_plot_leukaemia_health.png')
      plt.savefig(main_dir + pdf_dir + '/' +
                  level_name[i] + 'average_precision_plot_leukaemia_health.pdf')
      plt.show()

      print("Models:")
      print(str(short_model[len_model*i + 0:len_model*(i+1)]))
      print("Accuracy mean:")
      print(str(all_accu[len_model*i + 0:len_model*(i+1)]))
      print("Accuracy sd:")
      print(str(all_accu_sd[len_model*i + 0:len_model*(i+1)]))
      print("Average precision mean:")
      print(str(ave_pre[len_model*i + 0:len_model*(i+1)]))
      print("Average precision sd:")
      print(str(ave_pre_sd[len_model*i + 0:len_model*(i+1)]))
      
      
for i in range(3):
      for score_metric in score_metrics:
            if min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])< 0.8:
                  ylim_lower = min(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) - max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)]) - 0.1
            else:
                  ylim_lower = 0.8
                  
            if max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])> 1:
                  ylim_upper = max(overall_metrics[score_metric+'_mean'][len_model*i:len_model*(i+1)]) + max(overall_metrics[score_metric+'_sd'][len_model*i:len_model*(i+1)])  + 0.1
                  
            else: 
                  ylim_upper = 1
                
            plot_metrics(metrics_array_main = overall_metrics[score_metric+'_mean'], metrics_array_errbar = overall_metrics[score_metric+'_sd'], model_array = short_model, label = level_name[i], metric_name = score_metric, save_path = latent_fig_save_path_leukeamia_health, index1 = len_model*i, index2 = len_model*(i+1), width = 10, height = 8, ylim_lower=ylim_lower, ylim_upper=ylim_upper)
            



