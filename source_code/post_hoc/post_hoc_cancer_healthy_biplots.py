from sklearn.decomposition import PCA
from scipy.stats import ttest_ind as ttest_ind
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import math
from itertools import repeat
from pca import pca
seed = 42
np.random.seed(seed)


annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")

f_dir1 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")
f_dir2 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")
f_dir3 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")


dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"
dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"
dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"

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
plot_dir = 'degs_of_dimensions'
pdf_dir = 'ms_degs_of_dimensions_pdf'

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
normal_index = anno_trim.index[(anno_trim['Characteristics[disease]'] == 'normal') & (anno_trim['Characteristics[organism part]'] == 'lung')]
adenocarcinoma_index = anno_trim.index[anno_trim['Characteristics[cell type]']
                                       == 'lung adenocarcinoma cell line']


list_num = [i for i in range(1, 51)]
dimension_label = list(map(lambda x: "dimension_"+str(x), list_num))
anno = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv", index_col=0)
anno_name = list(anno.columns)
anno_name = [single[9:] for single in anno_name]


short_model = []
stat_res = []
pval_res = []
counter = 0 
len_model = 5
for res in all_res:
    res = np.load(res)
    res_ordered_cancer = res[adenocarcinoma_index]                                                                   
    res_ordered_normal = res[normal_index]
    t_stat, pval = ttest_ind(res_ordered_normal, res_ordered_cancer, axis = 0)
    for i in range(len_model):
        model_short = model_list[counter]
        short_model.append(model_short)
        stat_res.append(t_stat)
        pval_res.append(pval)

    stat_res = pd.DataFrame(stat_res)
    pval_res = pd.DataFrame(pval_res)

    stat_res = stat_res.set_index(pd.Index(short_model))
    pval_res = pval_res.set_index(pd.Index(short_model))
    trans_pval = pval_res.apply(lambda x: -np.log10(x)
                                if np.issubdtype(x.dtype, np.number) else x)

    for i in range(trans_pval.shape[0]):
        if trans_pval.index[i].find("prior") != -1:
            unsorted_list = [(pval, path_name) for pval, path_name in
                            zip(trans_pval.iloc[i, :], anno_name)]
            sorted_list = sorted(unsorted_list)

            pvalue_sorted = []
            path_sorted = []

            for j in sorted_list:
                pvalue_sorted += [j[0]]
                path_sorted += [j[1]]

            fig = plt.figure(figsize=(12, 8))

            plt.title("15 Most significant pathways for " +
                    str(trans_pval.index[i]), fontsize=15)

            plt.barh(path_sorted[-15:], pvalue_sorted[-15:])
            #print(result)
            plt.xticks(rotation=270)
            plt.savefig(main_dir + plot_dir +'/' + name_list[counter] + '_significant_pathways.png')
            plt.close()
            
            fig = plt.figure(figsize=(12, 8))

            plt.barh(path_sorted[-15:], pvalue_sorted[-15:])
            #print(result)
            plt.xticks(rotation=270)
            plt.savefig(main_dir + pdf_dir + '/' +
                        name_list[counter] + '_significant_pathways.svg')
            
        short_model = []
        stat_res = []
        pval_res = []    

    res_selected_all = res[adenocarcinoma_index.append(normal_index)]
    condition_health = [*list(repeat("adenocarcinoma", len(adenocarcinoma_index))), *list(
        repeat("healthy", len(normal_index)))]
    pca_model1 = PCA(n_components=2)
    pca_latent_selected = pca_model1.fit_transform(res_selected_all)
    finalDf = pd.DataFrame(data=pca_latent_selected)
    finalDf['condition_health'] = condition_health

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=13)
    ax.set_ylabel('Principal Component 2', fontsize=13)
    ax.set_title('2 component PCA for ' + model_short, fontsize=15)
    targets = ['adenocarcinoma', 'healthy']
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['condition_health'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 0],
                   finalDf.loc[indicesToKeep, 1], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.savefig(main_dir + plot_dir +'/' + name_list[counter] + '_significant_pathways.png')
        
    print(name_list[counter])
    print(pca_model1.explained_variance_ratio_)

    res_selected_all = res[adenocarcinoma_index.append(normal_index)]
    condition_health = [*list(repeat("adenocarcinoma", len(adenocarcinoma_index))), *list(
        repeat("healthy", len(normal_index)))]
    pca_model = pca(n_components=2)
    if model_list[counter].find("prior") != -1:
        pca_latent_selected = pca_model.fit_transform(
            res_selected_all, row_labels=condition_health, col_labels=anno_name)
        #res_selected_all.columns = anno_name
    else:
        #res_selected_all.columns = dimension_label
        pca_latent_selected = pca_model.fit_transform(
            res_selected_all, row_labels=condition_health, col_labels=dimension_label)
    fig, ax = pca_model.plot()
    plt.savefig(main_dir + plot_dir +'/' + name_list[counter] + '_pca_variance_plot_health.png')
    # Scatter first 2 PCs
    fig, ax = pca_model.scatter()
    plt.savefig(main_dir + plot_dir +'/' + name_list[counter] + '_pca_scatter_health.png')
    fig, ax = pca_model.biplot(n_feat=10)
    plt.gcf().set_size_inches(6, 5.6)
    plt.tight_layout()
    plt.savefig(main_dir + plot_dir +'/' + name_list[counter] + '_biplots_health.png')
    plt.savefig(main_dir + pdf_dir + '/' +
                name_list[counter] + '_biplots_health.svg')
    plt.close()
    counter +=1 
