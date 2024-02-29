import Latent_dimension_visualization_sample_based as latentDim
import os
import pandas as pd
import numpy as np
seed = 42
np.random.seed(seed)
from itertools import repeat
from pca import pca
import matplotlib.pyplot as plt


latent_mu_nsclc_data_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu_nsclc'
latent_mu_sclc_data_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu_sclc'
latent_fig_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/latent_dimension_plots'
os.makedirs(latent_fig_save_path_base, exist_ok=True)
latent_mu_names = os.listdir(latent_mu_nsclc_data_base)
sample_annotation = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'arrayExpress_annotation.csv'),index_col=0)

anno_MSigDB = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv", index_col=0)
anno_name_MSigDB = list(anno_MSigDB.columns)
anno_name_MSigDB = [single[9:] for single in anno_name_MSigDB]


anno_KEGG = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization.csv", index_col=0)
anno_name_KEGG = list(anno_KEGG.columns)
anno_name_KEGG = [single for single in anno_name_KEGG]
n=20


input_test_file = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
input_train_file = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl')

annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')
X_train = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl')
sample_nr = X_test.index
anno_trim = annotate_tem[annotate_tem['Source Name'].isin(sample_nr)]
anno_trim = anno_trim.set_index('Source Name')
anno_trim = anno_trim.loc[sample_nr]
anno_trim = anno_trim.reset_index()
annotate_tem_train = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
annotate_tem_train = annotate_tem_train.set_index('Source Name')
anno_condition = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/nsclc_sclc_annotation.csv")
anno_condition.set_index('CompositeSequence Identifier', inplace=True)


prior_MSigDB_all = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv', index_col= 0)
combined_MSigDB = prior_MSigDB_all.loc[anno_condition.index.intersection(prior_MSigDB_all.index)]
print(combined_MSigDB.shape)

prior_KEGG_noWC_all = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization.csv', index_col= 0)
combined_KEGG_noWC = prior_KEGG_noWC_all.loc[anno_condition.index.intersection(prior_KEGG_noWC_all.index)]

prior_KEGG_withWC_all = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization_wildcard_added.csv', index_col= 0)
combined_KEGG_withWC = prior_KEGG_withWC_all.loc[anno_condition.index.intersection(prior_KEGG_withWC_all.index)]



for latent_mu_name in latent_mu_names:
    shortened = latent_mu_name[0:-len('_latent_result.npy')]
    latent_file_nsclc = np.load(os.path.join(latent_mu_nsclc_data_base, latent_mu_name))
    latent_file_sclc = np.load(os.path.join(latent_mu_sclc_data_base, latent_mu_name))
    nsclc_len = latent_file_nsclc.shape[0]
    sclc_len = latent_file_sclc.shape[0]
    latent_file = np.vstack((latent_file_nsclc, latent_file_sclc))
    
    nsclc_index = np.random.choice(nsclc_len, n, replace=False)  
    sclc_index = np.random.choice(range(nsclc_len, nsclc_len + sclc_len), n, replace=False) 
    
    
    latent_cancer_pca_biplot = latent_fig_save_path_base + '/sclc_adenocarcinoma_phenotype_pca_biplots_twoDimensions'
    os.makedirs(latent_cancer_pca_biplot, exist_ok=True)
    latent_cancer_pca_biplot = latent_cancer_pca_biplot + '/' + shortened
    
    latent_cancer_deDims_plot = latent_fig_save_path_base + '/sclc_adenocarcinoma_phenotype_deDims'
    os.makedirs(latent_cancer_deDims_plot, exist_ok = True)
    latent_cancer_deDims_plot = latent_cancer_deDims_plot + '/' + shortened
    
    latent_cancer_heatmap_plot = latent_fig_save_path_base + '/sclc_adenocarcinoma_phenotype_latent_heatmaps'
    os.makedirs(latent_cancer_heatmap_plot, exist_ok = True)
    latent_cancer_heatmap_plot = latent_cancer_heatmap_plot + '/' + shortened
    
    latent_tables = latent_fig_save_path_base + '/sclc_adenocarcinoma_phenotype_results'
    os.makedirs(latent_tables, exist_ok = True)
    latent_cancer_numeric = latent_tables + '/' + shortened + '_deg_pvalues_correlations.csv'
    
    annotation_type_num = latent_file.shape[1]
    if annotation_type_num == 50: # MSigDB
        anno = anno_name_MSigDB
        prior_res = combined_MSigDB
    elif annotation_type_num == 149: #KEGG without Wildcard
        anno = anno_name_KEGG
        prior_res = combined_KEGG_noWC
    elif annotation_type_num == 150:
        anno = anno_name_KEGG + ['Wildcard dimension']
        prior_res = combined_KEGG_withWC
    else: 
        print('Error finding annotations for ' + latent_mu_name)
        
        
        
    latent_res = latent_file
    con1_index = nsclc_index
    con2_index = sclc_index
    con1_label = 'Adenocarcinoma'
    con2_label = 'SCLC'
    path_anno = anno 
    save_path = latent_cancer_pca_biplot 
    n_feat=10
    

    res_selected_all = np.vstack((latent_file[con1_index,:], latent_file[con1_index,:]))
    condition_all = [*list(repeat(con1_label, len(con1_index))), *list(
        repeat(con2_label, len(con2_index)))]
    contains_invalid_latent = np.isnan(res_selected_all).any() or np.isinf(res_selected_all).any()
    if contains_invalid_latent:
        print('Error in latent dimensions')
    else:
        pca_model = pca(n_components=2)
        pca_latent_selected = pca_model.fit_transform(
                res_selected_all, row_labels=condition_all, col_labels=path_anno)
        scatter_size = 10
        #fig, ax = pca_model.plot()
        #fig, ax = pca_model.scatter()
        fig, ax = pca_model.biplot(n_feat=n_feat)
        plt.gcf().set_size_inches(6, 5.6)
        plt.tight_layout()
        
        plt.savefig(save_path + '_biplots_plots_latent_dimension_for_with_' + str(n_feat) + '_features_' + con1_label + '_vs_' + con2_label + '.eps')
        plt.show()
        plt.close()

    
    df = latentDim.differential_expressed_latent_dimensions_with_given_indices(latent_file, prior_res, nsclc_index, sclc_index, 'NSCLC', 'SCLC', anno, latent_cancer_deDims_plot)
    
    latentDim.deg_latent_dimensions_with_given_indices_heatmap_conditions(latent_file, nsclc_index, sclc_index, 'NSCLC', 'SCLC', anno, latent_cancer_heatmap_plot)
    df.to_csv(latent_cancer_numeric)