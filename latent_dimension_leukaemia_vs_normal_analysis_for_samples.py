import Latent_dimension_visualization_sample_based as latentDim
import os
import pandas as pd
import numpy as np

'''
The main expected parameters are the mu of the latent dimensions
'''

latent_mu_data_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu'
latent_fig_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/latent_dimension_plots'
os.makedirs(latent_fig_save_path_base, exist_ok=True)
latent_mu_names = os.listdir(latent_mu_data_base)
sample_annotation = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_test.csv'),index_col=0)

annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')
sample_nr = X_test.index
anno_trim = annotate_tem[annotate_tem['Source Name'].isin(sample_nr)]
anno_trim = anno_trim.set_index('Source Name')
anno_trim = anno_trim.loc[sample_nr]
anno_trim = anno_trim.reset_index()

cancer_index = anno_trim.index[(anno_trim['Characteristics[organism part]'] == 'bone marrow') & (anno_trim['Characteristics[disease]'] == 'acute myeloid leukaemia')]
healthy_index = anno_trim.index[(anno_trim['Characteristics[organism part]'] == 'bone marrow') & (anno_trim['Characteristics[disease]'] == 'normal')]
print(len(cancer_index))
print(len(healthy_index))


anno_MSigDB = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv", index_col=0)
anno_name_MSigDB = list(anno_MSigDB.columns)
anno_name_MSigDB = [single[9:] for single in anno_name_MSigDB]


anno_KEGG = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization.csv", index_col=0)
anno_name_KEGG = list(anno_KEGG.columns)
anno_name_KEGG = [single for single in anno_name_KEGG]


input_test_file = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')


prior_MSigDB_all = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv', index_col= 0)
prior_MSigDB_test = prior_MSigDB_all.loc[input_test_file.index]

prior_KEGG_noWC_all = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization.csv', index_col= 0)
prior_KEGG_noWC_test = prior_KEGG_noWC_all.loc[input_test_file.index]

prior_KEGG_withWC_all = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization_wildcard_added.csv', index_col= 0)
prior_KEGG_withWC_test = prior_KEGG_withWC_all.loc[input_test_file.index]


for latent_mu_name in latent_mu_names:
    shortened = latent_mu_name[0:-len('_latent_result.npy')]
    latent_file = np.load(os.path.join(latent_mu_data_base, latent_mu_name))
    
    latent_cancer_pca_biplot = latent_fig_save_path_base + '/leukaemia_vs_normal_pca_biplots_twoDimensions'
    os.makedirs(latent_cancer_pca_biplot, exist_ok=True)
    latent_cancer_pca_biplot = latent_cancer_pca_biplot + '/' + shortened
    
    latent_cancer_deDims_plot = latent_fig_save_path_base + '/leukaemia_vs_normal_deDims'
    os.makedirs(latent_cancer_deDims_plot, exist_ok = True)
    latent_cancer_deDims_plot = latent_cancer_deDims_plot + '/' + shortened
    
    latent_cancer_heatmap_plot = latent_fig_save_path_base + '/leukaemia_vs_normal_latent_heatmaps'
    os.makedirs(latent_cancer_heatmap_plot, exist_ok = True)
    latent_cancer_heatmap_plot = latent_cancer_heatmap_plot + '/' + shortened
    
    latent_tables = latent_fig_save_path_base + '/leukaemia_vs_normal_results'
    os.makedirs(latent_tables, exist_ok = True)
    latent_cancer_numeric = latent_tables + '/' + shortened + '_deg_pvalues_correlations.csv'
    
    annotation_type_num = latent_file.shape[1]
    if annotation_type_num == 50: # MSigDB
        anno = anno_name_MSigDB
        prior_res = prior_MSigDB_test
    elif annotation_type_num == 149: #KEGG without Wildcard
        anno = anno_name_KEGG
        prior_res = prior_KEGG_noWC_test
    elif annotation_type_num == 150:
        anno = anno_name_KEGG + ['Wildcard dimension']
        prior_res = prior_KEGG_withWC_test
    else: 
        print('Error finding annotations for ' + latent_mu_name)
        
    
    latentDim.pca_biplots_based_on_selected_indices(latent_file,  healthy_index, cancer_index, 'Healthy', 'Acute Leukaemia', anno, latent_cancer_pca_biplot)
    
    df = latentDim.differential_expressed_latent_dimensions_with_given_indices(latent_file, prior_res, healthy_index, cancer_index, 'Healthy', 'Acute Leukaemia', anno, latent_cancer_deDims_plot)
    
    latentDim.deg_latent_dimensions_with_given_indices_heatmap_conditions(latent_file, healthy_index, cancer_index, 'Healthy', 'Acute Leukaemia', anno, latent_cancer_heatmap_plot)
    df.to_csv(latent_cancer_numeric)
    
    