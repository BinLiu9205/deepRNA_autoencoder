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
tissue_anno = sample_annotation['Characteristics[organism part]']



anno_MSigDB = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv", index_col=0)
anno_name_MSigDB = list(anno_MSigDB.columns)
anno_name_MSigDB = [single[9:] for single in anno_name_MSigDB]


anno_KEGG = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization.csv", index_col=0)
anno_name_KEGG = list(anno_KEGG.columns)

for latent_mu_name in latent_mu_names:
    shortened = latent_mu_name[0:-len('_latent_result.npy')]
    latent_file = np.load(os.path.join(latent_mu_data_base, latent_mu_name))
    
    latent_organ_pca_plot = latent_fig_save_path_base + '/organ_pca_plots_twoDimensions'
    os.makedirs(latent_organ_pca_plot, exist_ok=True)
    latent_organ_pca_plot = latent_organ_pca_plot + '/' + shortened
    latentDim.pca_plots_directly_based_on_annotation(latent_file, tissue_anno, 'organ', latent_organ_pca_plot)
    
    
    latent_organ_biplot = latent_fig_save_path_base + '/biplot_pca_plots_twoDimensions'
    os.makedirs(latent_organ_biplot, exist_ok=True)
    latent_organ_biplot = latent_organ_biplot + '/' + shortened
    
    annotation_type_num = latent_file.shape[1]
    if annotation_type_num == 50: # MSigDB
        anno = anno_name_MSigDB
    elif annotation_type_num == 149: #KEGG without Wildcard
        anno = anno_name_KEGG
    elif annotation_type_num == 150:
        anno = anno_name_KEGG + ['Wildcard dimension']
    else: 
        print('Error finding annotations for ' + latent_mu_name)
        
    
    latentDim.pca_biplots_directly_based_on_annotation(latent_file, tissue_anno, 'organ', anno, latent_organ_biplot)