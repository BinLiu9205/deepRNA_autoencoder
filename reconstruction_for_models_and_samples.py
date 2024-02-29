import Reconstruction_performance_metrics as recon
import os
import pandas as pd
import numpy as np

model_name = []
correlation_sd = []
correlation_mean = []

print('I am running the script')
recon_numeric_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/correlation_reconstruction_and_input_test'
recon_fig_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/reconstructed_metrics_plots'
recon_original_data_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/reconstructed_value'
sample_annotation = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_test.csv'),index_col=0)
tissue_anno = sample_annotation['Characteristics[organism part]']

recon_np_names = os.listdir(recon_original_data_base)
# The results for zscore might not be correct due to the simplification, but that's not the main focus. Ignore that at this stage
input_test_no_zscore = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
input_test_with_zscore = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_test_zscore.pkl')


for recon_np_name in recon_np_names:
    shortened = recon_np_name[0:-len('_reconstructed_result.npy')]
    print('Loading files for ' + shortened)
    #print('Calculating correlation: ' + shortened)
    if shortened in ['40st3s32' , 'mxdk7t21', 'r0ei40eo']:
        input_test = input_test_with_zscore
    else:
        input_test = input_test_no_zscore
    input_test_np = np.array(input_test)
    recon_file = np.load(os.path.join(recon_original_data_base, recon_np_name))
    #recon_correlation_value = recon.calculate_sample_wise_correlation_numpy(input_test_np, recon_file)
    #recon_save_path = os.path.join(recon_numeric_save_path_base, shortened+'_input_reconstruct_correlation.npy')
    #print(recon_correlation_value.shape)
    #np.save(recon_save_path, recon_correlation_value)
    
    print('Drawing pairwise correlation: ' + shortened)
    recon_correlation_original_value = recon_fig_save_path_base + '/pairwise_correlation_InputOutput_metrics/' + shortened 
    recon.input_reconstruction_pairwise_correlation_plot(input_test_np, recon_file, recon_correlation_original_value)
    
    print('Drawing pairwise correlation with tissue labels: ' + shortened)
    recon_correlation_tissue_value = recon_fig_save_path_base + '/pairwise_correlation_Tissue_metrics/' + shortened
    recon.input_reconstruction_pairwise_correlation_with_annotation_labels(input_test_np, recon_file, tissue_anno, recon_correlation_tissue_value)
    
    print('Tsne files: ' + shortened)
    recon_correlation_tsne_value = recon_fig_save_path_base + '/tsne_reconstruction/' + shortened
    recon.input_reconstruction_tsne_plot_most_frequent_organs(input_test_np, recon_file, tissue_anno, recon_correlation_tsne_value)
    
    
    print('UMAP files: ' + shortened)
    recon_correlation_UMAP_value = recon_fig_save_path_base + '/UMAP_reconstruction/' + shortened
    recon.input_reconstruction_umap_plot_most_frequent_organs(input_test_np, recon_file, tissue_anno, recon_correlation_UMAP_value)
    
    model_name.append(shortened)
    correlation_mean_sg, correlation_sd_sg  = recon.calculate_overall_reconstruction_each_model(input_test_np, recon_file)
    correlation_mean.append(correlation_mean_sg)
    correlation_sd.append(correlation_sd_sg)
    
df = pd.DataFrame({
    'Model Name': model_name,
    'Correlation Mean': correlation_mean,
    'Correlation Sd': correlation_sd
})

df.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/reconstructed_metrics_plots/correlation_values.csv')