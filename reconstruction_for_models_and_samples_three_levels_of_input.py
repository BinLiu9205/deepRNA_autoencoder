import Reconstruction_performance_metrics as recon
import os
import pandas as pd
import numpy as np

model_name = []
correlation_sd = []
correlation_mean = []
metrics = ['min_val', 'q1', 'median', 'q3', 'max_val', 'outliers']
df_longer = {}
df_longer['model_name'] = []
df_longer['level'] = []
for metric in metrics:
    df_longer[metric] = []

input_test_gene = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
input_test_transcript = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')
input_test_community = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl')

recon_numeric_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/correlation_reconstruction_and_input_test'
recon_fig_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/older_models_metrics/reconstructed_metrics_plots'
recon_original_data_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/reconstruction'
sample_annotation = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_test.csv'),index_col=0)
tissue_anno = sample_annotation['Characteristics[organism part]']


os.makedirs(os.path.join(recon_fig_save_path_base, 'pairwise_correlation_InputOutput_metrics'), exist_ok=True)
os.makedirs(os.path.join(recon_fig_save_path_base, 'pairwise_correlation_Tissue_metrics'), exist_ok=True)
os.makedirs(os.path.join(recon_fig_save_path_base, 'tsne_reconstruction'), exist_ok=True)
os.makedirs(os.path.join(recon_fig_save_path_base, 'UMAP_reconstruction'), exist_ok=True)
recon_np_names = os.listdir(recon_original_data_base)

for recon_np_name in recon_np_names:
    shortened = recon_np_name[0:-(len('_reconstructed_results.npy')+1)]
    recon_file = np.load(os.path.join(recon_original_data_base, recon_np_name))
    print('Loading files for ' + shortened)
    if 'community' in shortened:
        input_test = input_test_community
        df_longer['level'].append('community')
    elif 'gene' in shortened:
        input_test = input_test_gene
        df_longer['level'].append('gene')
    elif 'transcript' in shortened:
        input_test = input_test_transcript
        df_longer['level'].append('transcript')

    input_test_np = np.array(input_test)
    model_name.append(shortened)
    correlation_mean_sg, correlation_sd_sg  = recon.calculate_overall_reconstruction_each_model(input_test_np, recon_file)
    correlation_mean.append(correlation_mean_sg)
    correlation_sd.append(correlation_sd_sg)
    
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

    min_val , q1, median_val, q3, max_val , outlier = recon.calculate_overall_reconstruction_each_model_detailed_info(input_test_np, recon_file)
    df_longer['model_name'].append(shortened)
    df_longer['min_val'].append(min_val)
    df_longer['q1'].append(q1)
    df_longer['median'].append(median_val)
    df_longer['q3'].append(q3)
    df_longer['max_val'].append(max_val)
    df_longer['outliers'].append(outlier)

df = pd.DataFrame({
    'Model Name': model_name,
    'Correlation Mean': correlation_mean,
    'Correlation Sd': correlation_sd
})

df.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/older_models_metrics/reconstructed_metrics_plots/correlation_values.csv')

    
df_longer_res = pd.DataFrame(df_longer)
df_longer_res.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/older_models_metrics/reconstructed_metrics_plots/correlation_values_boxplot.csv')