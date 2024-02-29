import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Prior_latent_dims_mapping as latentMapping
import os
import math

savefig_path = os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/', 'Boxplot_between_prior_latent_mu_for_betas.pdf')
scatter_save_path_base = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/scatterplot_prior_latent_betas_sweep/'
anno_file = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/beta_relevant_models.csv')
run_list = anno_file['Run ID']
beta_val = anno_file['beta']

anno = pd.read_csv(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv", index_col=0)
anno_name = list(anno.columns)
anno_name = [single[9:] for single in anno_name]

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=len(run_list), figsize=(12, 4), sharey=True)
input_test_file = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
print(input_test_file.shape)
prior_mu_file_all = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv', index_col= 0)
prior_mu_file = prior_mu_file_all.loc[input_test_file.index]
print(prior_mu_file.shape)
latent_result_all = os.listdir('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu')

save_dimensional_latent_model_based = []
min_y = float('inf')
max_y = float('-inf')
# Use a for loop to generate boxplots
for i, runID in enumerate(run_list):
    latent_result_interest = [element for element in latent_result_all if runID in element]
    print(latent_result_interest)
    if len(latent_result_interest) > 1:
        print('Error!')
    else:
        latent_mu_file = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu/' + latent_result_interest[0])
        latentMapping.prior_latent_correlation_boxplot(prior_mu_file, latent_mu_file, str(beta_val[i]), axes[i], i)
        correlation_model = latentMapping.dimensional_prior_latent_correlation_calculation(prior_mu_file, latent_mu_file)
        save_dimensional_latent_model_based.append(correlation_model)
        new_max = np.max(correlation_model)
        new_min = np.min(correlation_model)
        if new_max > max_y:
            max_y = new_max
        if new_min < min_y:
            min_y = new_min
min_y = math.floor(min_y/0.1)*0.1
max_y = math.ceil(max_y/0.1)*0.1
inte_num = int((max_y - min_y)/0.1)


axes[0].set_yticks(np.linspace(min_y, max_y, num=inte_num))  
axes[0].set_yticklabels(np.round(np.linspace(min_y, max_y, num=inte_num), 1))  

fig.text(0.5, 0.01, 'Beta values', ha='center')
axes[0].spines['left'].set_visible(True)
axes[0].set_ylabel('Correlation between prior and latent mean', labelpad=20)

for ax in axes[1:]:
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig(savefig_path)
plt.close()

save_dimensional_latent_model_based_df = pd.DataFrame(save_dimensional_latent_model_based)
save_dimensional_latent_model_based_df.columns = anno_name
save_dimensional_latent_model_based_df.index = ['beta_' + str(beta_val_sg) for beta_val_sg in beta_val]
save_dimensional_latent_model_based_df.to_csv(os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/dimensional_correlation_between_prior_and_mu', 'dimensional_correlation_between_prior_and_mu_beta_sweep.csv'), index=True)

for i, runID in enumerate(run_list):
    latent_result_interest = [element for element in latent_result_all if runID in element]
    if len(latent_result_interest) > 1:
        print('Error!')
    else:        
        latent_mu_file = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu/' + latent_result_interest[0])
        scatter_save_path = os.path.join(scatter_save_path_base, 'Scatterplot_prior_latent_mu_for_sweep_beta_' + str(beta_val[i]) + '.pdf')
        latentMapping.prior_latent_direct_scatterplot(prior_mu_file, latent_mu_file,  anno_name, scatter_save_path)
        
