import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from priorVAE_dynamic_structure import priorVAE
import Intervention_on_latent_dimensions_for_gene_level_analysis as intervention

"""
The current wild card analysis was relevant for adding one more dimension to KEGG only
This script can be equally useful for finding most correlated genes with each given dimension
We are interested in understanding the wild latent genes
"""

# We hard the configuration here 
batch_size = 128
encoder_config = [1000, 100]
latent_dim = 150 
n_features_path = latent_dim
X_test = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
gene_names = X_test.columns.tolist()
n_input = X_test.shape[1]
n_features_trans = X_test.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_base_dir = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/trained_models'
latent_mu_base_dir = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_mu'
latent_sigma_base_dir = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_sigma'
recons_base_dir = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/reconstructed_value'
annotation_file = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/kegg_with_wildcard_models.csv',sep = '\t')

model_path_all = annotation_file['Model File Name']
run_ID_all = annotation_file['Run ID']
beta_vals = annotation_file['Beta']
"""
We have a few loops:
1. For each model, we load the model (this is relevant only to the model)
2. For each model, we load the latent results (this is relevant only to the latent dimension)
3. For each model/latent result, we iterate over a sigma of -3 to 3 -- range (-3,4)

In order to possibly reproduce the influence of sigma and mu, let's use the reparametrization way instead of loading z directly
"""

for model_path_trimmed, run_ID, beta_val in zip(model_path_all, run_ID_all, beta_vals):
    # Load the model
    model_path = os.path.join(model_base_dir, model_path_trimmed)
    model = priorVAE(encoder_config=encoder_config, n_input=n_input, latent_dim =latent_dim, n_features_path = n_features_path, n_features_trans=n_features_trans).to(device)
    model.load_state_dict(torch.load(
        model_path, map_location=torch.device(device)))
    model.eval()  # Set the model to inference mode
    model.to(device)
    # Load the pregenerated latent dimension
    latent_origin_mu = os.path.join(latent_mu_base_dir, run_ID + '_latent_result.npy')
    mu_origin_value = np.load(latent_origin_mu)
    latent_origin_sigma = os.path.join(latent_sigma_base_dir, run_ID + '_latent_sigma_result.npy')
    # Calculate the sd of the last dimension
    sigma_origin_value = np.load(latent_origin_sigma)
    orig_dim_sd = np.std(mu_origin_value[:, -1])
    print('Sd for model with ' + str(beta_val) + ' is ' + str(orig_dim_sd))
    # Load the original reconstruction
    reconstructed_origin = np.load(os.path.join(recons_base_dir, run_ID + '_reconstructed_result.npy'))
    # Create the directory
    reconstructed_new_dir = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_dimension_intervention/new_reconstruction_values_KEGG_wildcard_intervention_trained_with_beta_value_of_' + str(beta_val)
    deg_reconstructed_new_dir =  '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_dimension_intervention/DEGs_KEGG_wildcard_intervention_trained_with_beta_value_of_' + str(beta_val)
    os.makedirs(reconstructed_new_dir, exist_ok= True)   
    os.makedirs(deg_reconstructed_new_dir, exist_ok = True)
    
    for k in range(-3, 4):
        reconstructed_new = intervention.reconstruction_intervention_decoder(model, device, latent_mu = mu_origin_value, latent_sigma = sigma_origin_value, intervention_sd = orig_dim_sd,  intervention_dim = -1, intervention_value = k)
        significant_genes, results_df = intervention.deg_pairwised_reconstruction_comparison(gene_names,reconstructed_origin.transpose(), reconstructed_new.transpose())
        
        np.save(reconstructed_new_dir + '/reconstructed_value_KEGG_wildcard_with_beta_value_of_' + str(beta_val) + '_intervention_level_' + str(k) + '.npy', reconstructed_new)
        
        significant_genes.to_csv(deg_reconstructed_new_dir + '/degs_significant_genes_results_KEGG_wildcard_with_beta_value_of_' + str(beta_val) + '_intervention_level_' + str(k) + '.csv')
        results_df.to_csv(deg_reconstructed_new_dir + '/degs_all_genes_results_KEGG_wildcard_with_beta_value_of_' + str(beta_val) + '_intervention_level_' + str(k) + '.csv')
        
        if k != 0:
            reconstructed_new = intervention.reconstruction_intervention_decoder(model, device, latent_mu = mu_origin_value, latent_sigma = sigma_origin_value, intervention_sd = orig_dim_sd, intervention_dim = -1, intervention_value = 1/k)
            significant_genes, results_df = intervention.deg_pairwised_reconstruction_comparison(gene_names,reconstructed_origin.transpose(), reconstructed_new.transpose())
            
            np.save(reconstructed_new_dir + '/reconstructed_value_KEGG_wildcard_with_beta_value_of_' + str(beta_val) + '_intervention_level_reciprocal_of_' + str(k) + '.npy', reconstructed_new)
            
            significant_genes.to_csv(deg_reconstructed_new_dir + '/degs_significant_genes_results_KEGG_wildcard_with_beta_value_of_' + str(beta_val) + '_intervention_level_reciprocal_of_' + str(k) + '.csv')
            results_df.to_csv(deg_reconstructed_new_dir + '/degs_all_genes_results_KEGG_wildcard_with_beta_value_of_' + str(beta_val) + '_intervention_level_reciprocal_of_' + str(k) + '.csv')