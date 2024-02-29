import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf




def reconstruction_intervention_decoder(model, device, latent_mu, latent_sigma, intervention_sd, 
                                        intervention_dim, intervention_value):
    """
    Reconstructs the latent expression using the decoder to find the most influential genes.
    The idea is to generate new reparametrized samples with new mu and new standard deviation. However, we do linear transformation only by adding the new value (coeffient of X is 1) -- the variance will not change 
    """
    new_latent_mu = latent_mu.copy()
    new_latent_mu[:, intervention_dim] = latent_mu[ :,intervention_dim] + intervention_value*intervention_sd
    std = torch.exp(0.5*torch.tensor(latent_sigma,dtype=torch.float).to(device))
    eps = torch.randn_like(std)
    new_latent_dims_tensor = torch.tensor(new_latent_mu,dtype=torch.float).to(device)  + (eps*std)
    with torch.no_grad():  # Ensure gradients are not computed for inference
        intervened_reconstruction  = model.decoder(new_latent_dims_tensor)
    if device.type == 'cuda':
        intervened_reconstruction = intervened_reconstruction.cpu().numpy()
        return intervened_reconstruction
    else:
        intervened_reconstruction = intervened_reconstruction.numpy()
        return intervened_reconstruction
            


## For the comparison between the original reconstruction values and the intervented ones, it's better just to use pairwised than the limma based regression
# Assuming 'original_reconstructions' and 'intervened_reconstructions' are your DataFrames with gene expression data

def deg_pairwised_reconstruction_comparison(annotation, original_reconstructions, intervened_reconstructions):
    assert original_reconstructions.shape == intervened_reconstructions.shape
    results = []
    original_reconstructions_mean = np.mean(original_reconstructions, axis=1)
    intervened_reconstructions_mean = np.mean(intervened_reconstructions, axis=1)
    fold_change = np.log2((original_reconstructions_mean) / (intervened_reconstructions_mean))
    for gene_idx in range(original_reconstructions.shape[0]):
        original_expr = original_reconstructions[gene_idx, :]
        intervened_expr = intervened_reconstructions[gene_idx, :]
        
        # Perform a paired t-test for this gene
        _, p_value = stats.ttest_rel(original_expr, intervened_expr)
        #_, p_value = wilcoxon(original_expr, intervened_expr)
        
        results.append({
            'gene': annotation[gene_idx],
            'p_value': p_value
        })

    results_df = pd.DataFrame(results)
    results_df['p_value_adj'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    results_df['fold_change'] = fold_change

    # Filter for significant results, etc.
    significant_genes = results_df[results_df['p_value_adj'] < 0.05]
    
    return significant_genes, results_df