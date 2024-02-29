import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf

def deg_pairwised_ttest_version(input_df, con1_index, con2_index):
    con1_sample = input_df[con1_index].T
    con2_sample = input_df[con2_index].T
    assert con1_sample.shape[0] == con2_sample.shape[0]
    results = []

    for gene_idx in range(con1_sample.shape[0]):
        con1_expr = con1_sample.iloc[gene_idx, :]
        con2_expr = con2_sample.iloc[gene_idx, :]
        con1_mean = np.mean(con1_expr)
        con2_mean = np.mean(con2_expr)
        fold_change = np.log2((con2_mean) / (con1_mean))
        # Perform a paired t-test for this gene
        _, p_value = stats.ttest_ind(con1_expr, con2_expr)
        #_, p_value = wilcoxon(original_expr, intervened_expr)
        
        results.append({
            'gene': con1_sample.index[gene_idx],
            'log2FC' : fold_change,
            'p_value': p_value
        })

    results_df = pd.DataFrame(results)
    results_df['p_value_adj'] = multipletests(results_df['p_value'], method='fdr_bh')[1]


    # Filter for significant results, etc.
    significant_genes = results_df[results_df['p_value_adj'] < 0.05]
    
    return results_df, significant_genes


def deg_pairwised_linear_model_version(input_df, con1_index, con2_index, design_matrix):
    results = []

    con1_sample = input_df[con1_index]
    con2_sample = input_df[con2_index]
    con_all = pd.concat([con1_sample, con2_sample], axis=0)

    con_all = con_all.reset_index(drop=True)
    design_matrix = design_matrix.reset_index(drop=True)

    for gene in con1_sample.columns:
        
        model = sm.OLS(con_all[gene], design_matrix).fit()
        
        results.append({
            'gene': gene,
            'coef': model.params[1],  # Coefficient for the condition of interest
            'p_value': model.pvalues[1],  # P-value for the condition of interest
            'adj_r_squared': model.rsquared_adj  # Adjusted R-squared as a measure of model fit
        })

    results_df = pd.DataFrame(results)
    
    # Adjust p-values for multiple testing
    results_df['p_value_adj'] = multipletests(results_df['p_value'], method='fdr_bh')[1]

    # Optional: filter for significant results
    significant_genes = results_df[results_df['p_value_adj'] < 0.05]
    
    return results_df, significant_genes