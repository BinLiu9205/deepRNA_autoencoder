import gseapy as gp
import os
import pandas as pd
import re

beta_infos = ['beta_value_of_1', 'beta_value_of_50', 'beta_value_of_250']
gene_sets = ['/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/signatures_GSEA/h.all.v2023.2.Hs.symbols.gmt', '/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/signatures_GSEA/c5.go.bp.v2023.2.Hs.symbols.gmt', '/mnt/dzl_bioinf/binliu/deepRNA/Updates_plos_comp/signatures_GSEA/c2.cp.kegg_medicus.v2023.2.Hs.symbols.gmt']  
gene_set_names = ['MsigDB',  'GOBP' ,'kegg_medicus']
intervention_base = '/mnt/dzl_bioinf/binliu/deepRNA/major_revision/model_information/latent_dimension_intervention'
pattern = r"intervention_(.*?)\.csv"

for beta_info in beta_infos:
    deg_dfs = os.listdir(os.path.join(intervention_base,'DEGs_KEGG_wildcard_intervention_trained_with_' + beta_info))
    deg_sub_dfs = [element for element in deg_dfs if "_all_" in element]
    deg_sub_dfs_full = [os.path.join(intervention_base,'DEGs_KEGG_wildcard_intervention_trained_with_' + beta_info, element) for element in deg_sub_dfs]
    os.makedirs(os.path.join(intervention_base, 'gsea_results_wildcard_with_' + beta_info), exist_ok = True)
    
    for deg_sub_df, deg_sub_df_full in zip(deg_sub_dfs, deg_sub_dfs_full):
        ranked_gene_list_ori = pd.read_csv(deg_sub_df_full)
        # We sort based on p-value rather than adjusted-pvalues since the distribution of adjusted pvalues can be very tight due to the multiple testings correction and cause error
        ranked_gene_list_sorted = ranked_gene_list_ori.sort_values(by='p_value').head(2000)
        gene_list_for_gseapy = [(gene.upper(), log2fc) for gene, log2fc in zip(ranked_gene_list_sorted['gene'], ranked_gene_list_sorted['fold_change'])]
        df_gene_list_for_gseapy = pd.DataFrame(gene_list_for_gseapy, columns=['Gene', 'Rank'])
        match = re.search(pattern, deg_sub_df)
        if match is not None:
            result = match.group(1)
        else: 
            result = 'unknown file names'
        os.makedirs(os.path.join(intervention_base, 'gsea_results_wildcard_with_' + beta_info, result), exist_ok = True)
        intervention_level_base = os.path.join(intervention_base, 'gsea_results_wildcard_with_' + beta_info, result)
        
        for gene_set, gene_set_name in zip(gene_sets, gene_set_names):
            os.makedirs(os.path.join(intervention_level_base, gene_set_name), exist_ok = True)
            savepath_sig = os.path.join(intervention_level_base, gene_set_name)
            #print(str(savepath_sig))
            gsea_results = gp.prerank(rnk=df_gene_list_for_gseapy, gene_sets=gene_set, outdir=savepath_sig, min_size=5, max_size=3000, permutation_num=5000, verbose=True)
            



# Perform GSEA


# The `outdir` parameter specifies where to save the results and plots. Adjust the `min_size` and `max_size` to filter gene sets by their sizes, and `permutation_num` to set the number of permutations for significance testing.

