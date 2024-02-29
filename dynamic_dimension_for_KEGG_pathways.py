## This script saves modifications on priors generated on different data to accommodate the downstream training
import pandas as pd
mean_file = pd.read_csv("/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization.csv",index_col=0)
sigma_file = pd.read_csv("/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_sigma_KEGG_gene_level_no_normalization.csv",index_col=0)

print(mean_file.shape)
print(sigma_file.shape)

mean_file['WILDCARD_DIM'] = 0
sigma_file['WILDCARD_DIM'] = 1

print(mean_file.shape)
print(sigma_file.shape)

mean_file.to_csv("/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_mean_KEGG_gene_level_no_normalization_wildcard_added.csv")
sigma_file.to_csv("/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/kegg_no_normalization/gene_level/bootstrap_sigma_KEGG_gene_level_no_normalization_wildcard_added.csv")
