import pandas as pd

prior1 = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/bootstrap_mean_msigDB_train_gene_level_zscore.csv', index_col=0)
prior2 = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/bootstrap_mean_msigDB_test_gene_level_zscore.csv', index_col=0)
concatenated_df = pd.concat([prior1, prior2])
concatenated_df.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/bootstrap_mean_msigDB_gene_level_zscore_all.csv')

prior1 = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/bootstrap_sigma_msigDB_train_gene_level_zscore.csv', index_col=0)
prior2 = pd.read_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/bootstrap_sigma_msigDB_test_gene_level_zscore.csv', index_col=0)
concatenated_df = pd.concat([prior1, prior2])
concatenated_df.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/msigDB_zscore/gene_level/bootstrap_sigma_msigDB_gene_level_zscore_all.csv')