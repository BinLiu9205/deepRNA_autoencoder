# Calculating sample wise gene expression variance in different training and test data sets

import pandas as pd
import numpy as np

geneset_data_list = ['/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl','/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl', '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_train_zscore.pkl', '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_test_zscore.pkl' ]
geneset_name_list = ['gene_level_no_zscore_normalization_train', 'gene_level_no_zscore_normalization_test', 'gene_level_with_zscore_normalization_train', 'gene_level_with_zscore_normalization_test']

for geneset_data, geneset_name in zip(geneset_data_list, geneset_name_list):
    df = pd.read_pickle(geneset_data)
    df_var = df.var(axis=0)
    save_directory = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_gene_variance/' + geneset_name + '_column_variance.pkl'
    df_var.to_pickle(save_directory)
    print(df_var.shape)
    sorted_columns = df_var.sort_values(ascending=False).index
    save_directory_new1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_gene_variance/' + geneset_name + '_variance_descending_sorted_index.npy'
    np.save(save_directory_new1, sorted_columns)
    df_sorted = df[sorted_columns]
    save_directory_new2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_gene_variance/' + geneset_name + '_variance_descending_sorted_dataset.pkl'
    df_sorted.to_pickle(save_directory_new2)