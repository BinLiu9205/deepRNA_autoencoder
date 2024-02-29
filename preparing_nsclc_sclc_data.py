'''
This script is developed to solve the issue of having only one SCLC sample in the test data set. 
We put the NSCLC and SCLC relevant samples from the training set also into the visualization section
Do the reconstruction and the mu values together
'''

import pandas as pd

annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
annotate_tem_train = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
X_train = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl')
annotate_tem = annotate_tem.set_index('Source Name')
annotate_tem_train = annotate_tem_train.set_index('Source Name')
annotate_test = annotate_tem.reindex(index=X_test.index)
annotate_train = annotate_tem_train.reindex(index=X_train.index)


nsclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
nsclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
sclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'small cell lung cancer cell line']
sclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'small cell lung cancer cell line']

nsclc_test_input = X_test.loc[nsclc_test]
nsclc_train_input = X_train.loc[nsclc_train]
sclc_test_input = X_test.loc[sclc_test]
sclc_train_input = X_train.loc[sclc_train]

# Combine NSCLC test and train inputs together
nsclc_combined = pd.concat([nsclc_test_input, nsclc_train_input])

# Combine SCLC test and train inputs together
sclc_combined = pd.concat([sclc_test_input, sclc_train_input])


nsclc_combined.to_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/nsclc_gene_level.pkl')
sclc_combined.to_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/sclc_gene_level.pkl')


print(nsclc_combined.index[0:10])