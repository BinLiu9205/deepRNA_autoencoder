# There are five different kinds of input and a few combinations of priors.
# We are interested in getting a subset of nsclc and sclc for the different input 
import pandas as pd
import os

test_info_array = ['/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl', 
                   '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl', 
                   '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_test_zscore.pkl',
                   '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl',
                   '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test_zscore/all_samples_transcript_level_test_zscore.pkl'
                   ]
train_info_array = ['/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_train.pkl',
                    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl',
                    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_train_zscore.pkl',
                    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl',
                    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test_zscore/all_samples_transcript_level_train_zscore.pkl'
                    ]

annotate_test = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_test.csv'),index_col=0)
annotate_train = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_train.csv'),index_col=0)

nsclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
nsclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'lung adenocarcinoma cell line']
sclc_test = annotate_test.index[annotate_test['Factor Value[cell type]'] == 'small cell lung cancer cell line']
sclc_train = annotate_train.index[annotate_train['Factor Value[cell type]'] == 'small cell lung cancer cell line']

anno_new = pd.concat([annotate_train[annotate_train['Factor Value[cell type]'] == 'lung adenocarcinoma cell line'],
                      annotate_test[annotate_test['Factor Value[cell type]'] == 'lung adenocarcinoma cell line'], 
                      annotate_train[annotate_train['Factor Value[cell type]'] == 'small cell lung cancer cell line'],
                      annotate_test[annotate_test['Factor Value[cell type]'] == 'small cell lung cancer cell line']])

anno_new.to_csv('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/nsclc_sclc_annotation.csv')


for test_file, train_file in zip(test_info_array, train_info_array):
    test_df = pd.read_pickle(test_file)
    train_df = pd.read_pickle(train_file)
    
    assert train_df.index.equals(annotate_train.index), "Indices are not identical for trains."
    assert test_df.index.equals(annotate_test.index), "Indices are not identical for tests."
    
    
    nsclc_train_data = train_df.loc[nsclc_train]
    nsclc_test_data = test_df.loc[nsclc_test]
    sclc_train_data = train_df.loc[sclc_train]
    sclc_test_data = test_df.loc[sclc_test]
    
    nsclc_all = pd.concat([nsclc_train_data, nsclc_test_data], axis=0)
    sclc_all = pd.concat([sclc_train_data, sclc_test_data], axis=0)
    
    print(nsclc_all.shape)
    print(sclc_all.shape)
    
    parts = test_file.rsplit('test', 1) 
    modified_path_nsclc = 'nsclc'.join(parts)
    modified_path_sclc = 'sclc'.join(parts)
    
    nsclc_all.to_pickle(modified_path_nsclc)
    sclc_all.to_pickle(modified_path_sclc)