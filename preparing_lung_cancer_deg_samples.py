import pandas as pd
import numpy as np
from itertools import repeat

df1 = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_nsclc.pkl')
df2 = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_sclc.pkl')

df_new = pd.concat([df1.T, df2.T], axis=1)
df_new.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_nsclc_sclc/microarray_input_gene_level.csv')

df1 = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_nsclc.pkl')
df2 = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_sclc.pkl')

df_new = pd.concat([df1.T, df2.T], axis=1)
df_new.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_nsclc_sclc/microarray_input_transcript_level.csv')

anno_res = pd.read_csv('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/nsclc_sclc_annotation.csv')
batch_info_all = anno_res['Comment [ArrayExpress accession]']
condition_all_label = [*repeat('nsclc', df1.shape[0]), *repeat('sclc', df2.shape[0])]
annotation_list = pd.DataFrame(condition_all_label, columns=['Disease'])
annotation_list['Sample_information'] = batch_info_all

annotation_list.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_nsclc_sclc/annotation_list.csv')