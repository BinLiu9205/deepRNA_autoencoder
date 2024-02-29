import Preparing_samples_for_degs as Dea
import numpy as np
import pandas as pd
import os

input_test_gene = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
sample_annotation = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_test.csv'),index_col=0)
input_test_transcript = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')
#input_train_gene = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl')
#input_train_transcript = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl')
#sample_annotation_train = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_train.csv'),index_col=0)
# Added one-time line on 21.02.2024 since I want the train annotation to be in the same order as the samples
#sorted_sample_annotation_train = sample_annotation_train.loc[input_train_gene.index]
#sorted_sample_annotation_train.to_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_train.csv'))
#print(sorted_sample_annotation_train.index[0:3])
#print(input_train_gene.index[0:3])
#print(input_train_transcript.index[0:3])



# Checking the different conditions 

## Lung cancer vs healthy 

con1_index = (sample_annotation['Characteristics[cell type]'] == 'lung adenocarcinoma cell line')
con2_index = (sample_annotation['Characteristics[disease]'] == 'normal') & (sample_annotation['Characteristics[organism part]'] == 'lung')



con1_batch_info = sample_annotation[con1_index]['Comment [ArrayExpress accession]']
con2_batch_info = sample_annotation[con2_index]['Comment [ArrayExpress accession]']
batch_info_all = pd.concat([con1_batch_info, con2_batch_info], axis=0)
batch_info_all = batch_info_all.reset_index(drop=True)
             
             
             
degs_input_gene, annotation_list = Dea.slicing_numpy_design_dea_input(input_df = input_test_gene, con1_index = con1_index, con2_index = con2_index, con1_label = 'cancer', con2_label = 'health', label_name = 'Disease')

degs_input_transcript, _ = Dea.slicing_numpy_design_dea_input(input_df = input_test_transcript, con1_index = con1_index, con2_index = con2_index, con1_label = 'cancer', con2_label = 'health', label_name = 'Disease')

annotation_list['Sample_information'] = batch_info_all

#os.makedirs('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/', exist_ok=True)

degs_input_gene.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_cancer_health/microarray_input_gene_level.csv')

degs_input_transcript.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_cancer_health/microarray_input_transcript_level.csv')

annotation_list.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_cancer_health/annotation_list.csv')



## Breast cancer vs lung cancer
con1_index = (sample_annotation['Factor Value[cell type]'] == 'breast cancer cell line') & (sample_annotation['Characteristics[disease]'] == 'breast adenocarcinoma')
con2_index = (sample_annotation['Factor Value[cell type]'] == 'lung adenocarcinoma cell line')

con1_batch_info = sample_annotation[con1_index]['Comment [ArrayExpress accession]']
con2_batch_info = sample_annotation[con2_index]['Comment [ArrayExpress accession]']
batch_info_all = pd.concat([con1_batch_info, con2_batch_info], axis=0)
batch_info_all = batch_info_all.reset_index(drop=True)
             
             
             
degs_input_gene, annotation_list = Dea.slicing_numpy_design_dea_input(input_df = input_test_gene, con1_index = con1_index, con2_index = con2_index, con1_label = 'breast', con2_label = 'lung', label_name = 'Disease')

degs_input_transcript, _ = Dea.slicing_numpy_design_dea_input(input_df = input_test_transcript, con1_index = con1_index, con2_index = con2_index, con1_label = 'breast', con2_label = 'lung', label_name = 'Disease')

annotation_list['Sample_information'] = batch_info_all

#os.makedirs('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/', exist_ok=True)

degs_input_gene.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_breast_lung_cancer/microarray_input_gene_level.csv')

degs_input_transcript.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_breast_lung_cancer/microarray_input_transcript_level.csv')

annotation_list.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_breast_lung_cancer/annotation_list.csv')

