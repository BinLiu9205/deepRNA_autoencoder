import Preparing_samples_for_degs as Dea
import numpy as np
import pandas as pd
import os

input_test_gene = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
sample_annotation = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_test.csv'),index_col=0)
input_test_transcript = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')

# The first group of sample is from E-GEOD-15061, with 16 acute myeloid leukaemia and 10 normal samples
# The tissue of interest is the 'bone marrow'


con1_index = (sample_annotation['Comment [ArrayExpress accession]'] == 'E-GEOD-15061') & \
             (sample_annotation['Characteristics[disease]'].str.strip() == 'acute myeloid leukaemia') & \
             (sample_annotation['Characteristics[organism part]'].str.strip() == 'bone marrow')

con2_index = (sample_annotation['Comment [ArrayExpress accession]'] == 'E-GEOD-15061') & \
             (sample_annotation['Characteristics[disease]'].str.strip() == 'normal') & \
             (sample_annotation['Characteristics[organism part]'].str.strip() == 'bone marrow')
             
        
             
degs_input_gene, annotation_list = Dea.slicing_numpy_design_dea_input(input_df = input_test_gene, con1_index = con1_index, con2_index = con2_index, con1_label = 'acute_myeloid_leukaemia', con2_label = 'normal', label_name = 'Disease')

degs_input_transcript, _ = Dea.slicing_numpy_design_dea_input(input_df = input_test_transcript, con1_index = con1_index, con2_index = con2_index, con1_label = 'acute_myeloid_leukaemia', con2_label = 'normal', label_name = 'Disease')

degs_input_gene.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/microarray_input_gene_level.csv')

degs_input_transcript.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/microarray_input_transcript_level.csv')

annotation_list.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/annotation_list.csv')


## Have another version of putting all leukaemia samples together and normal together

input_test_gene = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
sample_annotation = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray', 'sample_annotation_test.csv'),index_col=0)
input_test_transcript = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')

# The first group of sample is from E-GEOD-15061, with 16 acute myeloid leukaemia and 10 normal samples
# The tissue of interest is the 'bone marrow'


con1_index = (sample_annotation['Characteristics[disease]'].str.strip() == 'acute myeloid leukaemia') & \
             (sample_annotation['Characteristics[organism part]'].str.strip() == 'bone marrow')

con2_index = (sample_annotation['Characteristics[disease]'].str.strip() == 'normal') & \
             (sample_annotation['Characteristics[organism part]'].str.strip() == 'bone marrow')
             
             
con1_batch_info = sample_annotation[con1_index]['Comment [ArrayExpress accession]']
con2_batch_info = sample_annotation[con2_index]['Comment [ArrayExpress accession]']
batch_info_all = pd.concat([con1_batch_info, con2_batch_info], axis=0)
batch_info_all = batch_info_all.reset_index(drop=True)
             
             
             
degs_input_gene, annotation_list = Dea.slicing_numpy_design_dea_input(input_df = input_test_gene, con1_index = con1_index, con2_index = con2_index, con1_label = 'acute_myeloid_leukaemia', con2_label = 'normal', label_name = 'Disease')

degs_input_transcript, _ = Dea.slicing_numpy_design_dea_input(input_df = input_test_transcript, con1_index = con1_index, con2_index = con2_index, con1_label = 'acute_myeloid_leukaemia', con2_label = 'normal', label_name = 'Disease')

annotation_list['Sample_information'] = batch_info_all

os.makedirs('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/', exist_ok=True)

degs_input_gene.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/microarray_input_gene_level.csv')

degs_input_transcript.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/microarray_input_transcript_level.csv')

annotation_list.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia_multiple_geos/annotation_list.csv')