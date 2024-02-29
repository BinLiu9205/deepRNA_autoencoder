import DEG_script_statistical_versions as Deg
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
             
             

ttest_res_gene, ttest_sig_gene = Deg.deg_pairwised_ttest_version(input_test_gene, con1_index, con2_index)
print(ttest_sig_gene.shape)
ttest_res_transcript, ttest_sig_transcript = Deg.deg_pairwised_ttest_version(input_test_transcript, con1_index, con2_index)
print(ttest_sig_transcript.shape)

conditions =  [1] * con1_index.sum() + [0] *  con2_index.sum()  # 0 for normal, 1 for disease
intercept = [1] * (con1_index.sum() + con2_index.sum())  # Intercept column

design_matrix = pd.DataFrame({'Intercept': intercept, 'Condition': conditions})

ols_res_gene, ols_sig_gene = Deg.deg_pairwised_linear_model_version(input_test_gene, con1_index, con2_index, design_matrix)
print(ols_sig_gene.shape)
ols_res_transcript, ols_sig_transcript = Deg.deg_pairwised_linear_model_version(input_test_transcript, con1_index, con2_index, design_matrix)
print(ols_sig_transcript.shape)

ttest_res_gene.to_csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/deg_results_ttest_gene_level.csv")
ttest_res_transcript.to_csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/deg_results_ttest_transcript_level.csv")
ols_res_gene.to_csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/deg_results_ols_gene_level.csv")
ols_res_transcript.to_csv("/mnt/dzl_bioinf/binliu/deepRNA/major_revision/numeric_results/degs_analysis_input_leukaemia/deg_results_ols_transcript_level.csv")