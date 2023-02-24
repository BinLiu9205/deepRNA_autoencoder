#!/bin/bash
beta_num=$1

mkdir -p /mnt/dzl_bioinf/binliu/deepRNA/beta_finetuning_results/results_for_beta_${beta_num}
echo "============ Beta coefficient: ${beta_num} ============="


python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed_adapted_finetunning.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl' -outputpath "/mnt/dzl_bioinf/binliu/deepRNA/beta_finetuning_results/results_for_beta_$beta_num" -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv' -modified_beta $beta_num -epochs 500 -data_transformation False
