#python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 400 -data_transformation False
#python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 400 -data_transformation False
#python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_transcript_level.csv" -modified_beta 250 -epochs 400 -data_transformation False

## Remove early stopping
#python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 200 -data_transformation False
#python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 200 -data_transformation False
#python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_transcript_level.csv" -modified_beta 250 -epochs 200 -data_transformation False

## Increase the number of epochs
## Remove early stopping
# python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 500 -data_transformation False
# python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 500 -data_transformation False
# python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_transcript_level.csv" -modified_beta 250 -epochs 500 -data_transformation False


## Increase the number of epochs to 500
## Remove early stopping
## Double check whether getting the wrong results of mu and sigma 

# python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed_sigma_mu_available.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 500 -data_transformation False

python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed_sigma_mu_available.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_gene_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_gene_level.csv" -modified_beta 250 -epochs 500 -data_transformation False

python /mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/models/settings_early_stopping_removed_sigma_mu_available.py ['simpleAE','priorVAE','beta_priorVAE','simpleVAE','beta_simpleVAE'] -inputdata_train '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl' -inputdata_test '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl' -outputpath '/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training' -prior_file_mu "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_transcript_level.csv" -prior_file_sigma "/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_sigma_MsigDB_transcript_level.csv" -modified_beta 250 -epochs 500 -data_transformation False