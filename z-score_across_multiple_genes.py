import pandas as pd
from scipy.stats import zscore

# Load the data
data = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_train.pkl')  # Assuming the first column contains gene names/identifiers
# Apply Z-score normalization across genes (rows)
data_normalized = data.T.apply(zscore).T
data_normalized.to_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_train_zscore.pkl')

# Load the data
data = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')  # Assuming the first column contains gene names/identifiers
# Apply Z-score normalization across genes (rows)
data_normalized = data.T.apply(zscore).T
data_normalized.to_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test_zscore/all_samples_gene_level_test_zscore.pkl')

# Load the data
data = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_train.pkl')  # Assuming the first column contains gene names/identifiers
# Apply Z-score normalization across genes (rows)
data_normalized = data.T.apply(zscore).T
data_normalized.to_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test_zscore/all_samples_transcript_level_train_zscore.pkl')

# Load the data
data = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test/all_samples_transcript_level_test.pkl')  # Assuming the first column contains gene names/identifiers
# Apply Z-score normalization across genes (rows)
data_normalized = data.T.apply(zscore).T
data_normalized.to_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/transcript_level_train_test_zscore/all_samples_transcript_level_test_zscore.pkl')
