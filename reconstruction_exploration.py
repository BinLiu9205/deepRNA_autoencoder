# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from adjustText import adjust_text
from collections import Counter
import os

# %%
input_df = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
output_df = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/reconstruction/gene level + beta_priorVAE_reconstruction_results.npy')

# %%
input_df  = input_df.T
output_df = output_df.T

# %%
anno = pd.read_csv('/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/sample_annotation_test.csv')

# %%
savepath_base = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/reconstruction_error_analysis'

# %% [markdown]
# # Gene level absolute error and relative error
# 
# ## Calculation, Run once only

# %%
#absolute_error = (input_df - output_df) ** 2
#mean_square_error = absolute_error.mean(axis=1)  
#average_expression = input_df.mean(axis=1)  
#relative_error = mean_square_error / average_expression


# %%
#correlations = input_df.corr()  # Transpose to get genes as columns, then calculate pairwise correlation

# For each gene, find the top 5 correlated genes (excluding itself), and take the average
#top_5_avg_correlation = correlations.apply(lambda x: x.nlargest(6).iloc[1:].mean(), axis=1)

# Max correlation per gene (excluding self-correlation)
#max_correlation = correlations.apply(lambda x: x.nlargest(2).iloc[1], axis=1)



# %%
#print(max_correlation.shape)
#print(top_5_avg_correlation.shape)
#print(relative_error.shape)
#print(absolute_error.shape)
#print(mean_square_error.shape)

# %%
#np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/max_correlation_gene_level.npy',max_correlation)
#np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/top5_avg_correlation_gene_level.npy',top_5_avg_correlation)
#np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/reletive_error_gene_level.npy',relative_error)
#np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/absolute_error_gene_level_sample.npy',absolute_error)
#np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/mean_absolute_error_gene_level_sample.npy',mean_square_error)

# %%
max_correlation = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/max_correlation_gene_level.npy')
relative_error = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/reletive_error_gene_level.npy')
absolute_error = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/absolute_error_gene_level_sample.npy')
top_5_avg_correlation = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/top5_avg_correlation_gene_level.npy')
mean_square_error = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/mean_absolute_error_gene_level_sample.npy')

# %%
#print(max_correlation.shape)
#print(top_5_avg_correlation.shape)
#print(relative_error.shape)
#print(absolute_error.shape)
#print(mean_square_error.shape)

# %%
mean_square_error = pd.Series(mean_square_error, index=input_df.index)
top_5_avg_correlation = pd.Series(top_5_avg_correlation, index=input_df.index)
N = 10 # Number of top genes you want to label
top_genes_by_abs_error = mean_square_error.nlargest(N).index

# %%
# Plot for Absolute Error vs. Input Noise Metric
fig1, ax1 = plt.subplots(figsize=(10,6))
color = 'tab:red'
ax1.set_xlabel('Input Noise Metric (Average of Top 5 Correlations)',color=color)
ax1.set_ylabel('Absolute Error', color=color)
ax1.scatter(top_5_avg_correlation, mean_square_error, color=color, alpha=0.3)
ax1.tick_params(axis='y', labelcolor=color)

shift_factor = 0.015
for i, gene in enumerate(top_genes_by_abs_error):
    x = top_5_avg_correlation[gene]
    y = mean_square_error[gene]
    # Slightly shift each subsequent label to reduce overlap
    ax1.annotate(gene, (x + (i % 2) * shift_factor, y + (i // 2) * shift_factor),
                 textcoords="offset points", xytext=(0,10), ha='center')

fig1.tight_layout()
plt.savefig(os.path.join(savepath_base, 'absolute_error_vs_input_noise.pdf'), dpi = 300)
plt.show()




# %%
relative_error = pd.Series(relative_error, index=input_df.index)
top_5_avg_correlation = pd.Series(top_5_avg_correlation, index=input_df.index)
N = 10 # Number of top genes you want to label
top_genes_by_rel_error = relative_error.nlargest(N).index

# %%
# Plot for Relative Error vs. Input Noise Metric
fig2, ax2 = plt.subplots(figsize=(10,6))
color = 'tab:blue'
ax2.set_xlabel('Input Noise Metric (Average of Top 5 Correlations)' , color=color)
ax2.set_ylabel('Relative Error', color=color)
ax2.scatter(top_5_avg_correlation, relative_error, color=color, alpha=0.3)
ax2.tick_params(axis='y', labelcolor=color)

shift_factor = 0.02
for i, gene in enumerate(top_genes_by_rel_error):
    x = top_5_avg_correlation[gene]
    y = relative_error[gene]
    # Slightly shift each subsequent label to reduce overlap
    ax2.annotate(gene, (x + (i % 4) * shift_factor, y + (i // 3) * shift_factor),
                 textcoords="offset points", xytext=(0,10), ha='center')

fig2.tight_layout()
plt.savefig(os.path.join(savepath_base, 'relative_error_vs_input_noise.pdf'), dpi = 300)
plt.show()

# %%
average_expression = input_df.mean(axis=1) 
average_expression = pd.DataFrame(average_expression, index=input_df.index)

# %%
average_expression.head(5)

# %%
relative_error = pd.Series(relative_error, index=input_df.index)
top_5_avg_correlation = pd.Series(top_5_avg_correlation, index=input_df.index)
N = 10 # Number of top genes you want to label
top_genes_by_rel_error = relative_error.nlargest(N).index


# %%
# Plot for Relative Error vs. Input Noise Metric
fig3, ax3 = plt.subplots(figsize=(10,6))
color = 'tab:blue'
ax3.set_xlabel('Expression Level', color=color)
ax3.set_ylabel('Relative Error', color=color)
ax3.scatter(average_expression, relative_error, color=color, alpha=0.3)
ax3.tick_params(axis='y', labelcolor=color)

shift_factor = 0.02
for i, gene in enumerate(top_genes_by_rel_error):
    y = relative_error[gene]
    x = average_expression.loc[gene,0]
    # Slightly shift each subsequent label to reduce overlap
    ax3.annotate(gene, (x + (i % 4) * shift_factor, y + (i // 3) * shift_factor),
                 textcoords="offset points", xytext=(0,10), ha='center')

fig3.tight_layout()
plt.savefig(os.path.join(savepath_base, 'relative_error_vs_expression_level.pdf'), dpi = 300)
plt.show()

# %%
mean_square_error = pd.Series(mean_square_error, index=input_df.index)
top_5_avg_correlation = pd.Series(top_5_avg_correlation, index=input_df.index)
N = 10 # Number of top genes you want to label
top_genes_by_abs_error = mean_square_error.nlargest(N).index

# %%
fig4, ax4 = plt.subplots(figsize=(10,6))
color = 'tab:red'
ax4.set_xlabel('Expression Level',color=color)
ax4.set_ylabel('Absolute Error', color=color)
ax4.scatter(average_expression, mean_square_error, color=color, alpha=0.3)
ax4.tick_params(axis='y', labelcolor=color)

shift_factor = 0.015
for i, gene in enumerate(top_genes_by_abs_error):
    x = average_expression.loc[gene,0]
    y = mean_square_error[gene]
    # Slightly shift each subsequent label to reduce overlap
    ax4.annotate(gene, (x + (i % 2) * shift_factor, y + (i // 2) * shift_factor),
                 textcoords="offset points", xytext=(0,10), ha='center')

fig4.tight_layout()
plt.savefig(os.path.join(savepath_base, 'absolute_error_vs_expression_level.pdf'), dpi = 300)
plt.show()

# %%
absolute_error = pd.DataFrame(absolute_error, index=input_df.index)

# %%
top_100_genes_by_relative_error = relative_error.sort_values(ascending=False).head(50).index

absolute_error_top_100 = absolute_error.loc[top_100_genes_by_relative_error]

# Generate heatmap
plt.figure(figsize=(20,10))  
sns.heatmap(absolute_error_top_100, cmap='coolwarm')  
#plt.title('Heatmap of Absolute Error for Top 50 Genes by Relative Error')
plt.savefig(os.path.join(savepath_base, 'absolute_error_across_sample_heatmap.pdf'), dpi = 300)
plt.show()

# %%
top_500_genes_by_relative_error = pd.DataFrame(relative_error.sort_values(ascending=False).head(500).index)
top_500_genes_by_relative_error.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/highest_relative_error_gene.csv',   header=False, index=False)

# %%
top_500_genes_by_absolute_error = pd.DataFrame(mean_square_error.sort_values(ascending=False).head(500).index)
top_500_genes_by_absolute_error.to_csv('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/genewise/highest_absolute_error_gene.csv', header=False, index=False)

# %% [markdown]
# Everything above is about the gene level expression and errors. 

# %% [markdown]
# # Sample_level exploration

# %%
input_df = pd.read_pickle('/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/gene_level_train_test/all_samples_gene_level_test.pkl')
output_df = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/reconstruction/gene level + beta_priorVAE_reconstruction_results.npy')

# %%
#absolute_error_sample = (input_df - output_df) ** 2
#mean_square_error_sample = absolute_error_sample.mean(axis=1)  
#average_expression_sample = input_df.mean(axis=1)  
#relative_error_sample = mean_square_error_sample / average_expression_sample

# %%
#print(mean_square_error_sample.shape)
#print(relative_error_sample.shape)
#print(absolute_error_sample.shape)

# %%
##np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/samplewise/reletive_error_gene_level_samplewise.npy',relative_error_sample)
#np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/samplewise/absolute_error_gene_level_samplewise.npy',absolute_error_sample)
#np.save('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/samplewise/mean_absolute_error_gene_level_samplewise.npy',mean_square_error_sample)

# %%
relative_error_sample = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/samplewise/reletive_error_gene_level_samplewise.npy')
absolute_error_sample = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/samplewise/absolute_error_gene_level_samplewise.npy')
mean_square_error_sample = np.load('/mnt/dzl_bioinf/binliu/deepRNA/major_revision/older_models/numeric_results/samplewise/mean_absolute_error_gene_level_samplewise.npy')

# %%
relative_error_sample = pd.Series(relative_error_sample, index=input_df.index)
absolute_error_sample = pd.DataFrame(absolute_error_sample, index=input_df.index)
mean_square_error_sample = pd.Series(mean_square_error_sample, index=input_df.index)

# %% [markdown]
# The idea of this section is to figure out whether some organs are prone to have higher errors

# %%
top_500_samples_by_relative_error_sample = relative_error_sample.sort_values(ascending=False).head(500).index

# %%
top_500_samples_by_relative_error_sample

# %%
anno.columns

# %%
organ_list = Counter(anno['Characteristics[organism part]'])

# %%
organ_list

# %%
anno_select = anno[anno['CompositeSequence Identifier'].isin(top_500_samples_by_relative_error_sample)]
organ_sub_list = Counter(anno_select['Characteristics[organism part]'])


# %%
organ_sub_list

# %%
final_frac = {}
for i in organ_sub_list.keys():
    final_frac[i] = round(organ_sub_list[i]/organ_list[i], 4)

# %%
final_frac.pop('  ', None)

# %%
sorted_frac = dict(sorted(Counter(final_frac).items(), key=lambda item: item[1], reverse=True))


# %%
plt.figure(figsize=(10, 8))
plt.bar(sorted_frac.keys(), sorted_frac.values())
plt.xlabel('Fraction')
plt.ylabel('Count')
plt.xticks(rotation=270)
plt.tight_layout()
plt.savefig(os.path.join(savepath_base, 'portion_top500_with_highest_error_organ_relationship.pdf'), dpi = 300)
plt.show()


