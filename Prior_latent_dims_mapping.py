import numpy as np
import matplotlib.pyplot as plt


# This function calculates how each latent dimension (pathway or wildcard) correlates with the given prior
# The way we use can get pairwised correlation across multiple dimensions, but the mapped ones are of the greatest importance
def dimensional_prior_latent_correlation_calculation(prior_matrix, latent_matrix):
    print(prior_matrix.shape)
    combined_matrix = np.vstack((prior_matrix.transpose(), latent_matrix.transpose()))
    # Calculate the correlation coefficient matrix
    correlation_matrix = np.corrcoef(combined_matrix)
    # Extract the correlation coefficients for the inter-matrix comparisons
    num_rows_matrix1 = prior_matrix.shape[1]
    # The correlation matrix consists of four main sections, the one between input and reconstruction belongs to [:num_rows_matrix1, num_rows_matrix1:]
    # The input matrix should remain the same with the same input data, but not the case with the output one and the inter one
    correlation_inter_matrix = correlation_matrix[:num_rows_matrix1, num_rows_matrix1:]
    correlation_paired_matrix = np.diag(correlation_inter_matrix)

    return correlation_paired_matrix

# Draw a boxplot of correlation coefficients across all the dimensions (pathways) 
# Important for comparing models/ beta parameters (e.g., in a sweep) 
# This returns one boxplot, should be used with for loop or lambda apply together to get a figure
def prior_latent_correlation_boxplot(prior_matrix, latent_matrix, annotation, ax, sequence_num, correlation_matrix = None):
    
    if correlation_matrix is not None:
        mapped_correlation_matrix = np.diag(correlation_matrix)
        ax.boxplot(mapped_correlation_matrix)
        ax.set_title(annotation)
    else: 
        combined_matrix = np.vstack((prior_matrix.transpose(), latent_matrix.transpose()))
        # Calculate the correlation coefficient matrix
        correlation_matrix = np.corrcoef(combined_matrix)
        # Extract the correlation coefficients for the inter-matrix comparisons
        num_rows_matrix1 = prior_matrix.shape[1]
        # The correlation matrix consists of four main sections, the one between input and reconstruction belongs to [:num_rows_matrix1, num_rows_matrix1:]
        # The input matrix should remain the same with the same input data, but not the case with the output one and the inter one
        correlation_inter_matrix = correlation_matrix[:num_rows_matrix1, num_rows_matrix1:]
        mapped_correlation_matrix = np.diag(correlation_inter_matrix)
        #print((annotation, np.min(mapped_correlation_matrix), np.max(mapped_correlation_matrix),np.mean(mapped_correlation_matrix), mapped_correlation_matrix.shape))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if sequence_num > 0:
            ax.spines['left'].set_visible(False)
            ax.set_yticks([])
        ax.boxplot(mapped_correlation_matrix, flierprops={'markersize': 0.5})
        #print(annotation)
        ax.set_xticklabels([annotation])
    return ax        


def prior_latent_direct_scatterplot(prior_matrix, latent_matrix,  annotation_pathway, savepath, fig_col_param = 5, fig_row_param = 10):
    fig_col = fig_col_param
    fig_row = fig_row_param
    fig = plt.figure(figsize=(25, 18))
    plt.rcParams.update({'font.size': 8})
    plt.subplots_adjust(top=0.8, bottom=0.4)
    for i in range(prior_matrix.shape[1]):
        fig.add_subplot(fig_row, fig_col, (i+1))
        plt.scatter(prior_matrix.iloc[:, i], latent_matrix[:, i],
                    c="#E64B35", s=12, alpha=0.25, rasterized=True)
        plt.title(annotation_pathway[i])
    plt.gcf().set_size_inches(16, 14)
    fig.tight_layout()
    plt.savefig(savepath, dpi=300)
    plt.clf()
    return 