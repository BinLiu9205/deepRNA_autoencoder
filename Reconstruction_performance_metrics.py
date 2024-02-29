# Working on the reconstruction performance across different models
import numpy as np
import pandas as pd
import sys
import os
import argparse
import seaborn as sns
import scipy.stats
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from random import random
import umap
import umap.plot
from collections  import Counter
from sklearn.model_selection import train_test_split
seed = 42
np.random.seed(seed)
import matplotlib.patches as mpatches



## Calculating the correlation between input and reconstruction
## Since we measure the samplewise correlation, we do the calculation based on rows
## The matrices have to be transposed beforehand
def calculate_overall_reconstruction_each_model(input_df, reconstruction_df):
    input_array = np.array(input_df)
    reconstruction_array = np.array(reconstruction_df)
    correlation_array = []
    length_trial = input_array.shape[0]
    for i in range(length_trial):
        correlation_array.append(np.corrcoef(input_array[i, :], reconstruction_array[i, :])[0, 1])
    correlation_mean = np.nanmean(correlation_array)
    correlation_std = np.nanstd(correlation_array)

    if np.isnan(correlation_mean) or np.isnan(correlation_std):
        correlation_mean, correlation_std = 0, 0
    return correlation_mean, correlation_std

def calculate_overall_reconstruction_each_model_detailed_info(input_df, reconstruction_df):
    input_array = np.array(input_df)
    reconstruction_array = np.array(reconstruction_df)
    correlation_array = []
    length_trial = input_array.shape[0]
    for i in range(length_trial):
        correlation_array.append(np.corrcoef(input_array[i, :], reconstruction_array[i, :])[0, 1])
    df_values = {}
    df_values['Values'] = correlation_array
    df = pd.DataFrame(df_values)

    min_val = df['Values'].min()
    q1 = df['Values'].quantile(0.25)
    median = df['Values'].median()
    q3 = df['Values'].quantile(0.75)
    max_val = df['Values'].max()
    iqr = q3 - q1


    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df['Values'] < lower_bound) | (df['Values'] > upper_bound)]['Values'].tolist()
    
    return min_val, q1, median, q3, max_val, outliers


def calculate_sample_wise_correlation_numpy(input_df, reconstruction_df):
    # Ensure inputs are numpy arrays
    input_array = np.array(input_df)
    reconstruction_array = np.array(reconstruction_df)
    length_trial = input_array.shape[0]
    correlation_matrix = np.zeros((length_trial, length_trial))

    for i in range(length_trial):  
        for j in range(length_trial):  
            correlation_matrix[i, j] = np.corrcoef(input_array[i, :], reconstruction_array[j, :])[0, 1]
    return correlation_matrix

    


## Traditional reconstruction plot (a few inputs + their reconstructed outputs) -- pairwise + no_label
def input_reconstruction_pairwise_correlation_plot(input_df, reconstruction_df, save_path, length_trial = 20, sample_method = 'non_random'):
    length_trial = int(length_trial)
    if sample_method == 'non_random':
        random_indices = range(0,length_trial)
    elif sample_method == 'random':
        range_size = reconstruction_df.shape[1]
        random_indices = random.sample(range(0, range_size), length_trial)
    else:
        raise ValueError("Sample method has to be random or non_random")
    
    input_sub = input_df[random_indices, :]
    reconstruction_sub = reconstruction_df[random_indices, :]
    merged_sub = np.vstack((input_sub, reconstruction_sub))
    
    train_samples = ['_'.join([str(i+1), 'Input']) for i in range(length_trial)]
    recon_samples = ['_'.join([str(i+1), 'Recon']) for i in range(length_trial)]
    all_samples = [*train_samples, *recon_samples]

    correlation_matrix = np.zeros((2*length_trial, 2*length_trial))

    for i in range(2*length_trial):  
        for j in range(2*length_trial):  
            correlation_matrix[i, j] = np.corrcoef(merged_sub[i, :], merged_sub[j, :])[0, 1]

    
    plt.figure(figsize=(17, 17))
    try:
        sns.clustermap(correlation_matrix, row_cluster=True, col_cluster=True, xticklabels=all_samples, yticklabels=all_samples,cmap='YlGnBu')
        plt.savefig(save_path + 'no_label_clustering.eps')
        plt.show()
    # Some of the plots cannot be clustered -- then remove the cluster part
    except Exception as e:
        print("Clustering failed:", e)
    # Try again without clustering
        sns.clustermap(correlation_matrix, row_cluster=False, col_cluster=False, xticklabels=all_samples, yticklabels=all_samples,cmap='YlGnBu')
        plt.savefig(save_path + '_without_cluster_cluster_not_working.eps')
        plt.show()
    plt.close() 
    return 0

## New reconstruction plot (having tissue specific information)
def input_reconstruction_pairwise_correlation_with_annotation_labels(input_df, reconstruction_df, anno_array, save_path, percentage_trial = 0.1):
    percentage_trial = float(percentage_trial)

    labels = np.array(anno_array) 
    class_counts = Counter(labels)


    organ_freq = Counter(anno_array)
    top_organs = [organ for organ, count in organ_freq.most_common(7)[1:]]
    classes_to_keep = [label for label, count in class_counts.items() if label in top_organs]


    mask = np.array([label in classes_to_keep for label in labels])


    filtered_input_data = input_df[mask]
    filtered_recon_data = reconstruction_df[mask]
    filtered_labels = labels[mask]
    _, random_indices = train_test_split(
    np.arange(len(filtered_labels)),  
    test_size=percentage_trial, 
    stratify=filtered_labels, 
    random_state=42
    )
    sorted_random_indices = sorted(random_indices, key=lambda i: anno_array[i])
    input_sub = filtered_input_data[sorted_random_indices, :]
    reconstruction_sub = filtered_recon_data[sorted_random_indices, :]
    length_trial = len(sorted_random_indices)
    correlation_matrix = np.zeros((length_trial, length_trial))

    for i in range(length_trial):  
        for j in range(length_trial):  
            correlation_matrix[i, j] = np.corrcoef(input_sub[i, :], reconstruction_sub[j, :])[0, 1]

    annotation_labels = filtered_labels[sorted_random_indices]
    filtered_labels_enumerate = list(range(1, len(annotation_labels) + 1))
    filtered_labels_enumerate = np.array(filtered_labels_enumerate)
    
    annotations_df = pd.DataFrame({'Annotations': annotation_labels})
    unique_labels = annotations_df['Annotations'].unique()
    palette = sns.color_palette("hsv", len(unique_labels))
    label_to_color = dict(zip(unique_labels, palette))
    annotations_df['Color'] = annotations_df['Annotations'].map(label_to_color)
    row_colors = annotations_df['Color'].to_numpy()
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]
    
    
    plt.figure(figsize=(17, 17))
    try:
        sns.clustermap(correlation_matrix,
                row_cluster=True,
                col_cluster=True,
                xticklabels=False,
                yticklabels=False, 
                cmap='coolwarm',   
                row_colors=row_colors,  
                col_colors=row_colors,  
                annot=False, 
                cbar_kws={'label': 'Correlation'}) 
        plt.legend(handles=legend_handles, title='Annotations', bbox_to_anchor=(2, 1), loc='upper left')
        plt.savefig(save_path + '_tissue_specific_heatmap_clustered.eps')
        plt.close()
    except Exception as e:
        print("Clustering failed:", e)
        sns.clustermap(correlation_matrix,
                row_cluster=False,
                col_cluster=False,
                xticklabels=False,
                yticklabels=False, 
                cmap='coolwarm',   
                row_colors=row_colors,  
                col_colors=row_colors,  
                annot=False, 
                cbar_kws={'label': 'Correlation'}) 
        plt.legend(handles=legend_handles, title='Annotations', bbox_to_anchor=(2, 1), loc='upper left')
        plt.savefig(save_path + '_tissue_specific_heatmap_not)clustered_NaN_similarity.eps')
        plt.close()
        
        
        plt.figure(figsize=(17, 17))
    try:
        g = sns.clustermap(correlation_matrix,
                row_cluster=True,
                col_cluster=True,
                xticklabels=filtered_labels_enumerate,
                yticklabels=filtered_labels_enumerate, 
                cmap='coolwarm',   
                row_colors=row_colors,  
                col_colors=row_colors,  
                annot=False, 
                cbar_kws={'label': 'Correlation'}) 
        plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=5)  
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=5)
        plt.legend(handles=legend_handles, title='Annotations', bbox_to_anchor=(2, 1), loc='upper left')
        plt.savefig(save_path + '_tissue_specific_heatmap_clustered_tissue_enumerated.eps')
        plt.close()
    except Exception as e:
        print("Clustering failed:", e)
        g = sns.clustermap(correlation_matrix,
                row_cluster=False,
                col_cluster=False,
                xticklabels=filtered_labels_enumerate,
                yticklabels=filtered_labels_enumerate, 
                cmap='coolwarm',   
                row_colors=row_colors,  
                col_colors=row_colors,  
                annot=False, 
                cbar_kws={'label': 'Correlation'}) 
        plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=5)  
        plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=5)
        plt.legend(handles=legend_handles, title='Annotations', bbox_to_anchor=(2, 1), loc='upper left')
        plt.savefig(save_path + '_tissue_specific_heatmap_not)clustered_NaN_similarity_tissue_enumerated.eps')
        plt.close()
    
    
    organ_ordering = {organ: i for i, organ in enumerate(top_organs)}

    # Apply mask and get filtered data
    filtered_labels = labels[mask]
    filtered_input_data = input_df[mask]  
    filtered_recon_data = reconstruction_df[mask]
    # Sort based on organ label order
    sorting_index = np.argsort([organ_ordering.get(label, -1) for label in filtered_labels])
    sorted_labels = filtered_labels[sorting_index]
    sorted_input_data = filtered_input_data[sorting_index]
    sorted_recon_data = filtered_recon_data[sorting_index]
    annotation_labels_enumerate = list(range(1, len(sorted_labels) + 1))
    annotation_labels_enumerate = np.array(annotation_labels_enumerate)
    unique_labels = np.unique(sorted_labels)  

    sorted_indices = []
    for label in unique_labels:
        # Find indices of all samples with the current label
        label_indices = np.where(sorted_labels == label)[0]
        
        # Calculate how many samples to select based on the desired percentage
        num_to_select = int(len(label_indices) * percentage_trial)
        sorted_indices.extend(label_indices[:num_to_select])


    sorted_random_indices = np.array(sorted_indices)

    input_sub = sorted_input_data[sorted_random_indices, :]
    reconstruction_sub = sorted_recon_data[sorted_random_indices, :]
    length_trial = len(sorted_random_indices)

    correlation_matrix_sorted = np.zeros((length_trial, length_trial))

    for i in range(length_trial):  
        for j in range(length_trial):  
            correlation_matrix_sorted[i, j] = np.corrcoef(input_sub[i, :], reconstruction_sub[j, :])[0, 1]

    # Generate annotations DataFrame
    annotation_labels = sorted_labels[sorted_random_indices]
    annotations_df = pd.DataFrame({'Annotations': annotation_labels})
    unique_labels = annotations_df['Annotations'].unique()
    palette = sns.color_palette("hsv", len(unique_labels))
    label_to_color = dict(zip(unique_labels, palette))
    annotations_df['Color'] = annotations_df['Annotations'].map(label_to_color)
    row_colors = annotations_df['Color'].to_numpy()

    # Plotting the heatmap
    plt.figure(figsize=(17, 17))
    sns.clustermap(correlation_matrix_sorted, cmap='coolwarm', annot=False, cbar_kws={'label': 'Correlation'}, yticklabels=False, xticklabels=False,               row_colors=row_colors,  
    col_colors=row_colors,               
    row_cluster=False,
    col_cluster=False)

    # Adding a legend for annotations
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]
    plt.legend(handles=legend_handles, title='Annotations', bbox_to_anchor=(2, 1), loc='upper left')

    plt.savefig(save_path + '_tissue_specific_heatmap_not_clustered.eps', bbox_inches='tight')
    plt.close()
    
    
    # Plotting the heatmap, with organ labels enumerated
    plt.figure(figsize=(17, 17))
    g = sns.clustermap(correlation_matrix_sorted, cmap='coolwarm', annot=False, cbar_kws={'label': 'Correlation'}, yticklabels=annotation_labels_enumerate, xticklabels=annotation_labels_enumerate,               row_colors=row_colors,  
    col_colors=row_colors,               
    row_cluster=False,
    col_cluster=False)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=5)  
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=5)
# Adding a legend for annotations
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]
    plt.legend(handles=legend_handles, title='Annotations', bbox_to_anchor=(2, 1), loc='upper left')

    plt.savefig(save_path + '_tissue_specific_heatmap_not_clustered_enumerated.eps', bbox_inches='tight')
    plt.close()
    


    return 0
    
    
def input_reconstruction_tsne_plot_most_frequent_organs(input_df, reconstruction_df, anno_array, save_path, frequent_num = 8):
    '''
    This expects a reconstruction based on the most frequent organs. The default number is 8 (the most common one is the missing label in most cases) -- will filter out if there are more than eight organs
    '''
    frequent_num  = int(frequent_num)
    organ_freq = Counter(anno_array)
    if len(organ_freq) > int(frequent_num):
        top_organs = [organ for organ, count in organ_freq.most_common(frequent_num+1)[1:]]
    else:
        top_organs = [organ for organ, count in organ_freq.most_common(len(organ_freq)+1)[1:]]
        
    top_organ_indices = [index for index, organ in enumerate(anno_array) if organ in top_organs]
    filtered_anno_array= np.array([anno_array[i] for i in top_organ_indices])
    sample_interest_input = input_df[top_organ_indices, :]
    sample_interest_reconstruction = reconstruction_df[top_organ_indices, :]
    contains_invalid_reconstruction = np.isnan(sample_interest_reconstruction).any() or np.isinf(sample_interest_reconstruction).any()
    if contains_invalid_reconstruction:
        print('Error in reconstruction')
    else:
        input_tsne = TSNE(n_components=2, init='pca').fit_transform(sample_interest_input)
        reconstruction_tsne = TSNE(n_components=2, init='pca').fit_transform(sample_interest_reconstruction)
        
        ## Map the colors
        
        unique_organs = sorted(set(filtered_anno_array))  # Sort to ensure consistent colors across plots
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_organs)))
        color_map = dict(zip(unique_organs, colors))
        
        scatter_size  = 10
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
        
        for organ in unique_organs:
            indices = np.where(filtered_anno_array == organ)[0]
            axs[0].scatter(input_tsne[indices, 0], input_tsne[indices, 1], label=organ, color=color_map[organ], s=scatter_size)

        axs[0].set_title('Original Input t-SNE')
        axs[0].set_xlabel('t-SNE 1')
        axs[0].set_ylabel('t-SNE 2')
        
        for organ in unique_organs:
            indices = np.where(filtered_anno_array == organ)[0]
            axs[1].scatter(reconstruction_tsne[indices, 0], reconstruction_tsne[indices, 1], label=organ, color=color_map[organ], s=scatter_size)

        axs[1].set_title('Reconstructed Input t-SNE')
        axs[1].set_xlabel('t-SNE 1')
        axs[1].set_ylabel('t-SNE 2')
        
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=organ,
                                    markerfacecolor=color, markersize=10) for organ, color in color_map.items()]
        fig.legend(handles=legend_handles, title='Organ', loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')

        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        
        plt.savefig(save_path + '_tsne_plots_original_reconstructed.eps', bbox_inches='tight')
        plt.show()
        plt.close()
    
    return 0
    
    
    
def input_reconstruction_umap_plot_most_frequent_organs(input_df, reconstruction_df, anno_array, save_path, frequent_num = 8):
    '''
    This expects a reconstruction based on the most frequent organs. The default number is 8 (the most common one is the missing label in most cases) -- will filter out if there are more than eight organs
    '''
    frequent_num  = int(frequent_num)
    organ_freq = Counter(anno_array)
    if len(organ_freq) > int(frequent_num):
        top_organs = [organ for organ, count in organ_freq.most_common(frequent_num+1)[1:]]
    else:
        top_organs = [organ for organ, count in organ_freq.most_common(len(organ_freq)+1)[1:]]
        
    top_organ_indices = [index for index, organ in enumerate(anno_array) if organ in top_organs]
    sample_interest_input = input_df[top_organ_indices, :]
    sample_interest_reconstruction = reconstruction_df[top_organ_indices, :]
    reducer = umap.UMAP(random_state=42)
    filtered_anno_array = np.array([anno_array[i] for i in top_organ_indices])
    contains_invalid_reconstruction = np.isnan(sample_interest_reconstruction).any() or np.isinf(sample_interest_reconstruction).any()
    if contains_invalid_reconstruction:
        print('Error in reconstruction')
    else:
        input_umap = reducer.fit_transform(sample_interest_input)
        reconstruction_umap = reducer.fit_transform(sample_interest_reconstruction)
        
        ## Map the colors
        scatter_size = 10
        
        unique_organs = sorted(set(filtered_anno_array))  # Sort to ensure consistent colors across plots
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_organs)))
        color_map = dict(zip(unique_organs, colors))
        
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
        
        for organ in unique_organs:
            indices = np.where(filtered_anno_array == organ)[0]
            axs[0].scatter(input_umap[indices, 0], input_umap[indices, 1], label=organ, color=color_map[organ], s=scatter_size)

        axs[0].set_title('Original Input UMAP')
        axs[0].set_xlabel('UMAP 1')
        axs[0].set_ylabel('UMAP 2')
        
        for organ in unique_organs:
            indices = np.where(filtered_anno_array == organ)[0]
            axs[1].scatter(reconstruction_umap[indices, 0], reconstruction_umap[indices, 1], label=organ, color=color_map[organ], s=scatter_size)

        axs[1].set_title('Reconstructed Input UMAP')
        axs[1].set_xlabel('UMAP 1')
        axs[1].set_ylabel('UMAP 2')
        
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=organ,
                                    markerfacecolor=color, markersize=10) for organ, color in color_map.items()]
        fig.legend(handles=legend_handles, title='Organ', loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')

        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        
        plt.savefig(save_path + '_umap_plots_original_reconstructed.eps', bbox_inches='tight')
        plt.show()
        plt.close()
    
    return 0
    