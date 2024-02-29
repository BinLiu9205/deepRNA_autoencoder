import pandas as pd
import numpy as np
import umap
import umap.plot
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from pca import pca
from collections import Counter
from itertools import repeat
import math
from scipy.stats import ttest_ind as ttest_ind


'''
For some of the columns, those with the highest frequency of labels are '' or missing ones, so we set the remove_top default to True
'''
def pca_plots_directly_based_on_annotation(latent_res, anno_array, anno_name, save_path, frequent_num = 8, remove_top = True):
    frequent_num  = int(frequent_num)
    organ_freq = Counter(anno_array)
    if len(organ_freq) > int(frequent_num):
        top_organs = [organ for organ, count in organ_freq.most_common(frequent_num+1)[1:]]
    else:
        top_organs = [organ for organ, count in organ_freq.most_common(len(organ_freq)+1)[1:]]
        
    top_organ_indices = [index for index, organ in enumerate(anno_array) if organ in top_organs]
    filtered_anno_array = np.array([anno_array[i] for i in top_organ_indices])
    sample_interest_latent = latent_res[top_organ_indices, :]
    contains_invalid_latent = np.isnan(sample_interest_latent).any() or np.isinf(sample_interest_latent).any()
    if contains_invalid_latent:
        print('Error in latent dimensions')
    else:
        pca_model1 = PCA(n_components=2)
        pca_latent_selected = pca_model1.fit_transform(sample_interest_latent)

        scatter_size = 10
        
        unique_organs = sorted(set(filtered_anno_array)) 
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_organs)))
        color_map = dict(zip(unique_organs, colors))
        
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=13)
        ax.set_ylabel('Principal Component 2', fontsize=13)
        
        for organ in unique_organs:
            indices = np.where(filtered_anno_array == organ)[0]
            ax.scatter(pca_latent_selected[indices, 0], pca_latent_selected[indices, 1], label=organ, color=color_map[organ], s=scatter_size)
        
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=organ,
                                     markerfacecolor=color, markersize=10) for organ, color in color_map.items()]
        fig.legend(handles=legend_handles, title='Organ', loc='upper right', bbox_to_anchor=(1.1, 1), fontsize='small')

        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        
        plt.savefig(save_path + '_PCA_plots_latent_dimension_for_' + anno_name + '.eps', bbox_inches='tight')
        plt.show()
        plt.close()
        
'''
The path_anno defines the annotation for latent dimensions, it can be either pathways from KEGG (149 + wd, or 149)/ MSigDB (50), or numbers for those which are not defined by the priors
'''        
def pca_biplots_directly_based_on_annotation(latent_res, anno_array, anno_name, path_anno, save_path, frequent_num = 8, remove_top = True, n_feat = 10):
    frequent_num  = int(frequent_num)
    organ_freq = Counter(anno_array)
    if len(organ_freq) > int(frequent_num):
        top_organs = [organ for organ, count in organ_freq.most_common(frequent_num+1)[1:]]
    else:
        top_organs = [organ for organ, count in organ_freq.most_common(len(organ_freq)+1)[1:]]
        
    top_organ_indices = [index for index, organ in enumerate(anno_array) if organ in top_organs]
    filtered_anno_array = np.array([anno_array[i] for i in top_organ_indices])
    filtered_anno_list = [anno_array[i] for i in top_organ_indices]
    sample_interest_latent = latent_res[top_organ_indices, :]
    contains_invalid_latent = np.isnan(sample_interest_latent).any() or np.isinf(sample_interest_latent).any()
    if contains_invalid_latent:
        print('Error in latent dimensions')
    else:
        pca_model = pca(n_components=2)
        pca_latent_selected = pca_model.fit_transform(
            sample_interest_latent, row_labels=filtered_anno_list, col_labels=path_anno)
        fig, ax = pca_model.biplot(n_feat=n_feat, cmap=False, label=False, legend=False)
        plt.savefig(save_path + '_biplots_plots_latent_dimension_for_with_' + str(n_feat) + '_features_' + anno_name + '.eps', bbox_inches='tight')
        plt.show()
        plt.close()


def pca_biplots_based_on_selected_indices(latent_res, con1_index, con2_index, con1_label, con2_label, path_anno, save_path, n_feat=10):
    res_selected_all = latent_res[con1_index.append(con2_index)]
    condition_all = [*list(repeat(con1_label, len(con1_index))), *list(
        repeat(con2_label, len(con2_index)))]
    contains_invalid_latent = np.isnan(res_selected_all).any() or np.isinf(res_selected_all).any()
    if contains_invalid_latent:
        print('Error in latent dimensions')
    else:
        pca_model = pca(n_components=2)
        pca_latent_selected = pca_model.fit_transform(
                res_selected_all, row_labels=condition_all, col_labels=path_anno)
        scatter_size = 10
        #fig, ax = pca_model.plot()
        #fig, ax = pca_model.scatter()
        fig, ax = pca_model.biplot(n_feat=n_feat)
        plt.gcf().set_size_inches(6, 5.6)
        plt.tight_layout()
        
        plt.savefig(save_path + '_biplots_plots_latent_dimension_for_with_' + str(n_feat) + '_features_' + con1_label + '_vs_' + con2_label + '.eps')
        plt.show()
        plt.close()
    return 'Done'
    
    
def differential_expressed_latent_dimensions_with_given_indices(latent_res, prior_res, con1_index, con2_index, con1_label, con2_label, path_anno, save_path, threshold = 50):
    con1_sample = latent_res[con1_index,:]
    con2_sample = latent_res[con2_index,:]
    sample_interest_latent = np.vstack((con1_sample, con2_sample))
    contains_invalid_latent = np.isnan(sample_interest_latent).any() or np.isinf(sample_interest_latent).any()
    result_dict = {}
    result_dict['pathway_name'] = path_anno
    if contains_invalid_latent:
        print('Error in latent dimensions')
        result_df = pd.DataFrame(result_dict)
    else:
        stat_res, pval_res = ttest_ind(con1_sample, con2_sample, axis=0)
        result_dict['diff_dims'] = pval_res
        #stat_res = pd.DataFrame(t_stat)
        #pval_res = pd.DataFrame(pval)
        ## Change the p-values to -log10(pvalues)
        contains_invalid_ttest =  np.isnan(pval_res).any() or np.isinf(pval_res).any()
        if contains_invalid_ttest:
            print('Error in running ttest')
            result_df = pd.DataFrame(result_dict)
        else:
            trans_pval = [-np.log10(x) for x in pval_res]
            combined_matrix = np.vstack((latent_res.transpose(), prior_res.transpose()))
            # Calculate the correlation coefficient matrix
            correlation_matrix = np.corrcoef(combined_matrix)
            # Extract the correlation coefficients for the inter-matrix comparisons
            num_rows_matrix1 = prior_res.shape[1]
            # The correlation matrix consists of four main sections, the one between input and reconstruction belongs to [:num_rows_matrix1, num_rows_matrix1:]
            # The input matrix should remain the same with the same input data, but not the case with the output one and the inter one
            correlation_inter_matrix = correlation_matrix[:num_rows_matrix1, num_rows_matrix1:]
            correlation_paired_matrix = np.diag(correlation_inter_matrix)
            result_dict['correlatin_values'] = correlation_paired_matrix
            result_df = pd.DataFrame(result_dict)
            unsorted_list = [(pval, index_nr, path_anno) for pval, index_nr, path_anno in zip(trans_pval, range(len(path_anno)), path_anno)]
            sorted_list = sorted(unsorted_list, reverse=True)
            pvalue_sorted = []
            path_sorted = []
            ind_sorted = []
            # Only top 50
            if len(sorted_list) > threshold:
                sorted_list = sorted_list[0:threshold]
                
            for j in sorted_list:
                pvalue_sorted += [j[0]]
                ind_sorted += [j[1]]
                path_sorted += [j[2]]
            
            sig_level = correlation_paired_matrix[ind_sorted]

            colors = []
            ## The color is red when the correlations are negative
            for value in sig_level:  
                if value < 0:
                    colors.append('#B50835')
                else:
                    colors.append('#1f77b4')

            plt.scatter(path_sorted, pvalue_sorted, s = abs(sig_level)*100, c = colors)
            plt.xticks(rotation=270)
            plt.gcf().set_size_inches(6.5, 6.9)
            plt.axhline(y=-np.log10(0.05), color='r', linestyle='--')
            plt.xlabel('Latent Dimensions') 
            plt.ylabel('-log10 Transformed P-values')
            plt.tight_layout()
            plt.savefig(save_path +  con1_label + '_vs_' + con2_label + "_correlation_vs_desig_dimensions.eps")
            plt.show()
            plt.close()

            print(np.sum(pval_res < 0.05))
    return result_df


def deg_latent_dimensions_with_given_indices_heatmap_conditions(latent_res, con1_index, con2_index, con1_label, con2_label, path_anno, save_path):
    colors = ["blue", "red"]
    color_bar = {con1_label:colors[0], con2_label:colors[1]}
    con1_sample = latent_res[con1_index,:]
    con2_sample = latent_res[con2_index,:]
    condition_all = [*list(repeat(con1_label, len(con1_index))), *list(
    repeat(con2_label, len(con2_index)))]
    sample_interest_latent = np.vstack((con1_sample, con2_sample))
    contains_invalid_latent = np.isnan(sample_interest_latent).any() or np.isinf(sample_interest_latent).any()
    if contains_invalid_latent:
            print('Error in running ttest')
    else:
        sns.clustermap(np.transpose(sample_interest_latent), standard_scale=0, col_colors=pd.Series(condition_all).map(color_bar).to_numpy(), yticklabels=path_anno, xticklabels=False, row_cluster=False, col_cluster=False,
                    cmap="coolwarm")
        if len(path_anno) <= 50:
            plt.gcf().set_size_inches(8, 8.4)
        else:
            plt.gcf().set_size_inches(10, 20)
        plt.tight_layout()
        plt.savefig(save_path +  con1_label + '_vs_' + con2_label + "_latent_dimension_heatmaps_not_clustered.eps")
        plt.close()
        
        
        
        try:
            sns.clustermap(np.transpose(sample_interest_latent), standard_scale=0, col_colors=pd.Series(condition_all).map(color_bar).to_numpy(), yticklabels=path_anno, xticklabels=False, row_cluster=True, col_cluster=True,
                        cmap="coolwarm")
            if len(path_anno) <= 50:
                plt.gcf().set_size_inches(8, 8.4)
            else:
                plt.gcf().set_size_inches(10, 20)
            plt.tight_layout()
            plt.savefig(save_path +  con1_label + '_vs_' + con2_label + "_latent_dimension_heatmaps_clustered.eps")
            plt.close()
            
            
        except Exception as e:
            print("Clustering failed:", e)
            sns.clustermap(np.transpose(sample_interest_latent), standard_scale=0, col_colors=pd.Series(condition_all).map(color_bar).to_numpy(), yticklabels=path_anno, xticklabels=False, row_cluster=False, col_cluster=False,
            cmap="coolwarm")
            if len(path_anno) <= 50:
                plt.gcf().set_size_inches(8, 8.4)
            else:
                plt.gcf().set_size_inches(10, 20)
            
            plt.tight_layout()
            plt.savefig(save_path +  con1_label + '_vs_' + con2_label + "_latent_dimension_heatmaps_clustering_failed_not_clustered.eps")
            plt.close()
        
            
            
    return 'Done'