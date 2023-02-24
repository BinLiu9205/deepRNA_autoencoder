import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, silhouette_score
seed = 42
np.random.seed(seed)


# f_dir1 = os.listdir(
#     "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/latent_results")
# f_dir2 = os.listdir(
#     "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/latent_results")
# f_dir3 = os.listdir(
#     "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/latent_results")


# dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs/latent_results/"
# dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs/latent_results/"
# dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs/latent_results/"


f_dir1 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")
f_dir2 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")
f_dir3 = os.listdir(
    "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results")


dir1 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"
dir2 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"
dir3 = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/new_latent_results/"


dir1_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_community_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir2_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_gene_no_earlystopping_500_epochs_sigma_mu_while_training/"
dir3_root = "/mnt/dzl_bioinf/binliu/deepRNA/data_and_results_all_samples_transcript_no_earlystopping_500_epochs_sigma_mu_while_training/"

add_f1 = "community level + "
add_f2 = "gene level + "
add_f3 = "transcript level + "

all_res = [dir1 + i for i in f_dir1]
all_res = (*all_res, *[dir2 + i for i in f_dir2])
all_res = (*all_res, *
           [dir3 + i for i in f_dir3])

name_list = [add_f1 + i[0:-4] for i in f_dir1]
name_list = (*name_list, *[add_f2 + i[0:-4] for i in f_dir2])
name_list = (*name_list, *
             [add_f3 + i[0:-4] for i in f_dir3])

anno1 = 'gene'
anno2 = 'gene'
anno3 = 'transcript'

model_list = [i[0:-4] for i in f_dir1]
model_list = (*model_list, *[i[0:-4] for i in f_dir2])
model_list = (*model_list, *
              [i[0:-4] for i in f_dir3])

annoName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno1 + '_level.csv'
annoName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno2 + '_level.csv'
annoName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/prior_distribution/bootstrap_mean_MsigDB_' + anno3 + '_level.csv'
annoList = [annoName1 for i in range(len(f_dir1))]
annoList = (*annoList, *[annoName2 for i in range(len(f_dir2))])
annoList = (*annoList, *[annoName3 for i in range(len(f_dir3))])

test1 = 'community'
test2 = 'gene'
test3 = 'transcript'

trainName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test1 + '_level_train_test/all_samples_' + test1 + '_level_train.pkl'
trainName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test2 + '_level_train_test/all_samples_' + test2 + '_level_train.pkl'
trainName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test3 + '_level_train_test/all_samples_' + test3 + '_level_train.pkl'
trainList = [trainName1 for i in range(len(f_dir1))]
trainList = (*trainList, *[trainName2 for i in range(len(f_dir2))])
trainList = (*trainList, *[trainName3 for i in range(len(f_dir3))])

testName1 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test1 + '_level_train_test/all_samples_' + test1 + '_level_test.pkl'
testName2 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test2 + '_level_train_test/all_samples_' + test2 + '_level_test.pkl'
testName3 = '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/' + \
    test3 + '_level_train_test/all_samples_' + test3 + '_level_test.pkl'
testList = [testName1 for i in range(len(f_dir1))]
testList = (*testList, *[testName2 for i in range(len(f_dir2))])
testList = (*testList, *[testName3 for i in range(len(f_dir3))])

# Change this part for saving the plots and numeric results
main_dir = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/deepRNA_Py/post_hoc_all_samples/no_early_stopping_500_epochs_new_sigma_mu/'
plot_dir = 'exploration_latent_plots_tsne_pca_init_mu_instead_of_z_for_vaes'
pdf_dir = 'ms_exploration_latent_plots_tsne_pdf'

path = os.path.join(main_dir, plot_dir)
os.makedirs(path, exist_ok=True)
path_new = os.path.join(main_dir, pdf_dir)
os.makedirs(path_new, exist_ok=True)


annotate_tem = pd.read_excel(
    "/mnt/dzl_bioinf/binliu/EMBL_ExpressionArray/arrayExpress_annotation.xlsx")
X_test = pd.read_pickle(
    '/mnt/dzl_bioinf/binliu/deepRNA/data_all_samples/community_level_train_test/all_samples_community_level_test.pkl')
annotate_tem = annotate_tem.set_index('Source Name')
annotate_test = annotate_tem.reindex(index=X_test.index)
annotate_test['normalStatus'] = (
    annotate_test['Characteristics[disease]'] == 'normal')
colors = {True: 'blue', False: 'red'}


organism = pd.unique(annotate_test['Characteristics[organism part]'])
organism_dis = annotate_test['Characteristics[organism part]'].value_counts()
organism_counts = pd.DataFrame(organism_dis)
organism_counts = organism_counts.reset_index()
organism_counts.columns = ['unique_organs',
                           'counts for sample']  # change column names
interested_organs = organism_counts.iloc[1:9, 0]


idxs = []
organ_list = []
for i in interested_organs:
    tem_idx = annotate_test['Characteristics[organism part]'] == i
    tem_anno = annotate_test.index[tem_idx]
    idxs = [*idxs, *tem_anno]
    organ_list_tem = [i] * len(tem_anno)
    #print(organ_list_tem)
    organ_list = [*organ_list, *organ_list_tem]

idxs_loc = [annotate_test.index.get_loc(idx) for idx in idxs]
color_pa = ['#ff6961', '#ffb480', '#f8f38d',
            '#42d6a4', '#08cad1', '#59adf6', '#9d94ff', '#c780e8']

counter = 0
current_train = 'inti'
for latent in all_res:
    res = np.load(latent)
    silhouette_avg_kmeans = []
    silhouette_avg_gm = []
    for k in range(2, 15):
        #Initialize the class object
        kmeans = KMeans(n_clusters=k)
        gmModel = GaussianMixture(n_components=k)
    #predict the labels of clusters.
        label1 = kmeans.fit(res).predict(res)
        label2 = gmModel.fit(res).predict(res)
        silhouette_avg_kmeans.append(silhouette_score(res, label1))
        silhouette_avg_gm.append(silhouette_score(res, label2))

    plt.plot(range(2, 15), silhouette_avg_kmeans)
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score kmeans')
    plt.title('Silhouette analysis For Optimal k in kmeans for ' +
              name_list[counter])
    plt.savefig(main_dir + plot_dir + '/' +
                name_list[counter] + '_Silhouette_score_kmeans.png')
    plt.show()
    plt.clf()

    plt.plot(range(2, 15), silhouette_avg_gm)
    plt.xlabel('Values of K')
    plt.ylabel('Silhouette score gmm')
    plt.title('Silhouette analysis For Optimal k in gmm for ' +
              name_list[counter])
    plt.savefig(main_dir + plot_dir + '/' +
                name_list[counter] + '_Silhouette_score_kmm.png')
    plt.show()
    plt.clf()

    print(name_list[counter])
    silhouette_kmeans = max(silhouette_avg_kmeans)
    silhouette_gmm = max(silhouette_avg_gm)
    if silhouette_gmm >= silhouette_kmeans:
        print("Use the Gaussian Mixture Model for " + name_list[counter])
        best_n = silhouette_avg_gm.index(silhouette_gmm) + 2
        print("Number of clusters: " + str(best_n))
    else:
        print("Use the Kmeans for " + latent[0:-4])
        best_n = silhouette_avg_kmeans.index(silhouette_kmeans) + 2
        print("Number of clusters: " + str(best_n))

    latent_tsne = TSNE(n_components=2, init='pca').fit_transform(res)
    # Normal scatter plot
    plt.scatter(latent_tsne[:, 0],
                latent_tsne[:, 1])
    plt.title(
        'Tsne dimensionality reduction for ' + latent[0:-4])
    plt.savefig(main_dir + plot_dir + '/' +
                name_list[counter] + '_tsne_pca_init.png')
    plt.show()
    plt.clf()
    #Scatter plot by health status
    plt.scatter(latent_tsne[:, 0],
                latent_tsne[:, 1], c=annotate_test['normalStatus'].apply(lambda x: colors[x]))
    plt.title(
        'Tsne dimensionality reduction for ' + model_list[counter] + ' (vs health status)')
    plt.savefig(main_dir + plot_dir + '/' +
                name_list[counter] + '_(health_status)_pca_init.png')
    plt.show()
    plt.clf()
    #Scatter plot by organsism

    for i, color in zip(interested_organs, color_pa):
        plt.scatter(latent_tsne[annotate_test['Characteristics[organism part]'] == i, 0],
                    latent_tsne[annotate_test['Characteristics[organism part]'] == i, 1], c=color)

    plt.title(
        'Tsne dimensionality reduction for ' + name_list[counter] + ' (vs organism)')
    plt.savefig(main_dir + plot_dir + '/' +
                name_list[counter] + '_(orgianism)_pca_init.png')
    plt.show()
    plt.clf()
    
    for i, color in zip(interested_organs, color_pa):
        plt.scatter(latent_tsne[annotate_test['Characteristics[organism part]'] == i, 0],
                    latent_tsne[annotate_test['Characteristics[organism part]'] == i, 1], c=color, alpha=0.5, s=plt.rcParams['lines.markersize'] ** 1.5)
    plt.gcf().set_size_inches(3.2, 2.9)
    plt.tight_layout()
    plt.savefig(main_dir + pdf_dir + '/' +
                name_list[counter] + '_(orgianism)_pca_init.svg')
    plt.show()
    plt.clf()

    counter += 1
