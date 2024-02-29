sweep_list = [
    "n5hd138p",
    "zurrd7db",
    "wstkfdy6",
    "75kmgmr7",
    "mmwhl929",
    "jhxcdj4q",
    "g559q1dp",
    "lbj8s20m",
    "wl2gxmnu",
    "0d7y0uh0",
    "edil514t",
    "uf100vsx",
    "m4kyravz"
]

kegg_list = [
    "0vq5iv0u",
    "on24bn8m",
    "3p2omtb7",
    "70mcbxe7",
    "h5u5kbut",
    "3ps08og1"
]


import pandas as pd
import numpy as np
import os

dir_path = '/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/'
entries = os.listdir(dir_path)
files = [entry for entry in entries if os.path.isfile(os.path.join(dir_path, entry))]

for file in files:
    tem_file = pd.read_csv(os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/', file), header=0, index_col=None)
    print(tem_file.head(5))
    sub_file = pd.DataFrame()
    for word in sweep_list:
        filtered_rows = tem_file[tem_file.iloc[:,0].str.contains(word, na=False)]
        sub_file = pd.concat([sub_file, filtered_rows], axis=0)

    sub_file.reset_index(drop=True, inplace=True)
    sub_file.to_csv(os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/sweep_criterion', 'sweep_value_' + file))

    sub_file = pd.DataFrame()
    for word in kegg_list:
        filtered_rows = tem_file[tem_file.iloc[:,0].str.contains(word, na=False)]
        sub_file = pd.concat([sub_file, filtered_rows], axis=0)

    sub_file.reset_index(drop=True, inplace=True)
    sub_file.to_csv(os.path.join('/mnt/dzl_bioinf/binliu/deepRNA/deepRNA/Experimental_update_revision/Updated_figures_posthocs/classifier_metrics/different_conditions_numeric/kegg_relevant', 'kegg_relevant_' + file))
