import numpy as np
import pandas as pd
from itertools import repeat

def slicing_numpy_design_dea_input(input_df, con1_index, con2_index, con1_label, con2_label, label_name):
    # input_df should be a dataframe 
    con1_sample = input_df[con1_index]
    con2_sample = input_df[con2_index]
    
    new_sample = np.vstack((con1_sample, con2_sample))
    new_sample_df = pd.DataFrame(new_sample, columns=input_df.columns)
    new_sample_df = new_sample_df.T
    condition_all_label = [*repeat(con1_label, con1_index.sum()), *repeat(con2_label, con2_index.sum())]
    condition_all_df = pd.DataFrame(condition_all_label, columns=[label_name])
    
    condition_all_df.index = new_sample_df.columns
    
    return new_sample_df, condition_all_df
