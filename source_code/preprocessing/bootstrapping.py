import numpy as np
import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy import stats
import sys

seed = 42
np.random.seed(seed)
class BootstrappingMSigDB:
    # target_data is the zscore caculated in all the samples in the larger dataset
    def __init__(self, sigData, targetData, 
    platAnnotation="/mnt/dzl_bioinf/binliu/jupyter/important_supportive_files/GPL570-55999.txt"):
        self.sigData = sigData
        self.targetData = targetData
        self.platAnnotation = platAnnotation

    # header can equal to None, to read no header
    def read_signature(self, header = 0):
        print("Reading the signature...",  file=sys.stderr)
        tem = pd.read_fwf(self.sigData, header = header)
        if header is not None:
            sigName = tem.columns[0]
        else:
            sigName = self.sigData
        geneList = tem[tem.columns[0]]
        return sigName, geneList

    def zScorePath(self, sigName, geneList):
        print("Running zScorePath...",  file=sys.stderr)
        platform_df = pd.read_table(self.platAnnotation, skiprows = 16, sep= "\t", low_memory= False)
        #  print(self.targetData.head(5))
        interest_locs = []
        counter = 0 
        for gene in geneList:
            expressionPattern = "^" + gene + "$|" + gene + " \/\/\/|\/\/\/ " + gene
            interest_loc = platform_df['ID'][platform_df['Gene Symbol'].str.contains(
                expressionPattern, regex=True, case=False, na=False)]
            if isinstance(interest_loc, pd.Series):
                if len(interest_loc) > 0:
                    counter += 1
                for i in interest_loc:
                    interest_locs.append(i)
            else:
                interest_locs.append(interest_loc)
                counter += 1 
        interest_locs = list(set(interest_locs))
        #print(str(interest_locs))
        print("Signature name: " + sigName + ", \nlength of the list: " +
              str(len(geneList)) + ", \n available in the set: " + str(counter) + 
              "\n" + str(len(geneList) - counter) + " genes are missing")
        return interest_locs
    
    def zScorePathGeneVersion(self, sigName, geneList):
        print("Running zScorePath...",  file=sys.stderr)
        platform_df = pd.read_table(self.platAnnotation, skiprows = 16, sep= "\t", low_memory= False)
        #  print(self.targetData.head(5))
        columnNames = self.targetData.columns.values.tolist()
        interest_locs = []
        counter = 0 
        for gene in geneList:
            #expressionPattern = "^" + gene + "$|" + gene + " \/\/\/|\/\/\/ " + gene
            #interest_loc = platform_df['ID'][platform_df['Gene Symbol'].str.contains(
            #    expressionPattern, regex=True, case=False, na=False)]
            interest_loc = [i for i,x in enumerate(columnNames) if x == gene]
            if len(interest_loc) > 0:
                counter += 1
                for i in interest_loc:
                    interest_locs.append(i)
            else:
                continue
        interest_locs = list(set(interest_locs))
        #print(str(interest_locs))
        print("Signature name: " + sigName + ", \nlength of the list: " +
              str(len(geneList)) + ", \n available in the set: " + str(counter) + 
              "\n" + str(len(geneList) - counter) + " genes are missing")
        return interest_locs
        

    @staticmethod           
    def bootstrapSingle(df, n_interations=50):
        mean_direct = np.mean(df)
        var_direct = np.var(df)
        mean_sample = list()
        var_sample = list()
        n_size = len(df)
        #print(n_size)
        for i in range(n_interations):
            data_sub = resample(df, n_samples=n_size)#, random_state=42)
            mean_sample.append(np.mean(data_sub))
            var_sample.append(np.var(data_sub))

        mean_bootstrap = np.mean(mean_sample)
        sigma_bootstrap = np.mean(var_sample)
        _, p_values_mean = stats.normaltest(mean_sample)
        #print(str(mean_sample))
        return mean_direct, var_direct, mean_bootstrap, sigma_bootstrap, p_values_mean


    def bootstrappingProcess(self, geneList, interest_locs):
        print("Running bootstrapping..", file=sys.stderr)
        subTable = self.targetData[interest_locs]
        resTotal = subTable.apply(self.bootstrapSingle, axis = 1)
        resTotal = pd.DataFrame(resTotal.tolist(),index=resTotal.index)
        #print(resTotal)
        return resTotal

    def bootstrappingProcess_pd(self, geneList, interest_locs):
        print("Running bootstrapping..", file=sys.stderr)
        subTable = self.targetData.iloc[:, interest_locs]
        resTotal = subTable.apply(self.bootstrapSingle, axis = 1)
        resTotal = pd.DataFrame(resTotal.tolist(),index=resTotal.index)
        #print(resTotal)
        return resTotal