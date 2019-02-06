'''
Vegard Bjørgan 2018

pca.py creates a 2-n dimentional principal component analysis of the data
'''

import pandas as pd
import numpy as np
import df_utils
import scaler as MiRNAScaler
import data_reader
from sklearn.decomposition import PCA
import interactive_scatterplot as scatter

# TODO: Add num principal components(?)

# Read df
df, target, group, lengths = data_reader.read_main(raw=False, es=True)
#df = df[['Normal_3', 'Normal_5', 'Normal_7', 'Tumor_3', 'Tumor_5', 'Tumor_7']]
#df = df[['Normal', 'Tumor']]
multi_select = len(lengths) > 1

# Separate features and targets / meta-data
features = df.axes[1].values

# Restrict to known mirna related to colon cancer
#features = ['hsa-miR-21-5p', 'hsa-miR-21-3p', 'hsa-miR-143-5p', 'hsa-miR-143-3p'] for CRC

df['target'] = target
df['group'] = group

x = df.loc[:,features].values
y = df.loc[:,'target'].values

# Apply normalization
if multi_select:
    x = MiRNAScaler.standard_scaler(x)
    #x = MiRNAScaler.set_scaler(df.loc[:,features], lengths)
    #x = MiRNAScaler.individual_scaler(x)
else:
    x = MiRNAScaler.standard_scaler(x)
    #x = MiRNAScaler.miRNA_scaler(x)
    #print(x)

df_index = df.axes[0]

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2']
             #,'principal component 3', 'principal component 4']
             , index = df_index)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print("PCA variance ratio:", pca.explained_variance_ratio_)

# Plot the principal components
#scatter.pca_scatter_latex(finalDf, multi_select, lengths)
scatter.pca_scatter(finalDf, multi_select, lengths)
