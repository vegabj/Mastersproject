"""
Vegard Bjørgan 2019

pca.py creates a 2-n dimentional principal component analysis of the data

Code adapted from:
https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
"""

import pandas as pd
import numpy as np
import df_utils
import scaler as MiRNAScaler
import data_reader
from sklearn.decomposition import PCA
import interactive_scatterplot as scatter

# Read data
df, target, group, lengths, es = data_reader.read_main(raw=False)
multi_select = len(lengths) > 1

# Separate features and targets / meta-data
features = df.axes[1].values

df['target'] = target
df['group'] = group

x = df.loc[:,features].values
y = df.loc[:,'target'].values

# Apply normalization
x_2 = x
x = MiRNAScaler.choose_scaling(df.loc[:,features], lengths)


df_index = df.axes[0]

# Make PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2']
             , index = df_index)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print("PCA variance ratio:", pca.explained_variance_ratio_)
print(len(finalDf))

# Make a second PCA (for pca_scatteR_latex)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_2)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2']
             , index = df_index)
finalDf_2 = pd.concat([principalDf, df[['target']]], axis = 1)

# Plot the principal components
scatter.pca_scatter(finalDf, multi_select, lengths)

# A more latex friendly scatter plot
scatter.pca_scatter_latex(finalDf_2, finalDf, multi_select, lengths)
