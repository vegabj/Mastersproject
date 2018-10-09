'''
Vegard Bjørgan 2018

pca.py creates a 2-n dimentional principal component analysis of the data
'''

import pandas as pd
import numpy as np
import df_utils

# TODO: UI
# num principal components
#

# Read df
import data_reader
names = data_reader.get_sets()
print("Available data sets are:")
for i,e in enumerate(names):
    print(str(i)+":", e)
selected = input("Select data set (multiselect separate with ' '): ")
selected = selected.split(' ')

multi_select = False if len(selected) == 1 else True
if multi_select:
    dfs = []
    targets = []
    groups = []
    for select in selected:
        df, tar, grp = data_reader.read_number(int(select))
        dfs.append(df)
        targets.append(tar)
        groups.append(grp)

    df = df_utils.merge_frames(dfs)
    target = targets[0]
    group = groups[0]
    for tar, gro in zip(targets[1:], groups[1:]):
        target = np.append(target, tar)
        group = np.append(group, gro)
    lengths = [d.values.shape[0] for d in dfs]
else:
    df, target, group = data_reader.read_number(int(selected[0]))
    lengths = []


# Separate features and targets / meta-data
features = df.axes[1].values
df['target'] = target
df['group'] = group

x = df.loc[:,features].values
y = df.loc[:,'target'].values

# Apply normalization
from scaler import MiRNAScaler
#x = MiRNAScaler.standard_scaler(x)
#x = MiRNAScaler.group_scaler(df, features)
x = MiRNAScaler.miRNA_scaler(x)
#x = MiRNAScaler.set_scaler(df, lengths, features)

df_index = df.axes[0]

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'
             ,'principal component 3', 'principal component 4']
             , index = df_index)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

#print(finalDf)

print("PCA variance ratio:", pca.explained_variance_ratio_)

# Plot the principal components
import interactive_scatterplot as scatter
scatter.pca_scatter(finalDf, multi_select, lengths)
