'''
Vegard Bjørgan 2018

pca.py creates a 2-n dimentional principal component analysis of the data
'''

import pandas as pd
import numpy as np

# Read df
import data_reader

df, target, groups = data_reader.read_hepmark_microarray()
#df, target, groups = data_reader.read_hepmark_paired_tissue()
df, target, groups = data_reader.read_hepmark_paired_tissue_formatted()
#df, target, groups = data_reader.read_publicCRC_PMID_26436952()
'''

df, target, groups = data_reader.read_hepmark_tissue()
df, target, groups = data_reader.read_hepmark_tissue_formatted()
df, target, groups = data_reader.read_guihuaSun_PMID_26646696_colon()
df, target, groups = data_reader.read_guihuaSun_PMID_26646696_rectal()
df, target, groups = data_reader.read_publicCRC_GSE46622_colon()
df, target, groups = data_reader.read_publicCRC_GSE46622_rectum()
df, target, groups = data_reader.read_guihuaSun_PMID_26646696()
df, target, groups = data_reader.read_publicCRC_GSE46622()
df, target, groups = data_reader.read_publicCRC_PMID_26436952()
'''


# Separate features and targets / meta-data
features = df.axes[1].values
df['target'] = target
df = df.dropna(axis=0)
df['group'] = groups

x = df.loc[:,features].values
y = df.loc[:,'target'].values


# Apply normalization to each group
from sklearn.preprocessing import StandardScaler
xs = []
grouped = df.groupby('group')

for groupname, group in grouped:
    xs.append(
        StandardScaler().fit_transform(
            group.loc[:, features]))

# Combine normalized groups or use one scaler for all data
x = np.concatenate((xs), axis=0) # StandardScaler().fit_transform(x)

df_index = df.axes[0]

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'
             ,'principal component 3', 'principal component 4']
             , index = df_index)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf)

from matplotlib import pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = set(y)
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

ax.legend(targets)
ax.grid()
# Label a target
'''
indicesToKeep = finalDf.loc['509-1-4']
print(indicesToKeep)
ax.scatter(indicesToKeep['principal component 1']
    , indicesToKeep['principal component 2']
    , c = 'b'
    , marker = "x"
    , s = 100)
'''


pca.explained_variance_ratio_


plt.show()

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 3', fontsize = 15)
ax.set_ylabel('Principal Component 4', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = set(y)
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 3']
               , finalDf.loc[indicesToKeep, 'principal component 4']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()
