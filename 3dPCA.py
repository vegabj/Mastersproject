import pandas as pd
import numpy as np

import Data_Reader
df, target = Data_Reader.Read_Hepmark_Microarray()
#df, target = Data_Reader.Read_Hepmark_Tissue()
#df, target = Data_Reader.Read_Hepmark_Hepmark_Paired_Tissue()
features = df.axes[1].values
# Add target to df
df['target'] = target
x = df.loc[:,features].values
y = df.loc[:,'target'].values #df.axes[0].values
idx = []
last = df.axes[0][0]
for i,index in enumerate(df.axes[0]):
    if index[:3] != last[:3]:
        idx.append(i)
        last = index
idx.append(len(x))


from sklearn.preprocessing import StandardScaler
# Standardizing the features
xs = []
last = 0
for i in idx:
    xs.append(StandardScaler().fit_transform(x[last:i]))
    last = i

x = np.concatenate((xs), axis=0)


df_index = df.axes[0]

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3']
             , index = df_index)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_zlabel('PC3', fontsize = 15)

targets = ['Normal', 'Tumor']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter3D(finalDf.loc[indicesToKeep, 'principal component 1']
        , finalDf.loc[indicesToKeep, 'principal component 2']
        ,finalDf.loc[indicesToKeep, 'principal component 3']
        , c=color, s=20)

plt.show()
