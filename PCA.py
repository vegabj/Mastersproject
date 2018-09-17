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
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2']
             , index = df_index)

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
#pd.concat([principalDf, df.axes[0]], axis = 1)
#pd.concat([principalDf, df[['target']]], axis = 1)

from matplotlib import pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA Hepmark-Microarray', fontsize = 20)
targets = ['Normal', 'Tumor', 'Undefined'] #['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_

plt.show()
