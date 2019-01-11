# Vegard B 2019


# A gmt (Gene Matrix Transposed) is a file used by gsea to evaluate sequences and score

import pandas as pd
import data_reader

# Import data
df, tar, grp = data_reader.read_number(0)

df['target'] = tar
scores = {}
df_tumor = df.loc[df['target'] == 'Tumor']
df_normal = df.loc[df['target'] == 'Normal']

df = df.drop('target', axis=1)

for ax in df.axes[1]:
    mean_tumor = df_tumor[ax].mean()
    mean_normal = df_normal[ax].mean()
    std_tumor = df_tumor[ax].std()
    std_normal = df_normal[ax].std()
    scores[ax] = (mean_tumor - mean_normal) / (std_tumor+std_normal)

for score in range(50):
    best = max(scores)
    print(best, scores[best])
    del scores[best]
