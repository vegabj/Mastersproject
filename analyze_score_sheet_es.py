"""
Vegard Bjørgan 2019

analyze score sheet generates a heatmap for enrichment score sheets
"""

import pandas as pd
from os import getcwd, listdir
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import heatmap
from utils import latexify

path = r'%s' % getcwd().replace('\\','/') + "/Out/scores/"
scores = listdir(path)

# Simples user interface for selecting score sheet
print("Scores to analyze:")
for i,score in enumerate(scores):
    print(i,score)
select = int(input("Select: "))
print('\n\n')
path = path + scores[select]
df = pd.read_csv(path, index_col = 0)
datasets = df.Dataset.unique()
for i, dataset in enumerate(['All']+list(datasets)):
    print(i, dataset)
selected_dataset = int(input("Select: "))

test_sizes = ['0', '1', '2', '4', '8', '16', 'all']

# Gather scores for each tile in the heatmap
score_dict = {"es_score": []}
for P in test_sizes:
    es_score_n = []
    #others_n = []
    for N in test_sizes:
        select = df.loc[(df["P"] == P) & (df["N"] == N)]
        if selected_dataset:
            select = select.loc[select["Dataset"] == datasets[selected_dataset-1]]
        if P in test_sizes[2:] and N in test_sizes[2:]:
            overall = select.loc[:, "ROC(auc)"].mean()
        else:
            overall = select.loc[:, "Balanced accuracy"].mean()
        es_score_n.append(overall)
    score_dict["es_score"].append(es_score_n)

test_sizes_p = [x+"P" for x in test_sizes]
test_sizes_n = [x+"N" for x in test_sizes]
ext = "hepmark_es_svm_"+str(selected_dataset-1)+".pdf" if selected_dataset else "hepmark_es_svm.pdf"

# Uncomment to Latexify the heatmap
#latexify(columns=2)

# Generate a Heatmap
scores = np.array(score_dict["es_score"])
fig, ax = plt.subplots()
im, cbar = heatmap.heatmap(scores, test_sizes_p, test_sizes_n, ax=ax,
                   vmin = 0.0, vmax = 1.0, cmap=cm.Reds, cbarlabel="score [AUC / Acc]")
texts = heatmap.annotate_heatmap(im, valfmt="{x:.2f}")

fig.tight_layout()
plt.show()
fig.savefig("C:/Users/Vegard/Desktop/Master/Mastersproject/Plots/analyze/"+ext)
