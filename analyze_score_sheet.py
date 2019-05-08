"""
Vegard Bjørgan 2019

analyze score sheet generates a heatmap per normalization
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

# Gather scores for each tile in the heatmaps
score_dict = {"none": [], "standard": [], "other": []}
for P in test_sizes:
    none_n = []
    standard_n = []
    other_n = []
    for N in test_sizes:
        select = df.loc[(df["P"] == P) & (df["N"] == N)]
        if selected_dataset:
            select = select.loc[select["Dataset"] == datasets[selected_dataset-1]]
        select_none = select.loc[df["Normalization"] == 'None']
        select_standard = select.loc[df["Normalization"] == 'MinMax']
        select_other = select.loc[df["Normalization"] == 'Closest']
        if P in test_sizes[2:] and N in test_sizes[2:]:
            overall = select.loc[:, "ROC(auc)"].mean()
            none = select_none.loc[:, "ROC(auc)"].mean()
            standard = select_standard.loc[:, "ROC(auc)"].mean()
            other = select_other.loc[:, "ROC(auc)"].mean()
        else:
            overall = select.loc[:, "Balanced accuracy"].mean()
            none = select_none.loc[:, "Balanced accuracy"].mean()
            standard = select_standard.loc[:, "Balanced accuracy"].mean()
            other = select_other.loc[:, "Balanced accuracy"].mean()
        print("P", P, "N", N)
        print("Overall:", overall, "None", none, "Standard", standard, "Other", other)
        none_n.append(none)
        standard_n.append(standard)
        other_n.append(other)
    score_dict["none"].append(none_n)
    score_dict["standard"].append(standard_n)
    score_dict["other"].append(other_n)

test_sizes_p = [x+"P" for x in test_sizes]
test_sizes_n = [x+"N" for x in test_sizes]
ds = [score_dict["none"], score_dict["standard"], score_dict["other"]]
ext = ["colon_none_rf_es.pdf", "colon_standard_rf_es.pdf", "colon_closest_rf_es.pdf"]

# Uncomment to Latexify the heatmap
#latexify(columns=2)

# Generate Heatmaps
for i,d in enumerate(ds):
    scores = np.array(d)
    fig, ax = plt.subplots()
    im, cbar = heatmap.heatmap(scores, test_sizes_p, test_sizes_n, ax=ax,
                       vmin = 0.0, vmax = 1.0, cmap=cm.Reds, cbarlabel="score [AUC / Acc]")
    texts = heatmap.annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()
    fig.savefig("C:/Users/Vegard/Desktop/Master/Mastersproject/Plots/analyze/"+ext[i])
