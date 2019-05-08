"""
Vegard Bjørgan 2019

dual_heatmap.py generates two heatmaps chosen through the user interface
and plots them into a single figure that is latex friendly.
"""

import pandas as pd
from os import getcwd, listdir
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import heatmap

path = r'%s' % getcwd().replace('\\','/') + "/Out/scores/"
scores = listdir(path)

test_sizes = ['0', '1', '2', '4', '8', '16', 'all']
score_heatmap = {"overall_0" : [], "overall_1" : []}

for t in range(2):
    # Simples user interface for selecting score sheets, data sets and normalization
    print("Scores to analyze:")
    for i,score in enumerate(scores):
        print(i,score)
    select = int(input("Select: "))
    print('\n\n')
    df_path = path + scores[select]
    df = pd.read_csv(df_path, index_col = 0)
    datasets = df.Dataset.unique()
    for i, dataset in enumerate(['All']+list(datasets)):
        print(i, dataset)
    selected_dataset = int(input("Select: "))
    print('\n\n')
    normalization = df.Normalization.unique()
    for i, norm in enumerate(['All']+list(normalization)):
        print(i, norm)
    selected_normalization = int(input("Select: "))

    # Extract data for each tile in the heatmap
    for P in test_sizes:
        overall_n = []
        for N in test_sizes:
            select = df.loc[(df["P"] == P) & (df["N"] == N)]
            if selected_dataset:
                select = select.loc[select["Dataset"] == datasets[selected_dataset-1]]
            if selected_normalization:
                select = select.loc[df["Normalization"] == normalization[selected_normalization-1]]
            if P in test_sizes[2:] and N in test_sizes[2:]:
                overall = select.loc[:, "ROC(auc)"].mean()
            else:
                overall = select.loc[:, "Balanced accuracy"].mean()
            overall_n.append(overall)
        score_heatmap["overall_"+str(t)].append(overall_n)

    test_sizes_p = [x+"P" for x in test_sizes]
    test_sizes_n = [x+"N" for x in test_sizes]

ds = [score_heatmap["overall_0"], score_heatmap["overall_1"]]


# latexify the plot
fig_width = 6.9
fig_height = fig_width / 2.1
params = {'backend': 'ps',
          'text.latex.preamble': ['\\usepackage{gensymb}'],
          'axes.labelsize': 8, # fontsize for x and y labels (was 10)
          'axes.titlesize': 8,
          'font.size': 8, # was 10
          'legend.fontsize': 8, # was 10
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': [fig_width,fig_height*1.2],
          'font.family': 'serif'
}
matplotlib.rcParams.update(params)


# create the figure
fig = plt.figure()

# Setup heatmap 1
ax1 = plt.subplot2grid((1,2), (0,0))
scores_1 = np.array(ds[0])
im, cbar = heatmap.heatmap(scores_1, test_sizes_p, test_sizes_n, ax=ax1,
   vmin = 0.0, vmax = 1.0, cmap=cm.Reds)
cbar.remove()
texts = heatmap.annotate_heatmap(im, valfmt="{x:.2f}")

# Setup heatmap 2
ax2 = plt.subplot2grid((1,2), (0,1))
scores_2 = np.array(ds[1])
im, cbar = heatmap.heatmap(scores_2, test_sizes_p, test_sizes_n, ax=ax2,
   vmin = 0.0, vmax = 1.0, cmap=cm.Reds, cbarlabel="score [AUC / Acc]")
cbar.remove()
texts = heatmap.annotate_heatmap(im, valfmt="{x:.2f}")

# Setup for colorbar
cbar = fig.colorbar(im, ax=[ax1,ax2], orientation='horizontal', fraction=0.10, pad=-.8, anchor=(0.0,1.0), panchor=False, shrink=1.15)
cbar.ax.set_xlabel("score [AUC / Acc]", rotation=0, va='top')

fig.tight_layout(pad=3.0, w_pad=4.5, h_pad=0.1)
plt.show()
