import pandas as pd
from os import getcwd, listdir
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import heatmap

path = r'%s' % getcwd().replace('\\','/') + "/Out/scores/"
scores = listdir(path)
# Simples ui choice for selecting score dataset
print("Scores to analyze:")
for i,score in enumerate(scores):
    print(i,score)
select = 0 #int(input("Select: "))
print('\n\n')
path = path + scores[select]
df = pd.read_csv(path, index_col = 0)
datasets = df.Dataset.unique()
for i, dataset in enumerate(['All']+list(datasets)):
    print(i, dataset)
selected_dataset = 0 #int(input("Select: "))

# TODO: UI choices
# Choose 1 dataset
# Choose one scaling
# Choose 1 Dataset
# Choose one scaling

test_sizes = ['0', '1', '2', '4', '8', '16', 'all']

score_dict = {"none": [], "standard": [], "other": []}
for P in test_sizes:
    none_n = []
    standard_n = []
    other_n = []
    #others_n = []
    for N in test_sizes:
        select = df.loc[(df["P"] == P) & (df["N"] == N)]
        if selected_dataset:
            select = select.loc[select["Dataset"] == datasets[selected_dataset-1]]
        select_none = select.loc[df["Normalization"] == 'None']
        select_standard = select.loc[df["Normalization"] == 'Standard']
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

# latexify
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


fig = plt.figure()
ax1 = plt.subplot2grid((1,2), (0,0))
scores_1 = np.array(ds[0])
im, cbar = heatmap.heatmap(scores_1, test_sizes_p, test_sizes_n, ax=ax1,
   vmin = 0.0, vmax = 1.0, cmap=cm.Reds)
cbar.remove()
texts = heatmap.annotate_heatmap(im, valfmt="{x:.2f}")
ax2 = plt.subplot2grid((1,2), (0,1))
scores_2 = np.array(ds[1])
im, cbar = heatmap.heatmap(scores_2, test_sizes_p, test_sizes_n, ax=ax2,
   vmin = 0.0, vmax = 1.0, cmap=cm.Reds, cbarlabel="score [AUC / Acc]")
cbar.remove()
texts = heatmap.annotate_heatmap(im, valfmt="{x:.2f}")

cbar = fig.colorbar(im, ax=[ax1,ax2], orientation='horizontal', fraction=0.1, pad=-.8, panchor=False)
cbar.ax.set_xlabel("score [AUC / Acc]", rotation=0, va='top')

fig.tight_layout(pad=3.0, w_pad=4.5, h_pad=0.1)
plt.show()
