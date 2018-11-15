import pandas as pd
from os import getcwd, listdir
import numpy as np

path = r'%s' % getcwd().replace('\\','/') + "/Out/"
scores = listdir(path)
# Simples ui choice for selecting score dataset
print("Scores to analyze:")
for i,score in enumerate(scores):
    print(i,score)
select = int(input("Select: "))
path = path + scores[select]
df = pd.read_csv(path, index_col = 0)

test_sizes = ['0', '1', '2', '4', '8', '16', 'all']

score_dict = {"overall": [], "none": [], "standard": [], "other": []}
for P in test_sizes:
    overall_n = []
    none_n = []
    standard_n = []
    other_n = []
    for N in test_sizes:
        select = df.loc[(df["P"] == P) & (df["N"] == N)]
        # Hepmark_Microarray  Hepmark_Paired_Tissue
        #select = select.loc[select["Dataset"] == "Hepmark_Paired_Tissue"]
        select_none = select.loc[df["Normalization"] == 'None']
        select_standard = select.loc[df["Normalization"] == 'Standard']
        select_other = select.loc[df["Normalization"] == 'Other']
        if P in test_sizes[2:] and N in test_sizes[2:]:
            overall = select.loc[:, "ROC(auc)"].mean()
            none = select_none.loc[:, "ROC(auc)"].mean()
            standard = select_standard.loc[:, "ROC(auc)"].mean()
            other = select_other.loc[:, "ROC(auc)"].mean()
        else:
            overall = select.loc[:, "Accuracy scaled"].mean()
            none = select_none.loc[:, "Accuracy scaled"].mean()
            standard = select_standard.loc[:, "Accuracy scaled"].mean()
            other = select_other.loc[:, "Accuracy scaled"].mean()
        print("\nP", P, "N", N)
        print("Overall:", overall, "None", none, "Standard", standard, "Other", other)
        overall_n.append(overall)
        none_n.append(none)
        standard_n.append(standard)
        other_n.append(other)
    score_dict["overall"].append(overall_n)
    score_dict["none"].append(none_n)
    score_dict["standard"].append(standard_n)
    score_dict["other"].append(other_n)

test_sizes_p = [x+"P" for x in test_sizes]
test_sizes_n = [x+"N" for x in test_sizes]
ds = [score_dict["overall"], score_dict["none"], score_dict["standard"], score_dict["other"]]

# Heatmap
import matplotlib.pyplot as plt
from matplotlib import cm
import heatmap

for d in ds:
    scores = np.array(d)
    fig, ax = plt.subplots()
    im, cbar = heatmap.heatmap(scores, test_sizes_p, test_sizes_n, ax=ax,
                       vmin = 0.0, vmax = 1.0, cmap=cm.coolwarm, cbarlabel="score [AUC / Sp]")
    texts = heatmap.annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()


# 3D plot
#from plot_3d import plot_3d
#plot_3d(np.array(score_dict["overall"]), test_sizes_p, test_sizes_n)
