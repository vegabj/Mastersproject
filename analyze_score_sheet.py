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

overall_nn = []
for P in test_sizes:
    overall_n = []
    for N in test_sizes:
        select = df.loc[(df["P"] == P) & (df["N"] == N)]
        select_none = select.loc[df["Normalization"] == 'None']
        select_standard = select.loc[df["Normalization"] == 'Standard']
        select_other = select.loc[df["Normalization"] == 'Other']
        overall = select.loc[:, "Value"].mean()
        none = select_none.loc[:, "Value"].mean()
        standard = select_standard.loc[:, "Value"].mean()
        other = select_other.loc[:, "Value"].mean()
        print("\nP", P, "N", N)
        print("Overall:", overall, "None", none, "Standard", standard, "Other", other)
        overall_n.append(overall)
    overall_nn.append(overall_n)

test_sizes_p = [x+"P" for x in test_sizes]
test_sizes_n = [x+"N" for x in test_sizes]
# Heatmap example
import matplotlib.pyplot as plt
from matplotlib import cm
import heatmap
overall_nn = np.array(overall_nn)
fig, ax = plt.subplots()
im, cbar = heatmap.heatmap(overall_nn, test_sizes_p, test_sizes_n, ax=ax,
                   cmap=cm.coolwarm, cbarlabel="score [AUC / Sp]")
texts = heatmap.annotate_heatmap(im, valfmt="{x:.1f}")

fig.tight_layout()
plt.show()


# 3D plot example
from plot_3d import plot_3d
plot_3d(overall_nn, test_sizes_p, test_sizes_n)
