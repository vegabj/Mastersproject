import pandas as pd
from os import getcwd, listdir
import numpy as np
import matplotlib.pyplot as plt
from utils import latexify

#TODO: Boxplot
latexify(fig_height=3.39*0.838 ,columns=1)
"""
- Read https://scikit-learn.org/stable/modules/preprocessing.html
- And http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html
"""

path = r'%s' % getcwd().replace('\\','/') + "/Out/"
scores = listdir(path)
# Simples ui choice for selecting score dataset
print("Scores to analyze:")
for i,score in enumerate(scores):
    print(i,score)
select = int(input("Select: "))
path = path + scores[select]
df = pd.read_csv(path, index_col = 0)

test_sizes = ['2', '4', '8', '16', 'all']

score_dict = {"none": {}, "standard": {}, "other": {}}
np_none, np_standard, np_other = np.array([]), np.array([]), np.array([])
for P in test_sizes:
    for N in test_sizes:
        select = df.loc[(df["P"] == P) & (df["N"] == N)]
        #select = select.loc[select["Dataset"] == "Hepmark_Paired_Tissue"]
        select_none = select.loc[df["Normalization"] == 'None']
        select_standard = select.loc[df["Normalization"] == 'Standard']
        select_other = select.loc[df["Normalization"] == 'Other']
        none = select_none.loc[:, "ROC(auc)"]
        standard = select_standard.loc[:, "ROC(auc)"]
        other = select_other.loc[:, "ROC(auc)"]
        score_dict["none"][P+N] = none
        score_dict["standard"][P+N] = standard
        score_dict["other"][P+N] = other

        # Gather data for one boxplot of all P and N combinations
        np_none = np.concatenate((np_none, none), axis=None)
        np_standard = np.concatenate((np_standard, standard), axis=None)
        np_other = np.concatenate((np_other, other), axis=None)

        # Generate a boxplot per P and N combination
        #fig, ax = plt.subplots()
        #ax.set_title("Boxplot")
        #ax.boxplot([none, standard, other], labels=['None', 'Standard', 'Closest'])
        #fig.tight_layout()
        #plt.show()



fig, ax = plt.subplots()
ax.set_title("Boxplot")
ax.boxplot([np_none, np_standard, np_other], labels=['None', 'Standard', 'Closest'])
fig.tight_layout()
plt.show()