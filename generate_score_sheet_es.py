"""
Vegard Bjørgan 2019

Generator for score spreadsheet enrichment score
Data sets are selected in user interface
file name must be manually set
the sheet is saved to /Out/scores/
"""

import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from os import getcwd
import scaler as MiRNAScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn import svm

file_name = "Scores_colon_es_svm"

"""
params: es_df - df with enrichment scores
returns: classification based on es_df
"""
def score_es(es_df, bias=0.0):
    es_names = es_df.axes[1]
    es_normal = [es for es in es_names if "Normal_" in es]
    es_tumor = [es for es in es_names if "Tumor_" in es]
    normal_score = es_df.loc[:, es_normal].sum(axis=1)
    tumor_score = es_df.loc[:, es_tumor].sum(axis=1)
    out = []
    for t,n in zip(tumor_score, normal_score):
        if t + bias > n:
            out.append(1)
        else:
            out.append(0)
    return out

# Import data
names = data_reader.get_sets()
print("Available data sets are:")
for i,e in enumerate(names):
    print(str(i)+":", e)
selected = input("Select data set (multiselect separate with ' '): ")
selected = selected.split(' ')

dfs, target, group, es = [], [], [], []
for select in selected:
    df, tar, grp = data_reader.read_number(int(select))
    es_df = data_reader.read_es(int(select))
    dfs.append(df)
    target.extend(tar)
    group.extend(grp)
    es.append(es_df)
df = df_utils.merge_frames(dfs)
es = pd.concat(es, axis=0)
lengths = [d.values.shape[0] for d in dfs]

# Prep features and target labels
features = df.axes[1]
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in target])
df["target"] = y

# Set seed for reproducability
np.random.seed(0)

# Run scoring
test_sizes = [0, 1, 2, 4, 8, 16, 'all']
out_df = pd.DataFrame(columns=["P", "N", "Dataset", "ROC(auc)", "Accuracy"
    , "Balanced accuracy", "Iteration", "Normalization"])

# ES
es_dropped = None
es_values = [None, None]

current_length = 0
for idx, length in enumerate(lengths):
    # Remove / Add ES data that can - todo move to method?
    # ES
    if es_dropped:
        es['Normal_'+es_dropped] = es_values[0]
        es['Tumor_'+es_dropped] = es_values[1]
    es_dropped=selected[idx] if selected[idx] != "6" else None
    if es_dropped:
        es_values = [es.loc[:, "Normal_"+es_dropped], es.loc[:, "Tumor_"+es_dropped]]
    features = df.axes[1].drop("target")

    DS = names[int(selected[idx])]
    X_test = df.tail(len(df)-current_length).head(length)
    X_train = df.drop(X_test.index)
    y_train = X_train.loc[:, "target"]

    es_test = es.tail(len(es)-current_length).head(length)
    es_train = es.drop(es_test.index)
    X_train = np.concatenate((X_train, es_train.values), axis=1)

    classifier = svm.SVC(probability=True)
    classifier.fit(es_train, y_train)

    current_length += length
    for P in tqdm(test_sizes):
        for N in test_sizes:
            if (P == 0 and N == 0) or (not df_utils.check_df_samples(X_test, P, N)):
                continue
            for i in range(len(X_test)):
                df_val, y_test = df_utils.fetch_df_samples(X_test, P, N)
                es_val = es_test.loc[df_val.index]

                # Do performance
                scores = classifier.predict_proba(es_val)[:, 1]
                if (P in test_sizes[2:] and N in test_sizes[2:]):
                    roc = roc_auc_score(y_test, scores)
                else:
                    roc = "N/A"
                scores_class = np.array([1 if s > 0.5 else 0 for s in scores])
                acc = accuracy_score(y_test, scores_class)
                if (P in test_sizes[1:] and N in test_sizes[1:]):
                    balanced_acc = balanced_accuracy_score(y_test, scores_class)
                else:
                    balanced_acc = acc
                balanced_acc = balanced_accuracy_score(y_test, scores_class)
                out_df = out_df.append({"P": P, "N": N, "Dataset": DS, "ROC(auc)": roc
                    , "Accuracy": acc, "Balanced accuracy": balanced_acc, "Iteration": i
                    , "Normalization": "N/A"} , ignore_index = True)

# Save scores to file
out_path = r'%s' % getcwd().replace('\\','/') + "/Out/scores/"+file_name+".csv"
out_df.to_csv(out_path)
