'''
Vegard Bjørgan 2018

Generator for score spreadsheet
'''
import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from os import getcwd
import scaler as MiRNAScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
import warnings

# TODO: Fix warnings from using f1_score, balanced_accuracy_score etc
warnings.filterwarnings("ignore")

file_name = "Scores_colon_rf"
use_enrichment_score = False

"""
params: es_df - df with enrichment scores
returns: classification based on es_df
"""
def score_es(es_df):
    es_names = es_df.axes[1]
    es_normal = [es for es in es_names if "Normal_" in es]
    es_tumor = [es for es in es_names if "Tumor_" in es]
    normal_score = es_df.loc[:, es_normal].sum(axis=1)
    tumor_score = es_df.loc[:, es_tumor].sum(axis=1)
    out = []
    for t,n in zip(tumor_score, normal_score):
        if t > n:
            out.append("Tumor")
        else:
            out.append("Normal")
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

# Setup classifier
'''
tuned_parameters = {'kernel': ['rbf'], 'gamma': [0.01, 1e-3, 0.002, 0.003, 0.005, 0.004, 0.006, 1e-4],
                     'C': [0.1, 1, 10, 100]}
classifier = GridSearchCV(svm.SVC(probability=True), tuned_parameters, cv=3, scoring='roc_auc')

classifier = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', gamma='auto', probability=True), n_estimators=100, max_samples=1.0, max_features=1.0, random_state = 0)
'''
classifier = RandomForestClassifier(n_estimators = 200)

# Run scoring
test_sizes = [0, 1, 2, 4, 8, 16, 'all']
out_df = pd.DataFrame(columns=["P", "N", "Dataset", "Precision", "Recall",
    "F1 score", "ROC(auc)", "Accuracy", "Balanced accuracy", "Iteration", "Normalization"])

# ES
es_dropped = None
es_values = [None, None]

current_length = 0
for idx, length in enumerate(lengths):
    # Remove / Add ES data that can / can not be used.
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

    # Scale training data
    scales, values = MiRNAScaler.set_scales(X_train.loc[:, features], lengths[:idx] + lengths[idx+1:])
    X_train = MiRNAScaler.set_scaler(X_train.loc[:, features], lengths[:idx] + lengths[idx+1:])
    # ES
    # Add es_scores to X_train and X_test
    if use_enrichment_score:
        es_test = es.tail(len(es)-current_length).head(length)
        es_train = es.drop(es_test.index)
        X_train = np.concatenate((X_train, es_train.values), axis=1)

    # Fit classifier
    classifier.fit(X_train, y_train)

    current_length += length
    for P in tqdm(test_sizes):
        for N in test_sizes:
            if (P == 0 and N == 0) or (not df_utils.check_df_samples(X_test, P, N)):
                continue
            for i in range(len(X_test)):
                #df_vals target gets removed here
                df_val, y_test = df_utils.fetch_df_samples(X_test, P, N)

                # Normalization strategy
                for ii in range(3):
                    if ii == 0:
                        normalization = "Standard"
                        df_val_final = StandardScaler().fit_transform(df_val.values)
                    if ii == 1:
                        normalization = "None"
                        df_val_final = df_val.values
                    if ii == 2:
                        normalization = "Closest"
                        val_scores = []
                        val_means, val_std = df_val.mean(axis=0), df_val.std(axis=0)
                        for value in values:
                            val_score = ((val_means - value[0]) ** 2).sum(0) ** .5 + ((val_std - value[1]) ** 2).sum(0) ** .5
                            val_scores.append(val_score)
                        myscale = scales[val_scores.index(min(val_scores))]
                        df_val_final = myscale.transform(df_val.values)

                    # ES
                    if use_enrichment_score:
                        es_val = es_test.loc[df_val.index]
                        df_val_final = np.concatenate((df_val_final, es_val.values), axis=1)

                    # Do performance
                    scores = classifier.predict_proba(df_val_final)[:, 1]
                    if (P in test_sizes[2:] and N in test_sizes[2:]):
                        roc = roc_auc_score(y_test, scores)
                    else:
                        roc = "N/A"
                    scores_class = np.array([1 if s > 0.5 else 0 for s in scores])
                    acc = accuracy_score(y_test, scores_class)
                    precision = precision_score(y_test, scores_class)
                    recall = recall_score(y_test, scores_class)
                    balanced_acc = balanced_accuracy_score(y_test, scores_class)
                    f1 = f1_score(y_test, scores_class)
                    out_df = out_df.append({"P": P, "N": N, "Dataset": DS, "Precision": precision
                        , "Recall": recall, "F1 score": f1, "ROC(auc)": roc, "Accuracy": acc, "Balanced accuracy": balanced_acc, "Iteration": i
                        , "Normalization": normalization} , ignore_index = True)

# Save scores to file
out_path = r'%s' % getcwd().replace('\\','/') + "/Out/scores/"+file_name+".csv"
out_df.to_csv(out_path)
