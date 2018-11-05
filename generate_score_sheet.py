'''
Vegard Bjørgan 2018

Generator for score spreadsheet
'''
import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from os import getcwd
from scaler import MiRNAScaler
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Import data
names = data_reader.get_sets()
print("Available data sets are:")
for i,e in enumerate(names):
    print(str(i)+":", e)
selected = input("Select data set (multiselect separate with ' '): ")
selected = selected.split(' ')

dfs = []
targets = []
groups = []
for select in selected:
    df, tar, grp = data_reader.read_number(int(select))
    dfs.append(df)
    targets.append(tar)
    groups.append(grp)

df = df_utils.merge_frames(dfs)
target = targets[0]
group = groups[0]
for tar, gro in zip(targets[1:], groups[1:]):
    target = np.append(target, tar)
    group = np.append(group, gro)
lengths = [d.values.shape[0] for d in dfs]

# Prep features and target labels
features = df.axes[1]
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in target])
df["target"] = y

# Setup classifier
classifier = RandomForestClassifier(n_estimators = 100)

# Set seed for reproducability
np.random.seed(0)
# Run scoring
test_sizes = [0, 1, 2, 4, 8, 16, 'all']
out_df = pd.DataFrame(columns=["P", "N", "Dataset", "Value", "Performance"
    , "Iteration", "Normalization"])

current_length = 0
for idx, length in enumerate(lengths):
    DS = names[int(selected[idx])]
    X_test = df.tail(len(df)-current_length).head(length)
    X_train = df.drop(X_test.index)
    y_train = X_train.loc[:, "target"]

    # Scale training data
    scales, values = MiRNAScaler.set_scales(X_train.loc[:, features], lengths[:idx] + lengths[idx+1:])
    X_train = MiRNAScaler.set_scaler(X_train.loc[:, features], lengths[:idx] + lengths[idx+1:])

    # Fit classifier
    classifier.fit(X_train, y_train)

    current_length += length
    for P in test_sizes:
        for N in test_sizes:
            if (P == 0 and N == 0) or (not df_utils.check_df_samples(X_test, P, N)):
                continue
            for i in range(len(X_test)):
                df_val, y_test = df_utils.fetch_df_samples2(X_test, P, N)
                #df_vals target gets removed here

                # Normalization strategy
                for ii in range(3):
                    if ii == 0:
                        df_val_final = StandardScaler().fit_transform(df_val.values)
                        normalization = "Standard"
                    if ii == 1:
                        df_val_final = df_val.values
                        normalization = "None"
                    if ii == 2:
                        val_scores = []
                        val_means, val_std = df_val.mean(axis=0), df_val.std(axis=0)
                        for value in values:
                            val_score = ((val_means - value[0]) ** 2).sum(0) ** .5 + ((val_std - value[1]) ** 2).sum(0) ** .5
                            val_scores.append(val_score)
                        myscale = scales[val_scores.index(min(val_scores))]
                        df_val_final = myscale.transform(df_val.values)
                        normalization = "Other"

                    # Do performance
                    if (P in test_sizes[2:] and N in test_sizes[2:]):
                        performance = "ROC (auc)"
                        scores = classifier.predict_proba(df_val_final)
                        fpr, tpr, thresholds = roc_curve(y_test, scores[:, 1])
                        score = auc(fpr, tpr)
                    else:
                        performance = "Sp"
                        scores = classifier.predict(df_val_final)
                        scores = [1 if s == y_test[jj] else 0 for jj, s in enumerate(scores)]
                        score = sum(scores) / len(scores)
                    out_df = out_df.append({"P": P, "N": N, "Dataset": DS, "Value": score
                        , "Performance": performance, "Iteration": i, "Normalization": normalization}
                        , ignore_index = True)


# Save scores to file
out_path = r'%s' % getcwd().replace('\\','/') + "/Out/Scores.csv"
out_df.to_csv(out_path)
