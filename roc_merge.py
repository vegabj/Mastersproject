import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from scaler import MiRNAScaler

# Import data
names = data_reader.get_sets()
print("Available data sets are:")
for i,e in enumerate(names):
    print(str(i)+":", e)
selected = input("Select data set (multiselect separate with ' '): ")
selected = selected.split(' ')

multi_select = False if len(selected) == 1 else True
if multi_select:
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
else:
    df, target, group = data_reader.read_number(int(selected[0]))
    lengths = [df.values.shape[0]]

# Scale data
X = MiRNAScaler.set_scaler(df, lengths)

# Set seed for reproducability
np.random.seed(0)


n_samples, n_features = X.shape

# Transform labels to real values
y = target
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in y])
#NOTE: label 2 should not be in any label

cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(n_estimators = 100)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    # Get class probabilities for test set
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Start validation:
classifier.fit(X, y)
select = int(input("Validation set (0-11): "))
df_val, tar_val, _ = data_reader.read_number(select)
df_val["target"] = tar_val
df_val_pos = df_val.loc[df_val["target"] == "Normal"]
df_val_neg = df_val.loc[df_val["target"] == "Tumor"]
features = df.axes[1].values

while True:
    # Fetch validation params
    number_of_positive = int(input("pos(0-"+str(len(df_val_pos))+"): "))
    number_of_negative = int(input("neg(0-"+str(len(df_val_neg))+"): "))
    df_val_final, tar_val = df_utils.fetch_df_samples(df_val, number_of_positive, number_of_negative)

    # Allign feature axis
    df_val_final = df_utils.merge_frames([df, df_val_final]).tail(len(df_val_final))
    df_val_final = df_val_final.loc[:, features]

    X_val = StandardScaler().fit_transform(df_val_final.values)
    y_val = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in tar_val])

    scores = classifier.predict_proba(X_val)

    fpr, tpr, thresholds = roc_curve(y_val, scores[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2,
             label='ROC val (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
