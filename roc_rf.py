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
import scaler as MiRNAScaler

# Import data
df, target, group, lengths = data_reader.read_main(raw=False, es=True)
#df = df[['Normal_3', 'Normal_5', 'Normal_7', 'Tumor_3', 'Tumor_5', 'Tumor_7']]


# Scale data
scales, values = MiRNAScaler.set_scales(df, lengths)
X = MiRNAScaler.set_scaler(df, lengths)

# Set seed for reproducability
np.random.seed(0)


n_samples, n_features = X.shape

# Transform labels to real values
y = target
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in y])
#NOTE: label 2 should not be in any label

cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(n_estimators = 200)

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


'''
# Start validation:
classifier.fit(X, y)
select = int(input("Validation set (0-11): "))
df_val, tar_val, _ = data_reader.read_number(select)
df_val["target"] = tar_val
df_val_neg = df_val.loc[df_val["target"] == "Normal"]
df_val_pos = df_val.loc[df_val["target"] == "Tumor"]
features = df.axes[1].values


while True:
    # Fetch validation params
    number_of_positive = int(input("pos samples(0-"+str(len(df_val_pos))+"): "))
    number_of_negative = int(input("neg samples(0-"+str(len(df_val_neg))+"): "))
    df_val_final, tar_val = df_utils.fetch_df_samples(df_val, number_of_positive, number_of_negative)

    # Allign feature axis
    df_val_final = df_utils.merge_frames([df, df_val_final]).tail(len(df_val_final))
    # Drop non traied features
    df_val_final = df_val_final.loc[:, features]

    # Normalization strategy
    """
    * Check comparable mean and std for each scale.
    * Choose the scale that is closest.
    * TODO: Assumption that training frame has more than or equal columns
    * Smart solution to when enough samples is gathered
    """
    # Standard
    X_val1 = StandardScaler().fit_transform(df_val_final.values)
    # non-normalized
    X_val2 = df_val_final.values
    # From others
    val_scores = []
    val_means, val_std = df_val_final.mean(axis=0), df_val_final.std(axis=0)
    for value in values:
        val_score = ((val_means - value[0]) ** 2).sum(0) ** .5 + ((val_std - value[1]) ** 2).sum(0) ** .5
        val_scores.append(val_score)
    myscale = scales[val_scores.index(min(val_scores))]
    X_val3 = myscale.transform(df_val_final.values)

    y_val = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in tar_val])
    X_val = [X_val1, X_val2, X_val3]
    names = ["Standard", "Non_normalized", "From others"]
    for X_val_sample, name in zip(X_val, names):
        scores = classifier.predict_proba(X_val_sample)
        fpr, tpr, thresholds = roc_curve(y_val, scores[:, 1])
        roc_auc = auc(fpr, tpr)
        #print(tpr, fpr)

        # NB: Sove problem with no roc curve if 0 neg or 0 pos samples
        if number_of_positive == 0 or number_of_negative == 0:
            scores = classifier.predict(X_val_sample)
            correct = abs(sum(y_val) - sum(scores))
            print(name, (len(y_val)-correct)/len(y_val))
        plt.plot(fpr, tpr, lw=2,
                 label='ROC val '+name+' (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
'''
