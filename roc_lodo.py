"""
Vegard Bjørgan 2019

ROC curve for Leave One Dataset Out.
"""
import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy import interp
import scaler as MiRNAScaler
from sklearn.neighbors import KNeighborsClassifier
from utils import latexify
from sklearn import svm

# Import data
df, target, group, lengths, _ = data_reader.read_main(raw=False)

features = df.axes[1]
samples = df.axes[0]

# Scale data
print(df.shape)
X = MiRNAScaler.choose_scaling(df, lengths)
df = pd.DataFrame(X, index=samples, columns=features)
print("DF shape", X.shape)

# Transform labels to real values
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in target])
df["target"] = y

# Set seed for reproducability
np.random.seed(0)

n_samples, n_features = X.shape

#classifier = RandomForestClassifier(n_estimators = 200)
#"""
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
                     'C': [0.1, 1, 5, 10]},]
classifier = GridSearchCV(svm.SVC(probability=True), tuned_parameters, n_jobs=4, iid=False, cv=5, scoring='roc_auc')
#"""
#classifier = svm.LinearSVC()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 1
current_length = 0
for idx, length in enumerate(lengths):
    # Setup train and test sets
    X_test = df.tail(len(df)-current_length).head(length)
    X_train = df.drop(X_test.index)
    y_train = X_train.loc[:, "target"]
    y_test = X_test.loc[:, "target"]
    # Remove target
    X_train = X_train.drop("target", axis=1)
    X_test = X_test.drop("target", axis=1)
    current_length += length

    # Get class probabilities for test set
    classifier.fit(X_train, y_train)
    probas_ = classifier.predict_proba(X_test)

    # Compute ROC curve and area the curve
    if all(x in y_test.values for x in [0,1]):
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    else:
        scores_class = np.array([1 if s > 0.5 else 0 for s in probas_[:, 1]])
        score = accuracy_score(y_test, scores_class)
        print(i, score)
        aucs.append(score)
        tprs.append(tprs[-1])


    i += 1


plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = np.mean(aucs)
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
