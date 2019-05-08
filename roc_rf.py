"""
Vegard Bjørgan 2019

Create ROC curves for Random Forest on selected data sets.
Change global variables for feature selection or showing choosen params at each fold
"""
import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy import interp
import scaler as MiRNAScaler
from sklearn.neighbors import KNeighborsClassifier
from utils import latexify
from sklearn.feature_selection import RFECV

FEATURE_SELECTION = False
GRID_SEARCH = False

# Import data
df, target, group, lengths, es = data_reader.read_main(raw=False)


# Scale data
X = MiRNAScaler.choose_scaling(df, lengths)
print(X.shape)

# Transform labels to real values
y = target
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in y])

if FEATURE_SELECTION:
    estimator = RandomForestClassifier(n_estimators = 100)

# Set seed for reproducability
np.random.seed(0)


n_samples, n_features = X.shape

# Transform labels to real values
y = target
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in y])
#NOTE: label 2 should not be in any label

cv = StratifiedKFold(n_splits=10)

if GRID_SEARCH:
    tuned_parameters = [{'n_estimators': [10, 50, 100, 200, 500] ,
                        'criterion': ['gini', 'entropy'],
                         'max_features': ['auto', 'log2', 1.0, 0.5]}
                        ]
    classifier = GridSearchCV(RandomForestClassifier(), tuned_parameters, n_jobs=1, iid=False, cv=5, scoring='roc_auc')
else:
    classifier = RandomForestClassifier(n_estimators = 200)


if FEATURE_SELECTION:
    print("Before FS:",X.shape[1])

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    if FEATURE_SELECTION:
        selector = RFECV(estimator, step=1 , cv=3, scoring='roc_auc')
        selector = selector.fit(X[train], y[train])
        X_r = selector.transform(X)
        print("After FS"+str(i+1)+":",X_r.shape[1])
    else:
        X_r = X
    # Get class probabilities for test set
    classifier.fit(X_r[train], y[train])

    # Grid search output
    if GRID_SEARCH:
        print("Grid scores on development set:")
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

    probas_ = classifier.predict_proba(X_r[test])

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
