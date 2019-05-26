"""
Vegard Bjørgan 2019

Create ROC curves for SVM on selected data sets.
Change global variables for boosting, bagging, feature selection
or showing choosen params at each fold
"""

import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy import interp
import scaler as MiRNAScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV

FEATURE_SELECTION = False
SHOW_PARAMS = False
BOOSTING = False
BAGGING = False

# Import data
df, target, group, lengths, es = data_reader.read_main(raw=False)

# Scale data
X = MiRNAScaler.choose_scaling(df, lengths)
#X = MiRNAScaler.individual_scaler(df)

# Set seed for reproducability
np.random.seed(0)

n_samples, n_features = X.shape

# Transform labels to real values
y = target
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in y])
#NOTE: label 2 should not be in any label


cv = StratifiedKFold(n_splits=10)

# Search parameters for cross-validation
full_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.01, 1e-3, 0.002, 0.003, 0.005, 0.004, 0.006, 1e-4, 1e-5],
                     'C': [0.1, 1, 5, 10]},
                    {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'degree': [1,2,3],
                    'gamma': [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5], 'coef0': [0.0, 1.0]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 5, 10]},
                    {'kernel': ['sigmoid'], 'C': [0.1, 1, 5, 10],
                    'gamma': [0.1, 0.01, 1e-3, 1e-4, 1e-5], 'coef0': [0.0, 1.0]}
                    ]
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
                     'C': [0.1, 1, 5, 10]},
                    ]
boosting_parameters = [{'base_estimator': [svm.SVC(kernel='linear', probability=True)],
                        'n_estimators' : [10, 30, 50, 100],
                        'learning_rate' : [0.1, 0.3, 0.5, 1.0]}]

bagging_parameters = [{'base_estimator': [svm.SVC(kernel='linear', probability=True)],
                        'n_estimators' : [10, 30, 50, 100],
                        'max_samples' : [0.1, 0.5, 1.0]}]

classifier = GridSearchCV(svm.SVC(probability=True), tuned_parameters, n_jobs=4, iid=False, cv=5, scoring='roc_auc')
if BOOSTING:
    classifier = GridSearchCV(AdaBoostClassifier(random_state = 0), boosting_parameters, n_jobs=4, iid=False, cv=5, scoring='roc_auc')
if BAGGING:
    classifier = GridSearchCV(BaggingClassifier(), bagging_parameters, n_jobs=4, iid=False, cv=5, scoring='roc_auc')

# Setup For Feature selection
if FEATURE_SELECTION:
    print("Before FS:",X.shape[1])
    fs_params = [{'kernel': ['linear'], 'C': [0.1, 1, 10]}]
    grid_estimators = GridSearchCV(svm.SVC(probability=True), fs_params, n_jobs=1, iid=False, cv=5, scoring='roc_auc')


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

for train, test in cv.split(X, y):
    if FEATURE_SELECTION:
        grid_estimators.fit(X[train], y[train])
        selector = RFECV(grid_estimators.best_estimator_, step=5 , cv=3, scoring='roc_auc')
        selector = selector.fit(X[train], y[train])
        X_r = selector.transform(X)
        print("After FS"+str(i+1)+":",X_r.shape[1])
    else:
        X_r = X

    # Get class probabilities for test set
    classifier.fit(X_r[train], y[train])

    # Code for looking at parameters
    if SHOW_PARAMS:
        print("Best params "+str(i+1)+" :")
        print(classifier.best_params_)
        print("Grid scores on development set:")
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

    y_true, probas_ = y[test], classifier.predict_proba(X_r[test])[:, 1]

    # Compute ROC curve and area the curve
    fpr, tpr, _ = roc_curve(y[test], probas_)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

    # Compute and print accuracy score
    #scores_class = np.array([1 if s > 0.5 else 0 for s in probas_])
    #print("BACC", i, balanced_accuracy_score(y[test], scores_class))
    #print("ACC", i, accuracy_score(y[test], scores_class))
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
