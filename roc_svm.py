import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy import interp
import scaler as MiRNAScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report

# Import data
df, target, group, lengths, es = data_reader.read_main()

# Scale data
#scales, values = MiRNAScaler.set_scales(df, lengths)
X = MiRNAScaler.set_scaler(df, lengths)

# Set seed for reproducability
np.random.seed(0)

n_samples, n_features = X.shape

# Transform labels to real values
y = target
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in y])
#NOTE: label 2 should not be in any label


cv = StratifiedKFold(n_splits=10)

# Set the parameters by cross-validation
"""
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.01, 1e-3, 0.002, 0.003, 0.005, 0.004, 0.006, 1e-4, 1e-5],
                     'C': [0.1, 1, 5, 10]},
                    {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'degree': [1,2,3],
                    'gamma': [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5], 'coef0': [0.0, 1.0]}
                    ]
"""
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.01, 1e-3, 1e-4],
                     'C': [0.1, 1, 10]},
                    #{'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [1,2,3],
                    #'gamma': [0.1, 0.01, 1e-3, 1e-4], 'coef0': [0.0, 1.0]}
                    ]

classifier = GridSearchCV(svm.SVC(probability=True), tuned_parameters, n_jobs=-1, iid=False, cv=5, scoring='roc_auc')
# DS 0-2 score: 0.91 +/- 0.04
# DS 3-7 score: 0.68 +/- 0.20
# DS 3, 5-7 score: 0.95 +/- 0.07

# Default SVC params: C=1.0, rbf, gamma = 'auto' (1/n_features)
#classifier = BaggingClassifier(base_estimator=svm.SVC(kernel='rbf', gamma='auto', probability=True), n_estimators=100, max_samples=1.0, max_features=1.0, n_jobs=-1, random_state = 0)

# DS 0-2 score: 0.90 +/- 0.03
# DS 3-7 score: 0.73 +/- 0.21
# DS 3, 5-7 score: 0.95 +/- 0.06
# Bagging + GridSearchCV: # classifier, random_state = 0)
# DS 0-2 score: 0.88 +/- 0.05
# DS 3-7 score: 0.67 +/- 0.16
# DS 3, 5-7 score: 0.90 +/- 0.07

#classifier = AdaBoostClassifier(base_estimator=svm.SVC(probability=True, kernel='rbf', gamma='auto'), max_samples=0.5, random_state = 0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

for train, test in cv.split(X, y):
    # Get class probabilities for test set
    classifier.fit(X[train], y[train])
    #print("Best params %d :" % i+1)
    #print(classifier.best_params_)
    '''
    print("Grid scores on development set:")
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    '''
    y_true, probas_ = y[test], classifier.predict_proba(X[test])[:, 1]
    print(classifier.decision_function(X[test]))
    print(classifier.predict(X[test]))
    #pred = classifier.predict(X[test])
    #print(classification_report(y[test], pred, target_names=["Normal", "Tumor"]))

    # Compute ROC curve and area the curve
    fpr, tpr, _ = roc_curve(y[test], probas_)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
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


#classifier.fit(X, y)
#print("Best params overall :")
#print(classifier.best_params_)
