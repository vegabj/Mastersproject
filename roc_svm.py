import numpy as np
import pandas as pd
import data_reader
import df_utils
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from scaler import MiRNAScaler
from sklearn.model_selection import GridSearchCV

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

# Set the parameters by cross-validation
# Sigmoid kernel is rearly the best
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.01, 1e-3, 0.002, 0.003, 0.005, 0.004, 0.006 1e-4, 1e-5],
                     'C': [0.1, 1, 10, 100]},
                    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                    {'kernel': ['poly'], 'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5], 'coef0': [0.0, 1.0]},
                    {'kernel': ['sigmoid'], 'C': [0.1, 1, 10, 100],
                    'gamma': [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5], 'coef0': [0.0, 1.0]}]

classifier = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='roc_auc')

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

for train, test in cv.split(X, y):
    # Get class probabilities for test set
    classifier.fit(X[train], y[train])
    print("Best params %d :" % i+1)
    print(classifier.best_params_)
    '''
    print("Grid scores on development set:")
    means = classifier.cv_results_['mean_test_score']
    stds = classifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    '''
    y_true, probas_ = y[test], classifier.predict(X[test])

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


classifier.fit(X, y)
print("Best params overall :")
print(classifier.best_params_)
