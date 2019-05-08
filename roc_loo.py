"""
Vegard Bjørgan 2019

ROC curve for Leave One Out
"""
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
from sklearn.model_selection import LeaveOneOut
from scipy import interp
import scaler as MiRNAScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# Import data
df, target, group, lengths, es = data_reader.read_main()

features = df.axes[1]
samples = df.axes[0]

# Scale data
print(df.shape)
X = MiRNAScaler.choose_scaling(df, lengths)
df = pd.DataFrame(X, index=samples, columns=features)
print("DF shape", X.shape)

# Set seed for reproducability
np.random.seed(0)

n_samples, n_features = X.shape

# Transform labels to real values
y = target
y = np.array([0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in y])
#NOTE: label 2 should not be in any label


cv = LeaveOneOut()

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5],
                     'C': [0.1, 1, 5, 10]},
                    #{'kernel': ['poly'], 'C': [0.1, 1, 5, 10], 'degree': [1,2,3],
                    #'gamma': [0.1, 0.01, 1e-3, 1e-4, 1e-5], 'coef0': [0.0, 1.0]},
                    #{'kernel': ['linear'], 'C': [0.1, 1, 5, 10]},
                    #{'kernel': ['sigmoid'], 'C': [0.1, 1, 5, 10],
                    #'gamma': [0.1, 0.01, 1e-3, 1e-4, 1e-5], 'coef0': [0.0, 1.0]}
                    ]

classifier = GridSearchCV(svm.SVC(probability=True), tuned_parameters, n_jobs=4, iid=False, cv=3 )
classifier = RandomForestClassifier(n_estimators = 200)

score = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

for train, test in tqdm(cv.split(X, y)):
    # Get class probabilities for test set
    classifier.fit(X[train], y[train])
    score.append(classifier.score(X[test], y[test]))
    '''
    y_true, probas_ = y[test], classifier.predict(X[test])
    #print(y_true, probas_)
    if y_true == probas_:
        score += 1
    '''
    i += 1

#print(score,i)
score = np.array(score)
print(score.mean())
