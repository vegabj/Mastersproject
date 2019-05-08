"""
Vegard Bjørgan 2019

Prints the feature importance in SVM and Random Forest
Creates a plot over top 20 features in SVM
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
from scipy import interp
import scaler as MiRNAScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import utils

# Set seed for reproducability
np.random.seed(0)

# Import data
df, target, group, lengths, es = data_reader.read_main()

# Scale data
X = MiRNAScaler.choose_scaling(df, lengths)
y = target


def plot_coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure()
    plt.tight_layout()
    colors = ['blue' if c < 0 else 'red' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=40, ha='right')
    plt.show()

linear_svm = LinearSVC()
linear_svm.fit(X, y)
rf_classifier = RandomForestClassifier(n_estimators = 200)
rf_classifier.fit(X, y)
#utils.latexify(columns=2)
#features = [feat[8:] for feat in df.axes[1].values]
#plot_coefficients(linear_svm, features)

features = [feat for feat in df.axes[1].values]
feature_scores_svm = linear_svm.coef_.ravel()

print("Sorted Feature SVM list:")
feature_list = [(x,z) for z,x in sorted(zip(feature_scores_svm, df.axes[1].values))]
for f in feature_list[::-1]:
    print(f)


feature_scores_rf = rf_classifier.feature_importances_
# Scatter of features
d = {'rf': np.array(feature_scores_rf), 'svm': np.array(feature_scores_svm)}
feature_df = pd.DataFrame(data=d, index=features)
print(feature_df)

print("Sorted Feature list RF:")
feature_list = [(x,s) for s,x in sorted(zip(feature_scores_rf, features))]
for f in feature_list:
    print(f)

plt.scatter(feature_scores_rf, feature_scores_svm)
plt.show()
