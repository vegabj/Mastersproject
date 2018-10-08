import numpy as np
import data_reader
import df_utils

# Import training data
df_training, labels_training, groups_training = data_reader.read_hepmark_microarray()
training_length = df_training.values.shape[0]

# Import testing data
df_test, labels_test, groups_test = data_reader.read_hepmark_tissue_formatted()
test_length = df_test.values.shape[0]

# Import validation data
df_validation, labels_validation, groups_validation = data_reader.read_hepmark_paired_tissue_formatted()
validation_length = df_validation.values.shape[0]

# Make dfs comparable
df = df_utils.merge_frames([df_training, df_test, df_validation])
df_training = df.head(training_length)
df_test = df.tail(test_length+validation_length).head(test_length)
df_validation = df.tail(validation_length)

# Transform labels to real values
from sklearn.preprocessing import label_binarize
labels_training = [0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in labels_training]
labels_test = [0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in labels_test]
labels_validation = [0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in labels_validation]
labels_training = label_binarize(labels_training, classes=[0, 1])
labels_test = label_binarize(labels_test, classes=[0, 1])
labels_validation = label_binarize(labels_validation, classes=[0, 1])


# Normalize test Increases hits from ~0.5 to 0.7
from sklearn.preprocessing import StandardScaler
df_training = StandardScaler().fit_transform(df_training)
df_test = StandardScaler().fit_transform(df_test)
df_validation = StandardScaler().fit_transform(df_validation)

# Train classifier
# TODO
import classifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
random_state = np.random.RandomState(0)

#clf = KNeighborsClassifier()
#clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                            random_state=random_state))
#clf_score = clf.fit(df_training, labels_training).decision_function(df_test)
clf = RandomForestClassifier(n_estimators = 100)
clf = clf.fit(df_training, labels_training.ravel()) # Ravel makes y to 1d array

# ROC
#TODO
clf_score = clf.predict(df_test)
print(clf_score)
print(labels_test.ravel())

# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(labels_test, clf_score)
#print(len(clf_score))
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(labels_test.ravel(), clf_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# plot roc curve for test
print(fpr)
print(tpr)
import matplotlib.pyplot as plt
plt.figure()
line_width = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=line_width, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()



# Make predictions
test_hits = 0
for val, label in zip(df_test, labels_test):
    if clf.predict([val]) == label:
        test_hits += 1
print("Test:", test_hits/len(labels_test))

val_hits = 0
for val, label in zip(df_validation, labels_validation):
    if clf.predict([val]) == label:
        val_hits += 1
print("Validation:", val_hits/len(labels_validation), "Hits:", val_hits, "of", len(labels_validation))

'''
# Validation ROC
clf_score = clf.predict(df_validation)
fpr, tpr, _ = roc_curve(labels_validation, clf_score)
roc_auc = auc(fpr, tpr)

print(labels_validation, clf_score)
print(fpr)
print(tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=line_width, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
'''
