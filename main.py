import numpy as np
import data_reader
import df_utils

# Import training data
df_training, labels_training, groups_training = data_reader.read_hepmark_microarray()
training_length = df_training.values.shape[0]

# Import testing data
df_test, labels_test, groups_test = data_reader.read_hepmark_tissue_formatted()
test_length = df_test.values.shape[0]

# Make dfs comparable
df = df_utils.merge_frames(df_training, df_test)
df_training = df.head(training_length)
df_test = df.tail(test_length)

# Transform labels to real values
labels_training = [0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in labels_training]
labels_test = [0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in labels_test]
#labels = np.append(labels_training, labels_test)

# Normalize test Increases hits from ~0.5 to 0.7
from sklearn.preprocessing import StandardScaler
df_training = StandardScaler().fit_transform(df_training)
df_test = StandardScaler().fit_transform(df_test)

# Train classifier
# TODO
import classifier
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf = clf.fit(df_training, labels_training)


# Make predictions
training_hits = 0
for val, label in zip(df_training, labels_training):
    if clf.predict([val]) == label:
        training_hits += 1
print("Training:", training_hits/len(labels_training))

test_hits = 0
for val, label in zip(df_test, labels_test):
    if clf.predict([val]) == label:
        test_hits += 1
print("Test:", test_hits/len(labels_test))

print(clf.predict(df_test))
print(labels_test)
