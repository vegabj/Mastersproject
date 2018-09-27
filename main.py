# Import training data

import data_reader

df, labels, groups = data_reader.read_hepmark_microarray()
print(df.values)
# Transform labels to real values
labels = [0 if l == 'Normal' else 1 if l == 'Tumor' else 2 for l in labels]
print(labels)
print(df.values.shape)


# Train classifier
# TODO
import classifier
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(df.values, labels)


# Make prediction
print(df.iloc[0].values)
print(clf.predict([df.iloc[0].values]))
print(clf.predict([df.iloc[1].values]))
