"""
Vegard Bjørgan 2019

Code adapted from:
https://gist.github.com/WillKoehrsen/ff77f5f308362819805a3defd9495ffd
"""

from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from subprocess import call

def visualize(estimator, path, feature_names, target_names, name='tree'):
    # Export as dot file
    export_graphviz(estimator, out_file=path+name+'.dot',
                    feature_names = feature_names,
                    label = 'all',
                    impurity = False,
                    #leaves_parallel = True,
                    class_names = target_names,
                    rounded = True,
                    precision = 2, filled = True)

    # Convert to pdf using system command (requires Graphviz)
    call(['dot', '-Tpdf', path+name+'.dot', '-o', path+name+'.pdf'])

    # Display in matplotlib
    #img=mpimg.imread(path+name+'.png')
    #imgplot = plt.imshow(img)
    #plt.show()


def visualize_test():
    from sklearn.datasets import load_iris
    iris = load_iris()
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=10)
    # Train
    model.fit(iris.data, iris.target)
    # Extract single tree
    estimator = model.estimators_[5]
    from os import getcwd
    path = r'%s' % getcwd().replace('\\','/') + "/Out/images/test/"

    visualize(estimator, path, iris.feature_names, iris.target_names, name='treetest')

#visualize_test()
