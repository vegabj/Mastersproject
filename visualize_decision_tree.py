from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from subprocess import call

def visualize(estimator, path, feature_names, target_names, name='tree'):
    # Export as dot file
    export_graphviz(estimator, out_file=path+name+'.dot',
                    feature_names = feature_names,
                    class_names = target_names,
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    #for pdf: dot -Tpdf graph1.dot -o graph1.pdf
    #call(['dot', '-Tpng', path+name+'.dot', '-o', path+name+'.png', '-Gdpi=600'])
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
    path = r'%s' % getcwd().replace('\\','/') + "/Out/images/"

    visualize(estimator, path, iris.feature_names, iris.target_names)

#visualize_test()
'''
import visualize_decision_tree
path = r'%s' % getcwd().replace('\\','/') + "/Out/images/"
for i, est in enumerate(classifier.estimators_):
    visualize_decision_tree.visualize(est, path, features, target, name=str(i))
x = input("Finished!")
'''
