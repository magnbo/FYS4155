import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

X = np.loadtxt('default_X.txt')
Y = np.loadtxt('default_Y.txt')

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.8)

tree_clf = DecisionTreeClassifier(max_depth=1,min_samples_leaf=200)
tree_clf.fit(X_train, Y_train)
tree.export_graphviz(tree_clf, out_file='tree.dot')
