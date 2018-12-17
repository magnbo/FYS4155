import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(font='serif', font_scale=0.9)


X = np.loadtxt('default_X.txt')
Y = np.loadtxt('default_Y.txt')

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.8)

max_depth = np.ceil(np.linspace(1, 26, 6))
min_weight_fraction_leaf = [0.00005, 0.005, 0.01, 0.015, 0.2]
min_samples_leaf = [1, 4, 8, 12, 200]
m = len(max_depth)
n = len(min_samples_leaf)
tree_cross_value_score = np.zeros((m, n))
train_accuracy = np.zeros((m, n))
for i in range(0, m):
    for j in range(0, n):
        tree_clf = DecisionTreeClassifier(max_depth=max_depth[i],min_samples_leaf=min_samples_leaf[j])
        tree_clf.fit(X_train, Y_train)
        tree_cross_value_score[i, j] = cross_val_score(tree_clf, X_test, Y_test, cv=5).mean()
        train_accuracy[i, j] = tree_clf.score(X_train, Y_train)

ax = plt.figure()
sns.heatmap(train_accuracy, xticklabels=min_samples_leaf, yticklabels=max_depth, annot=True, cmap="viridis", cbar=False)
plt.xlabel('min samples leaf')
plt.ylabel('max depth')
plt.show()

ax = plt.figure()
sns.heatmap(tree_cross_value_score, xticklabels=min_samples_leaf, yticklabels=max_depth, annot=True, cmap="viridis", cbar=False)
plt.xlabel('min samples leaf')
plt.ylabel('max depth')
plt.show()
