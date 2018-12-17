import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='serif', font_scale=0.9)


X = np.loadtxt('default_X.txt')
Y = np.loadtxt('default_Y.txt')

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.8)

n_estimators = np.arange(1, 13, 2)
leaf_sizes = np.arange(2, 14, 2)
n= len(n_estimators)
m= len(leaf_sizes)
RFC_train_accuracy=np.zeros((n,m))
RFC_test_accuracy=np.zeros((n,m))

for i in range(0,n):
    for j in range(0,m):
        RFC = RandomForestClassifier(n_estimators=n_estimators[i], min_samples_split=leaf_sizes[j], random_state=0)
        RFC.fit(X_train, Y_train)
        RFC_train_accuracy[i,j] = RFC.score(X_train, Y_train)
        RFC_test_accuracy[i,j] = cross_val_score(RFC, X_test, Y_test, cv=5).mean()
        std = cross_val_score(RFC, X_test, Y_test, cv=5).std()


ax = plt.figure()
sns.heatmap(RFC_train_accuracy, xticklabels=leaf_sizes, yticklabels=n_estimators, annot=True, cmap="viridis", cbar=False)
plt.xlabel('Grener')
plt.ylabel('Trær')
plt.show()

ax = plt.figure()
sns.heatmap(RFC_test_accuracy, xticklabels=leaf_sizes, yticklabels=n_estimators, annot=True, cmap="viridis", cbar=False)
plt.xlabel('Grener')
plt.ylabel('Trær')
plt.show()
