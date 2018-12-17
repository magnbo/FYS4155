import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

X = np.loadtxt('default_X.txt')
Y = np.loadtxt('default_Y.txt')

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.8)


RFC = RandomForestClassifier()
skplt.estimators.plot_learning_curve(RFC, X, Y, train_sizes=np.linspace(.1, 1.0, 10))
plt.show()
