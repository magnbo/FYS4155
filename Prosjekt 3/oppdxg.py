import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
sns.set(font='serif', font_scale=0.9)


X = np.loadtxt('default_X.txt')
Y = np.loadtxt('default_Y.txt')

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.8)

import xgboost as xgb

param = {'max_depth':2, 'eta':0.1, 'silent':1, 'objective':'binary:hinge' }
num_round = 500
dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)
evallist = [(dtest, 'eval'), (dtrain, 'train')]
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)
ypred = bst.predict(dtest)
accuracy_score = accuracy_score(Y_test, ypred)
print(accuracy_score)
