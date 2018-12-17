import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import timeit
import matplotlib.pyplot as plt


X = np.loadtxt('default_X.txt')
Y = np.loadtxt('default_Y.txt')

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, train_size=0.8)

import timeit

import warnings
#Comment to turn on warnings
warnings.filterwarnings("ignore")

#We will check 



min_estimators = 10
max_estimators = 101
n_estimators=np.arange(min_estimators, max_estimators, 10)
leaf_sizes=np.arange(2, 10, 1)
n=len(n_estimators)
m=len(leaf_sizes)

#Allocate Arrays for various quantities

RFC_OOB_accuracy=np.zeros((n,m))
RFC_train_accuracy=np.zeros((n,m))
RFC_test_accuracy=np.zeros((n,m))
RFC_critical_accuracy=np.zeros((n,m))
run_time=np.zeros((n,m))

print_flag=True

for i in range(0,n):
    for j in range(0,m):
        
        print('n_estimators: '+str(n_estimators[i])+', leaf_size: '+str(leaf_sizes[j]))
        
        start_time = timeit.default_timer()
        RFC = RandomForestClassifier(n_estimators=n_estimators[i], max_depth=None,min_samples_split=leaf_sizes[j],oob_score=True, random_state=0)
        RFC.fit(X_train, Y_train)
        run_time[i,j] = timeit.default_timer() - start_time

    
    # check accuracy
        RFC_train_accuracy[i,j]=RFC.score(X_train,Y_train)
        RFC_OOB_accuracy[i,j]=RFC.oob_score_
        RFC_test_accuracy[i,j]=RFC.score(X_test,Y_test)
        if print_flag:
            print('accuracy:time, train, OOB estimate,test, critical')
            print('liblin: %0.4f, %0.4f, %0.4f,  %0.4f \n' %(run_time[i,j],RFC_train_accuracy[i,j],RFC_OOB_accuracy[i,j], RFC_test_accuracy[i,j]))



plt.figure()
plt.plot(n_estimators,RFC_train_accuracy[:,1],'--b^',label='Train (coarse)')
plt.plot(n_estimators,RFC_test_accuracy[:,1],'--r^',label='Test (coarse)')
plt.plot(n_estimators,RFC_critical_accuracy[:,1],'--g^',label='Critical (coarse)')

plt.plot(n_estimators,RFC_train_accuracy[:,0],'o-b',label='Train (fine)')
plt.plot(n_estimators,RFC_test_accuracy[:,0],'o-r',label='Test (fine)')
plt.plot(n_estimators,RFC_critical_accuracy[:,0],'o-g',label='Critical (fine)')

#plt.semilogx(lmbdas,train_accuracy_SGD,'*--b',label='SGD train')

plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Accuracy')
lgd=plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("Ising_RF.pdf",bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.show()

plt.plot(n_estimators, run_time[:,0], 'o-k',label='Fine')
plt.plot(n_estimators, run_time[:,1], '--k^',label='Coarse')
plt.xlabel('$N_\mathrm{estimators}$')
plt.ylabel('Run time (s)')


plt.legend(loc=2)
#plt.savefig("Ising_RF_Runtime.pdf")

plt.show()