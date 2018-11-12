import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import pickle,os
from sklearn.model_selection import train_test_split
from numba import jit

# np.random.seed(2)

# eta = 0.1
# n_iterations = 1000
# m = 100

# theta = np.random.randn(2, 1)

# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
# X_b = np.c_[np.ones((100, 1)), X]

@jit
def learning_schedule(t):
    t0, t1 = 5, 50
    return(t0/(t + t1))

def mb_sgd(x_0, y_0, n_epochs = 50):
    m, n = x_0.shape
    mini_batch_size = 32
    mini_batches = int(np.ceil(m / 32))
    theta = np.random.randn(2, 1)
    for epoch in range(n_epochs):
        for i in range(mini_batches):
            xb_random_batch, y_random_batch = resample(x_0, y_0, n_samples=mini_batch_size, replace=False)
            gradients = (2 / mini_batch_size) * xb_random_batch.T @ (xb_random_batch @ theta - y_random_batch)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
    return theta

# theta = mb_sgd(X_b, y)
@jit
def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

@jit
def logistic_function(t_0):
    return 1/(1 + np.exp(-t_0))

@jit
def prediction(x_0, theta_0):
    probability = logistic_function(x_0 @ theta_0)
    m = x_0.shape[0]
    prediction = np.zeros(m)
    for i in range(m):
        if probability[i] < 0.5:
            prediction[i] = 0.0
        else:
            prediction[i] = 1.0
    return prediction

t0, t1 = 5, 50


@jit
def logistic_regression_SGD(x_0, y_0, n_epochs, learning_rate):
    x_train, x_test, y_train, y_test = train_test_split(x_0, y_0, test_size=0.3)
    m, n = x_train.shape
    mini_batch_size = 32
    mini_batches = int(np.ceil(m / mini_batch_size))
    theta = np.zeros(n)
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = x_train[random_index:random_index+1]
            yi = y_train[random_index:random_index+1]
            h = logistic_function(xi @ theta)
            gradients = xi.T @ (h - yi)
            eta = learning_rate #learning_rate * 0.5 ** np.floor(epoch / 5)
            theta = theta - eta * gradients
    train_prediction = prediction(x_train, theta)
    train_accuracy = accuracy_score_numpy(y_train, train_prediction)
    test_prediction = prediction(x_test, theta)
    test_accuracy = accuracy_score_numpy(y_test, test_prediction)
    return(theta, train_accuracy, test_accuracy)

@jit
def logistic_regression_SGDmb(x_0, y_0, n_epochs, learning_rate):
    x_train, x_test, y_train, y_test = train_test_split(x_0, y_0, test_size=0.3)
    m, n = x_train.shape
    mini_batch_size = 32
    mini_batches = int(np.ceil(m / mini_batch_size))
    theta = np.zeros(n)
    for epoch in range(n_epochs):
        for i in range(mini_batches):
            xb_random_batch, y_random_batch = resample(x_0, y_0, n_samples=mini_batch_size, replace=False)
            h = logistic_function(xb_random_batch @ theta)
            gradients = (1 / mini_batch_size) * xb_random_batch.T @ (h - y_random_batch)
            eta = learning_rate #learning_rate * 0.5 ** np.floor(epoch / 5)
            theta = theta - eta * gradients
    train_prediction = prediction(x_train, theta)
    train_accuracy = accuracy_score_numpy(y_train, train_prediction)
    test_prediction = prediction(x_test, theta)
    test_accuracy = accuracy_score_numpy(y_test, test_prediction)
    return(theta, train_accuracy, test_accuracy)


# for epoch in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index:random_index+1]
#         yi = y[random_index:random_index+1]
#         gradients = 2 * xi.T @ (xi @ theta - yi)
#         eta = learning_schedule(epoch * m + i)
#         theta = theta - eta * gradients


if __name__ == '__main__':
    path_to_data=os.path.expanduser('~')+'/OneDrive/Dokumenter/FYS4155/Project_2/isingMC/'
    np.random.seed(2)
    file_name = "Ising2DFM_reSample_L40_T=All.pkl"
    data = pickle.load(open(path_to_data+file_name,'rb'))
    data = np.unpackbits(data).reshape(-1, 1600)
    data=data.astype('int')
    data[np.where(data==0)]=-1
    
    file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"
    labels = pickle.load(open(path_to_data+file_name,'rb'))
    
    X_ordered=data[:70000,:]
    Y_ordered=labels[:70000]
    
    X_critical = data[70000:100000,:]
    Y_critical = labels[70000:100000]
    
    X_disordered=data[100000:,:]
    Y_disordered=labels[100000:]
    
    del data,labels
    
    X = np.concatenate((X_ordered,X_disordered))
    Y = np.concatenate((Y_ordered,Y_disordered))

    LEARNING_RATES = np.geomspace(1e-10, 1e0, 11)
    TRAIN_ACCURACY = np.zeros(11)
    TEST_ACCURACY = np.zeros(11)
    CRITICAL_ACCURACY = np.zeros(11)

    THETA1, TRAIN_ACCURACY1, TEST_ACCURACY1 = logistic_regression_SGDmb(X, Y, 25, 1e-4)
    CRITICAL_PREDICTION1 = prediction(X_critical, THETA1)
    CRITICAL_ACCURACY1 = accuracy_score_numpy(Y_critical, CRITICAL_PREDICTION1)
    print(TRAIN_ACCURACY1)
    print(TEST_ACCURACY1)
    print(CRITICAL_ACCURACY1)
    @jit
    def numb_loop():
        for i in range(11):
            THETA, TRAIN_ACCURACY[i], TEST_ACCURACY[i] = logistic_regression_SGDmb(X, Y, 100, LEARNING_RATES[i])
            CRITICAL_PREDICTION = prediction(X_critical, THETA)
            CRITICAL_ACCURACY[i] = accuracy_score_numpy(Y_critical, CRITICAL_PREDICTION)
            print(i)
        plt.xscale('log')
        plt.plot(LEARNING_RATES, TRAIN_ACCURACY, 'o', label='Train')
        plt.plot(LEARNING_RATES, TEST_ACCURACY, 's', label='Test')
        plt.plot(LEARNING_RATES, CRITICAL_ACCURACY, 'o', label='Critical')
        plt.xlabel('Læringsfrekvens')
        plt.ylabel('Nøyaktighet')
        plt.legend()
        plt.show()
    # numb_loop()
    # TEST_ACCURACY = accuracy_score_numpy(X_TEST, Y_TEST)
    # CRITICAL_ACCURACY = accuracy_score_numpy(X_critical, Y_critical)

