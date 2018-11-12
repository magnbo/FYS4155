import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random
import pickle,os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class NeuralNetwork:
    def __init__(
        self,
        X_data,
        Y_data,
        n_hidden_neurons=50,
        n_categories=10,
        epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0,

    ):
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = sigmoid(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias
        a_h = sigmoid(z_h)

        z_o = np.matmul(a_h, self.output_weights) + self.output_bias
        
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

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
    train_size = int(np.ceil(0.8 * X.shape[0]))
    test_size = int(np.ceil(0.2 * X.shape[0]))
    epochs = 100
    batch_size = 100
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size)
    n_inputs, n_features = X_train.shape


    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train.astype(int)), to_categorical_numpy(Y_test.astype(int))
    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    critical_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            dnn = NeuralNetwork(X_train, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=25, batch_size=32,
                                n_hidden_neurons=20, n_categories=2)
            dnn.train()
            
            DNN_numpy[i][j] = dnn
            
            test_predict = dnn.predict(X_test)
            test_accuracy[i][j] = accuracy_score(Y_test, test_predict)
            train_predict = dnn.predict(X_train)
            train_accuracy[i][j] = accuracy_score(Y_train, train_predict)
            critical_predict = dnn.predict(X_critical)
            critical_accuracy[i][j] = accuracy_score(Y_critical, critical_predict)

    np.savetxt('test_accuracy.txt', test_accuracy)
    np.savetxt('train_accuracy.txt', train_accuracy)
    np.savetxt('critical_accuracy.txt', critical_accuracy)

    ax = plt.figure()
    sns.heatmap(train_accuracy, annot=True, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()
    
    ax = plt.figure()
    sns.heatmap(test_accuracy, annot=True, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    ax = plt.figure()
    sns.heatmap(critical_accuracy, annot=True, cmap="viridis")
    ax.set_title("Critical Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()


