import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split



# Området Franke funksjonen blir sett over
X1 = np.arange(0, 1, 0.05)
Y1 = np.arange(0, 1, 0.05)
X1, Y1 = np.meshgrid(X1, Y1)

# Franke funksjonen som ble lagt ut med oppgaven
def franke_function(x_0, y_0):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x_0 - 2)**2) - 0.25 * ((9 * y_0 - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x_0 + 1)**2) / 49.0 - 0.1 * (9 * y_0 + 1))
    term3 = 0.5 * np.exp(-(9 * x_0 - 7)**2 / 4.0 - 0.25 * ((9 * y_0 - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x_0 - 4)**2 - (9 * y_0 - 7)**2)
    return term1 + term2 + term3 + term4

# Forberedelser av konstanter 
np.random.seed(2)
Z1 = franke_function(X1, Y1)  + 0.1 * np.random.randn(20, 20)
Z2 = Z1.flatten()
X2 = X1.flatten()
Y2 = Y1.flatten()
C = np.ones((400, 1))
XB1 = np.c_[C, X2, Y2]
XB2 = np.c_[C, X2, Y2, X2**2, Y2**2, X2 * Y2]
XB3 = np.c_[C, X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2]
XB4 = np.c_[C, X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2, X2**4, X2**3 * Y2,
            X2**2 * Y2**2, X2 * Y2**3, Y2**4]
XB5 = np.c_[C, X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2, X2**4, X2**3 * Y2,
            X2**2 * Y2**2, X2 * Y2**3, Y2**4, X2**5, X2**4 * Y2, X2**3 * Y2**2, X2**2 * Y2**3
            , X2 * Y2**4, Y2**5]
def bias_var(xb_0, lam_0):
    '''Bootstrap som tar ut 20 % av variablene til å bruke som et konstant test set for å
     sammenligne varians, MSE og bias'''
    xb_train, xb_test, z_train, z_test = train_test_split(xb_0, Z2, test_size=0.2)
    m, n = xb_0.shape
    mse_boot = np.zeros(50)
    zpredict_boot = np.zeros((50, 80))
    for i in range(50):
        xbnew_boot_train, z_boot_train = resample(xb_train, z_train, n_samples=224)
        oob_xb = np.array([x for x in xb_train if x.tolist() not in xbnew_boot_train.tolist()])
        oob_z = np.array([x for x in z_train if x not in z_boot_train])
        xbnew_boot_test, z_boot_test = resample(oob_xb, oob_z, n_samples=96)
        beta_boot_train = (np.linalg.inv(xbnew_boot_train.T @ xbnew_boot_train + lam_0 *
                                         np.identity(n)).dot(xbnew_boot_train.T).dot(z_boot_train))
        zpredict_boot[i, :] = xb_test @ beta_boot_train
        for j in range(80):
            mse_boot[i] += (1/80) * (z_test[j] - zpredict_boot[i, j])**2
    bias = np.mean((z_test - np.mean(zpredict_boot, axis=0, keepdims=True))**2)
    mse_boot_mean = np.mean(mse_boot)
    varians = np.mean(np.var(zpredict_boot, axis=0))
    return bias, mse_boot_mean, varians

BIAS = np.zeros(5)
MSE = np.zeros(5)
VARIANS = np.zeros(5)
BIAS[0], MSE[0], VARIANS[0] = bias_var(XB1, 0)
BIAS[1], MSE[1], VARIANS[1] = bias_var(XB2, 0)
BIAS[2], MSE[2], VARIANS[2] = bias_var(XB3, 0)
BIAS[3], MSE[3], VARIANS[3] = bias_var(XB4, 0)
BIAS[4], MSE[4], VARIANS[4] = bias_var(XB5, 0)
CHECK = np.zeros(5)

for i in range(5):
    if MSE[i] >= (BIAS[i] + VARIANS[i]):
        CHECK[i] = 1
    else:
        CHECK[i] = 0

print(CHECK)

for i in range(5):
    print(' %10.8f & %10.8f & %10.18f & %10.18f \\\\' % (BIAS[i], VARIANS[i], MSE[i],
                                                       BIAS[i] + VARIANS[i]))
