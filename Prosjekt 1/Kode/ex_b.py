import numpy as np
from sklearn.utils import resample
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def ridge(xb, lam_0):
    '''
    '''
    m, n = xb.shape
    beta = (np.linalg.inv(xb.T @ xb + lam_0 * np.identity(n)).dot(xb.T).dot(Z2N))
    xbnew = xb
    zpredict = xbnew @ beta
    mse = 0
    for i in range(400):
        mse += (1/400) * (Z2[i] - zpredict[i])**2
    mse_boot = np.zeros(50)
    zpredict_boot = np.zeros((50, 400))
    beta_boot_train = np.zeros((50, 21))
    for i in range(50):
        xbnew_boot_train, z_boot_train = resample(xbnew, Z2, n_samples=280)
        oob_xb = np.array([x for x in xbnew if x.tolist() not in xbnew_boot_train.tolist()])
        oob_z = np.array([x for x in Z2 if x not in z_boot_train])
        xbnew_boot_test, z_boot_test = resample(oob_xb , oob_z , n_samples=120)
        beta_boot_train[i, :] = (np.linalg.inv(xbnew_boot_train.T @ xbnew_boot_train + lam_0 *
                                         np.identity(n)).dot(xbnew_boot_train.T).dot(z_boot_train))
        zpredict_boot_test = xbnew_boot_test @ beta_boot_train[i, :]
        zpredict_boot[i,:] = xbnew @ beta_boot_train[i, :]
        for j in range(120):
            mse_boot[i] += (1/120) * (z_boot_test[j] - zpredict_boot_test[j])**2
    bias = np.mean((Z2 - np.mean(zpredict_boot, axis=0))**2)
    mse_boot_mean = np.mean(mse_boot)
    mse_boot_upper = np.percentile(mse_boot, 97.5)
    mse_boot_lower = np.percentile(mse_boot, 2.5)
    varians = np.mean(np.var(zpredict_boot, axis=0))
    beta = np.mean(beta_boot_train, axis=0)
    beta_boot_upper = np.percentile(beta_boot_train, 97.5, axis=0)
    beta_boot_lower = np.percentile(beta_boot_train, 2.5, axis=0)
    return (beta, beta_boot_upper, beta_boot_lower)

if __name__ == '__main__':
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
    Z1 = franke_function(X1, Y1)
    Z1N = franke_function(X1, Y1) + 0.1 * np.random.randn(20, 20)
    Z2 = Z1.flatten()
    Z2N = Z1N.flatten()
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

    # LAM = np.geomspace(1e-6, 1e-3, num=20)
    # MSE = np.zeros(20)
    # MSE_BOOT = np.zeros(20)
    # UPPER = np.zeros(20)
    # LOWER = np.zeros(20)
    # BIAS = np.zeros(20)
    # VARIANS = np.zeros(20)
    beta, beta_boot_upper, beta_boot_lower = ridge(XB5, 1e-5)
    np.savetxt('beta.txt', beta, delimiter=',')
    np.savetxt('beta_boot_upper.txt', beta_boot_upper, delimiter=',')
    np.savetxt('beta_boot_lower.txt', beta_boot_lower, delimiter=',')
    # for i in range(20):
    #     MSE[i], MSE_BOOT[i], UPPER[i], LOWER[i], BIAS[i], VARIANS[i] = ridge(XB3, LAM[i])
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.errorbar(LAM , MSE_BOOT, yerr=[LOWER, UPPER], fmt='o', label='MSE boot')
    # plt.plot(LAM, MSE, 's',label='MSE')
    # plt.plot(LAM, BIAS, 'v', label='Bias')
    # plt.plot(LAM, VARIANS, 'D', label='Varians')
    # plt.xlabel('Lambda')
    # plt.legend()
    # plt.show()
