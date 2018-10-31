import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split


def lasso(xb, lam_0):
    '''
    '''
    m, n = xb.shape
    reg = linear_model.Lasso(alpha = lam_0, max_iter=1e5, fit_intercept=False)
    model = reg.fit(xb, Z2N)
    beta = np.c_[model.coef_].flatten()
    xbnew = xb
    zpredict = xbnew @ beta
    mse = 0
    for i in range(400):
        mse += (1/400) * (Z2[i] - zpredict[i])**2
    mse_boot = np.zeros(20)
    zpredict_boot = np.zeros((20, 400))
    for i in range(20):
        x_train, x_test, z_train, z_test, z_train_real, z_test_real = train_test_split(xb, Z2N, Z2, test_size=0.3)
        xbnew_boot_train, z_boot_train = resample(x_train, z_train, n_samples=280)
        xbnew_boot_test, z_boot_test, z_test_real = resample(x_test, z_test, z_test_real, n_samples=120)
        reg = linear_model.Lasso(alpha = lam_0, max_iter=1e5, fit_intercept=False)
        model = reg.fit(xbnew_boot_train, z_boot_train)
        beta_boot_train = np.c_[model.coef_].flatten()
        zpredict_boot_test = xbnew_boot_test @ beta_boot_train
        zpredict_boot[i, :] = xbnew @ beta_boot_train
        for j in range(120):
            mse_boot[i] += (1/120) * (z_test_real[j] - zpredict_boot_test[j])**2
    bias = np.mean((Z2 - np.mean(zpredict_boot, axis=0))**2)
    mse_boot_mean = np.mean(mse_boot)
    mse_boot_upper = np.percentile(mse_boot, 97.5)
    mse_boot_lower = np.percentile(mse_boot, 2.5)
    varians = np.mean(np.var(zpredict_boot, axis=0))
    return (mse, mse_boot_mean, mse_boot_upper, mse_boot_lower, bias, varians)

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

    LAM = np.geomspace(1e-7, 1e-5, num=10)
    MSE = np.zeros(10)
    MSE_BOOT = np.zeros(10)
    UPPER = np.zeros(10)
    LOWER = np.zeros(10)
    BIAS = np.zeros(10)
    VARIANS = np.zeros(10)

    for i in range(10):
            MSE[i], MSE_BOOT[i], UPPER[i], LOWER[i], BIAS[i], VARIANS[i] = lasso(XB5, LAM[i])
    plt.xscale('log')
    plt.yscale('log')
    # plt.errorbar(LAM , MSE_BOOT, yerr=[LOWER, UPPER], fmt='o', label='MSE boot')
    plt.plot(LAM, MSE, 's',label='MSE')
    plt.plot(LAM, BIAS, 'v', label='Bias')
    plt.plot(LAM, VARIANS, 'D', label='Varians')
    plt.xlabel('Lambda')
    plt.legend()
    plt.show()



# reg = linear_model.Lasso(alpha = 0.0001, max_iter=1e5, fit_intercept=False)
# model = reg.fit(XB5, Z2)
# beta = np.c_[model.coef_].flatten()
# print(model.coef_)
# zpredict = XB5 @ np.c_[model.coef_]
