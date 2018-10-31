import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split


def ridge(xb, lam_0):
    m, n = xb.shape
    beta = (np.linalg.inv(xb.T @ xb + lam_0 * np.identity(n)).dot(xb.T).dot(Z2))
    xbnew = xb
    zpredict = xbnew @ beta
    mse = 0.
    for i in range(num_data):
        mse += (1/num_data) * (Z2[i] - zpredict[i])**2
    mse_boot = np.zeros(20)
    zpredict_boot = np.zeros((20, 5000))
    beta_boot_train = np.zeros((20, 55))
    for i in range(20):
        x_train, x_test, z_train, z_test = train_test_split(xb, Z2, test_size=0.3)
        xbnew_boot_train, z_boot_train = resample(x_train, z_train, n_samples=3500)
        xbnew_boot_test, z_boot_test = resample(x_test, z_test, n_samples=1500)
        beta_boot_train[i, :] = (np.linalg.inv(xbnew_boot_train.T @ xbnew_boot_train + lam_0 * np.identity(n)).dot(xbnew_boot_train.T).dot(z_boot_train))
        zpredict_boot_test = xbnew_boot_test @ beta_boot_train[i, :]
        zpredict_boot[i, :] = xbnew @ beta_boot_train[i, :]
        for j in range(1500):
            mse_boot[i] += (1/1500) * (z_boot_test[j] - zpredict_boot_test[j])**2
    beta = np.mean(beta_boot_train, axis=0)
    beta_boot_upper = np.percentile(beta_boot_train, 97.5, axis=0)
    beta_boot_lower = np.percentile(beta_boot_train, 2.5, axis=0)
    return (beta, beta_boot_upper, beta_boot_lower)


if __name__ == '__main__':
    X = np.loadtxt('X.txt', delimiter=',').flatten()
    y = np.loadtxt('Y.txt', delimiter=',').flatten()
    z = np.loadtxt('Z.txt', delimiter=',').flatten()
    Z2 = z
    Y2 = y
    X2 = X
    x_ = X
    y_ = y
    num_data = len(X)
    num_train = int(np.ceil(num_data*0.7))
    num_test = int(np.ceil(num_data*0.3))
    XB1 = np.c_[np.ones((num_data,1)), X2, Y2]
    XB2 = np.c_[np.ones((num_data,1)), X2, Y2, X2**2, Y2**2, X2 * Y2]
    XB3 = np.c_[np.ones((num_data,1)), X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2]
    XB4 = np.c_[np.ones((num_data,1)), X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2, X2**4, X2**3 * Y2,
                X2**2 * Y2**2, X2 * Y2**3, Y2**4]
    XB5 = np.c_[np.ones((num_data,1)), X2, X2**2, Y2, Y2**2, X2 * Y2, X2**3, Y2**3, X2*X2*Y2, X2*Y2*Y2, X2**4, X2**3 * Y2,
                X2**2 * Y2**2, X2 * Y2**3, Y2**4, X2**5, X2**4 * Y2, X2**3 * Y2**2, X2**2 * Y2**3
                , X2 * Y2**4, Y2**5]
    XB6 = np.c_[np.ones((num_data,1)), X, y, \
                 X**2, X*y, y**2, \
                 X**3, X**2*y, X*y**2, y**3, \
                 X**4, X**3*y, X**2*y**2, X*y**3,y**4, \
                 X**5, X**4*y, X**3*y**2, X**2*y**3,X*y**4, y**5, \
                 x_**6, x_**5*y_, x_**4*y_**2, x_**3*y_**3,x_**2*y_**4, x_*y_**5, y_**6]
    XB7 = np.c_[np.ones((num_data,1)), X, y, \
                 X**2, X*y, y**2, \
                 X**3, X**2*y, X*y**2, y**3, \
                 X**4, X**3*y, X**2*y**2, X*y**3,y**4, \
                 X**5, X**4*y, X**3*y**2, X**2*y**3,X*y**4, y**5, \
                 x_**6, x_**5*y_, x_**4*y_**2, x_**3*y_**3,x_**2*y_**4, x_*y_**5, y_**6, \
                 x_**7, x_**6*y_, x_**5*y_**2, x_**4*y_**3,x_**3*y_**4, x_**2*y_**5, x_*y_**6, y_**7]
    XB8 = np.c_[np.ones((num_data,1)), X, y, \
                 X**2, X*y, y**2, \
                 X**3, X**2*y, X*y**2, y**3, \
                 X**4, X**3*y, X**2*y**2, X*y**3,y**4, \
                 X**5, X**4*y, X**3*y**2, X**2*y**3,X*y**4, y**5, \
                 x_**6, x_**5*y_, x_**4*y_**2, x_**3*y_**3,x_**2*y_**4, x_*y_**5, y_**6, \
                 x_**7, x_**6*y_, x_**5*y_**2, x_**4*y_**3,x_**3*y_**4, x_**2*y_**5, x_*y_**6, y_**7, \
                 x_**8, x_**7*y_, x_**6*y_**2, x_**5*y_**3,x_**4*y_**4, x_**3*y_**5, x_**2*y_**6, x_*y_**7,y_**8]
    XB9 = np.c_[np.ones((num_data,1)), X, y, \
                 X**2, X*y, y**2, \
                 X**3, X**2*y, X*y**2, y**3, \
                 X**4, X**3*y, X**2*y**2, X*y**3,y**4, \
                 X**5, X**4*y, X**3*y**2, X**2*y**3,X*y**4, y**5, \
                 x_**6, x_**5*y_, x_**4*y_**2, x_**3*y_**3,x_**2*y_**4, x_*y_**5, y_**6, \
                 x_**7, x_**6*y_, x_**5*y_**2, x_**4*y_**3,x_**3*y_**4, x_**2*y_**5, x_*y_**6, y_**7, \
                 x_**8, x_**7*y_, x_**6*y_**2, x_**5*y_**3,x_**4*y_**4, x_**3*y_**5, x_**2*y_**6, x_*y_**7,y_**8, \
                 x_**9, x_**8*y_, x_**7*y_**2, x_**6*y_**3,x_**5*y_**4, x_**4*y_**5, x_**3*y_**6, x_**2*y_**7,x_*y_**8, y_**9]


    # LAM = np.geomspace(1e-2, 1e-1, num=10)
    # MSE = np.zeros(10)
    # MSE_BOOT = np.zeros(10)
    # UPPER = np.zeros(10)
    # LOWER = np.zeros(10)
    # BIAS = np.zeros(10)
    # VARIANS = np.zeros(10)
    # for i in range(10):
    #         MSE[i], MSE_BOOT[i], UPPER[i], LOWER[i], BIAS[i], VARIANS[i] = ridge(XB9, LAM[i])
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.errorbar(LAM, MSE_BOOT, yerr=[LOWER, UPPER], fmt='o', label='MSE boot')
    # plt.plot(LAM, MSE, 's',label='MSE')
    # plt.plot(LAM, BIAS, 'v', label='Bias')
    # plt.plot(LAM, VARIANS, 'D', label='Varians')
    # plt.xlabel('Lambda')
    # plt.legend()
    # plt.show()
    # reg = linear_model.Lasso(alpha = 1e-2, max_iter=1e4, fit_intercept=False)
    # model = reg.fit(XB9, Z2)
    # print(model.coef_)
    beta, beta_boot_upper, beta_boot_lower = ridge(XB9, 3.973e-14)
    np.savetxt('Tbeta.txt', beta, delimiter=',')
    np.savetxt('Tbeta_boot_upper.txt', beta_boot_upper, delimiter=',')
    np.savetxt('Tbeta_boot_lower.txt', beta_boot_lower, delimiter=',')


    # Z2 = np.reshape(Z2, (100, 50))
    # R = np.loadtxt('R.txt', delimiter=',')
    # C = np.loadtxt('C.txt', delimiter=',')
    # FIG = plt.figure()
    # AX = FIG.gca(projection='3d')
    # AX.plot_surface(C, R, Z2, cmap=cm.viridis, linewidth=0)
    # AX.set_xlabel('X')
    # AX.set_ylabel('Y')
    # AX.set_zlabel('Z')
    # plt.show()


