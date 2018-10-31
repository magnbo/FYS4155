'''Python program for løsning av første oppgave i prosjekt 1 i FYS-STK4155'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import r2_score

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
Z1 = franke_function(X1, Y1) # + 0.1 * np.random.randn(20, 20)
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

def linear_regresjon(xb_0, y_0):
    '''Lager en lineær modell med prestasjonmålingene r-squared, mean squared error og relativ
      feil til beta.
    '''
    beta = (np.linalg.inv(xb_0.T @ xb_0).dot(xb_0.T).dot(y_0))
    zpredict = xb_0 @ beta
    m, n = xb_0.shape
    variance = 0
    for i in range(400):
        variance += (1/(400-n-1)) * (Z2[i] - zpredict[i])**2
    covariance_matrix = np.linalg.inv(xb_0.T @ xb_0) * variance
    mse = 0.
    for i in range(400):
        mse += (1/400) * (Z2[i] - zpredict[i])**2
    u = 0.
    b = 0.
    zmean = np.mean(Z1)
    for i in range(400):
        u += (Z2[i] - zpredict[i])**2
        b += (Z2[i] - zmean)**2
    rr = 1 - u/b
    relative_error_beta = np.linalg.norm(((np.sqrt(np.diag(covariance_matrix))))/beta)
    return zpredict, mse, rr, relative_error_beta

def linear_bootstrap(xb_0, y_0):
    '''Bootstrap re sampling med mean squared error, r squared og bias.'''
    m, n = xb_0.shape
    rr_boot = np.zeros(50)
    mse_boot = np.zeros(50)
    zpredict_boot = np.zeros((50, 400))
    for i in range(50):
        xbnew_boot_train, z_boot_train = resample(xb_0, y_0, n_samples=280)
        oob_xb = np.array([x for x in xb_0 if x.tolist() not in xbnew_boot_train.tolist()])
        oob_z = np.array([x for x in y_0 if x not in z_boot_train])
        xbnew_boot_test, z_boot_test = resample(oob_xb, oob_z, n_samples=120)
        beta_boot_train = (np.linalg.inv(xbnew_boot_train.T @ xbnew_boot_train)
                           .dot(xbnew_boot_train.T).dot(z_boot_train))
        zpredict_boot_test = xbnew_boot_test @ beta_boot_train
        zpredict_boot[i,:] = xb_0 @ beta_boot_train
        for j in range(120):
            mse_boot[i] += (1/120) * (z_boot_test[j] - zpredict_boot_test[j])**2
        rr_boot[i] = r2_score(z_boot_test, zpredict_boot_test)
    bias = np.mean((Z2 - np.mean(zpredict_boot, axis=0))**2)
    mse_boot_mean = np.mean(mse_boot)
    mse_boot_std = (np.std(mse_boot))/np.sqrt(50)
    varians = np.mean(np.var(zpredict_boot, axis=0))
    rr_boot_mean = np.mean(rr_boot)
    rr_boot_std = (np.std(rr_boot))/np.sqrt(50)
    return (mse_boot_mean, mse_boot_std, rr_boot_mean, rr_boot_std, bias, varians)



def figur_plot(xb_0, y_0):
    '''For visualisering av modeller'''
    zpredict = linear_regresjon(xb_0, y_0)[0]
    zpredict2 = np.reshape(zpredict, (20, 20))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X1, Y1, zpredict2, cmap=cm.viridis, linewidth=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30., azim=60)
    plt.show()


def ridge(xb, lam_0):
    m, n = xb.shape
    beta = (np.linalg.inv(xb.T @ xb + lam_0 * np.identity(n)).dot(xb.T).dot(Z2))
    xbnew = xb
    zpredict = xbnew @ beta
    zpredict2 = np.reshape(zpredict, (20, 20))
    variance = 0
    for i in range(400):
        variance += (1/(400-n-1)) * (Z2[i] - zpredict[i])**2
    covariance_matrix = np.linalg.inv(xb.T @ xb) * variance
    mse = 0.
    for i in range(400):
        mse += (1/400) * (Z2[i] - zpredict[i])**2
    u = 0.
    b = 0.
    zmean = np.mean(Z1)
    for i in range(400):
        u += (Z2[i] - zpredict[i])**2
        b += (Z2[i] - zmean)**2
    rr = 1 - u/b
    rr_boot = np.zeros(50)
    mse_boot = np.zeros(50)
    zpredict_boot = np.zeros((50,400))
    for i in range(50):
        xbnew_boot_train, z_boot_train = resample(xbnew, Z2, n_samples=280)
        oob_xb = np.array([x for x in xbnew if x.tolist() not in xbnew_boot_train.tolist()])
        oob_z = np.array([x for x in Z2 if x not in z_boot_train])
        xbnew_boot_test, z_boot_test = resample(oob_xb ,oob_z , n_samples=120)
        beta_boot_train = (np.linalg.inv(xbnew_boot_train.T @ xbnew_boot_train + lam_0 *
                                         np.identity(n)).dot(xbnew_boot_train.T).dot(z_boot_train))
        zpredict_boot_test = xbnew_boot_test @ beta_boot_train
        zpredict_boot[i,:] = xbnew @ beta_boot_train
        for j in range(120):
            mse_boot[i] += (1/120) * (z_boot_test[j] - zpredict_boot_test[j])**2
        rr_boot[i] = r2_score(z_boot_test, zpredict_boot_test)
    bias = np.mean((Z2 - np.mean(zpredict_boot, axis=0))**2)
    mse_boot_mean = np.mean(mse_boot)
    mse_boot_std = (np.std(mse_boot))/np.sqrt(50)
    rr_boot_mean = np.mean(rr_boot)
    rr_boot_std = (np.std(rr_boot))/np.sqrt(50)
    varians = np.mean(np.var(zpredict_boot, axis=0))
    relative_error_beta =  np.linalg.norm(((np.sqrt(np.diag(covariance_matrix))))/beta)
    return (beta, relative_error_beta, mse, rr, mse_boot_std, rr_boot_mean,
            rr_boot_std, zpredict2,mse_boot_mean , bias, varians)


# figur_plot(XB3, Z2)


# BETA, RELATIVE_ERROR_BETA, MSE, RR, MSE_BOOT_STD, RR_BOOT_MEAN, RR_BOOT_STD, ZPREDICT2, MSE_BOOT_MEAN, BIAS, VARIANS = ridge(XB5, 0)
# print('\( 10^{-6}\). & %5.3f & %6.4f & %5.3f & %6.4f \(\pm\) %6.4f & %5.3f \(\pm\) %5.3f & %5.3f & %6.4f \\\\' % (RELATIVE_ERROR_BETA, MSE, RR, MSE_BOOT_MEAN, MSE_BOOT_STD, RR_BOOT_MEAN, RR_BOOT_STD, BIAS, VARIANS))
BIAS = np.zeros(5)
MSE = np.zeros(5)
VARIANS = np.zeros(5)
BIAS[0], MSE[0], VARIANS[0] = ridge(XB1, 1e-5)[8:]
BIAS[1], MSE[1], VARIANS[1] = ridge(XB2, 1e-5)[8:]
BIAS[2], MSE[2], VARIANS[2] = ridge(XB3, 1e-5)[8:]
BIAS[3], MSE[3], VARIANS[3] = ridge(XB4, 1e-5)[8:]
BIAS[4], MSE[4], VARIANS[4] = ridge(XB5, 1e-5)[8:]

plt.plot(BIAS,'.-', label='Bias')
plt.plot(MSE,'.-', label='MSE')
plt.plot(VARIANS,'.-', label='Varians')
plt.plot(BIAS + VARIANS,'.-', label='Bias + Varians')
plt.xlabel('Polynom grad')
plt.legend()
plt.show()
# FIG = plt.figure()
# AX = FIG.gca(projection='3d')
# SURF = AX.plot_surface(X1, Y1, Z1, cmap=cm.viridis, linewidth=0)
# FIG = plt.figure()
# AX = FIG.gca(projection='3d')
# AX.plot_surface(X1, Y1, ZPREDICT2, cmap=cm.viridis, linewidth=0)
# AX.plot_surface(X1, Y1, Z1, cmap=cm.viridis, linewidth=0)
# AX.set_xlabel('X')
# AX.set_ylabel('Y')
# AX.set_zlabel('Z')
# AX.view_init(elev=30., azim=60)
# plt.show()
