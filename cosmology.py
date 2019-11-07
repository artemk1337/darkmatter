import pandas as pd
import numpy as np
import scipy as sc
import opt
from scipy import integrate
import matplotlib.pyplot as plt
import json


data = np.loadtxt('jla_mub.txt')
d_mu = data[:, 1].astype('float64')
d_z = data[:, 0].astype('float64')
x0 = np.array([0.5, 50])


def j(x):
    def integrand1(z_):
        return (-1 + (1 + z_) ** 3) /\
               (2 * (x[0] + (1 - x[0]) * (1 + z_) ** 3) ** (3 / 2))

    def integrand2(z_):
        return 1 /\
               (x[0] + (1 - x[0]) * (1 + z_) ** 3) ** (1 / 2)
    arr_j = np.asarray(data)
    for i in range(arr_j.shape[0]):
        arr_j[i, 0] = 5 * integrate.quad(integrand1, 0, d_z[i])[0] /\
                      integrate.quad(integrand2, 0, d_z[i])[0] /\
                      np.log(10)
        arr_j[i, 1] = -5 / x[1] / np.log(10)
    return arr_j


def mu_z(x):
    def integrand(z_, i):
        return 1 /\
               (x[0] + (1 - x[0]) * (1 + z_) ** 3) ** (1 / 2)
    arr_mu = np.empty(len(d_mu), dtype=float)
    for i in range(len(arr_mu)):
        arr_mu[i] = 5 * np.log10(integrate.quad(integrand, 0, d_z[i], args=i)[0] * 3 * 10 ** 11 * (1 + d_z[i]) /
                                 x[1]) - 5
    return arr_mu


def plot_mu_z():
    # <==========LM==========> #
    lm = opt.lm(d_mu, mu_z, j, x0)
    x_lm = lm.x
    plt.plot(d_z, mu_z(x_lm), color='r')
    plt.plot(d_z, d_mu, 'o', color='dodgerblue')
    plt.title('Левенберг_макраврдт')
    plt.xlabel('z - красное смещение')
    plt.ylabel('mu - модуль расстояния')
    plt.text(0.6, 34, f'H0 = {round(x_lm[1], 1)}, Omega = {round(x_lm[0], 2)}')
    plt.legend(['optimized',
                'data'])
    try:
        plt.pause(8)
        plt.close()
    except Exception:
        pass
    plt.show()
    # <==========GN==========> #
    gauss = opt.gauss_newton(d_mu, mu_z, j, x0)
    x_gauss = gauss.x
    plt.plot(d_z, mu_z(x_gauss), color='r')
    plt.plot(d_z, d_mu, 'o', color='dodgerblue')
    plt.title('Гаусс-Ньютон')
    plt.xlabel('z - красное смещение')
    plt.ylabel('mu - модуль расстояния')
    plt.text(0.6, 34, f'H0 = {round(x_gauss[1], 1)}, Omega = {round(x_gauss[0], 2)}')
    plt.legend(['optimized',
                'data'])
    plt.savefig('mu-z.png')
    try:
        plt.pause(8)
        plt.close()
    except Exception:
        pass
    plt.show()
    return x_gauss, gauss, x_lm, lm


def save(x_gauss, gauss, x_lm, lm):
    with open('parametrs.json', 'w') as file:
        json.dump({"Gauss-Newton": {"H0": x_gauss[1],
                                    "Omega": x_gauss[0],
                                    "nfev": gauss.nfev},
                   "Levenberg-Marquardt": {"H0": x_lm[1],
                                           "Omega": x_lm[0],
                                           "nfev": lm.nfev}},
                  file, indent=4, separators=(',', ': '))


def plot_cost(gauss, lm):
    plt.semilogy(np.arange(len(lm.cost)), lm.cost, '-.')
    plt.semilogy(np.arange(len(gauss.cost)), gauss.cost, '--')
    plt.title('cost')
    plt.xlabel('итерационный шаг ')
    plt.ylabel('cost')
    plt.legend(['gauss',
                'lm'])
    plt.savefig('cost.png')
    try:
        plt.pause(8)
        plt.close()
    except Exception:
        pass
    plt.show()


def main():
    x_gauss, gauss, x_lm, lm = plot_mu_z()
    save(x_gauss, gauss, x_lm, lm)
    plot_cost(gauss, lm)


main()

