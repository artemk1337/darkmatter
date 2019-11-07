#!/usr/bin/env python3
import pandas as pd
import numpy as np
from collections import namedtuple


Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов можельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива меньше nfev
gradnorm : float
    Норма градиента на финальном итерационном шаге
x : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
"""


def gauss_newton(y, f, j, x0, k=1, atol=1e-4):
    x, i = np.asarray(x0, dtype='float64'), 0
    cost = []
    while 1:
        i += 1
        res = y - f(x)
        cost.append(0.5 * np.dot(res, res))
        jac = j(x)
        g = np.dot(j(x).T, res)
        delta_x = np.linalg.solve(np.dot(jac.T, jac), g)
        x += k * delta_x
        if np.linalg.norm(g) <= atol:
            break
    cost = np.array(cost)
    return Result(nfev=i, cost=cost, gradnorm=np.linalg.norm(g), x=x)


def lm(y, f, j, x0, lmbd0=1e-2, nu=2, tol=1e-4):
    def delta(x, jac, lmbd_new):
        grad = jac.T @ (y - f(x))
        return np.linalg.solve(jac.T @ jac + lmbd_new * np.identity(np.shape(grad)[0]), grad)

    def f_res(x):
        return 0.5 * (y - f(x)) @ (y - f(x))
    nfev, gradnorm, lmbd_new, cost, x = 0, 0, lmbd0, [], x0.copy()
    jac = j(x)
    while np.linalg.norm(delta(x, jac, lmbd_new)) > tol:
        nfev += 1
        cost.append(f_res(x))
        gradnorm = np.linalg.norm(jac.T @ (y - f(x)))
        f1_res, f2_res = f_res(x + delta(x, j(x), lmbd_new)), f_res(x + delta(x, j(x), lmbd_new / nu))
        if f2_res <= f_res(x):
            lmbd_new /= nu
        while (f1_res > f_res(x)) and\
                (f1_res >= f_res(x)) and\
                (np.linalg.norm(delta(x, jac, lmbd_new)) > tol):
            lmbd_new *= nu
        x += delta(x, j(x), lmbd_new)
    return Result(nfev, cost, gradnorm, x)
