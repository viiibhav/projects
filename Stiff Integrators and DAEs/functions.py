# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:49:49 2020

@author: vdabadgh
"""

import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set_style('whitegrid')


# Explicit Euler
def exp_euler(y, dt, func):
    return y + dt * func(y)


# Implicit Euler
def imp_euler(yk, dt, func):
    def imp_euler_expr(y):
        return y - yk - dt * func(y)
    
    sol = root(imp_euler_expr, yk)
    
    return sol.x


# RK4
def RK4(y, dt, func):
    f1 = func(y)
    f2 = func(y + (dt / 2) * f1)
    f3 = func(y + (dt / 2) * f2)
    f4 = func(y + dt * f3 )
    return y + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4)


# Backward Difference Formulae (BDF)
def bdf(order, yk, dt, func):
    def bdf_expr(y, order):
        if order == 2:
            y0, y1 = yk
            expr = y - 4 / 3 * y1 + 1 / 3 * y0 - 2 / 3 * dt * func(y)
        
        elif order == 3:
            y0, y1, y2 = yk
            expr = 0
            expr += y
            expr -= 18 / 11 * y2
            expr += 9 / 11 * y1
            expr -= 2 / 11 * y0
            expr -= 6 / 11 * dt * func(y)
        
        elif order == 6:
            y0, y1, y2, y3, y4, y5 = yk
            expr = 0
            expr += y
            expr -= 360 / 147 * y5
            expr += 450 / 147 * y4
            expr -= 400 / 147 * y3
            expr += 225 / 147 * y2
            expr -= 72 / 147 * y1
            expr += 10 / 147 * y0
            expr -= 60 / 147 * dt * func(y)

        return expr
    
    sol = root(bdf_expr, yk[-1], args=(order))
    
    return sol.x


# Two CSTRs in series (ODE)
def cstr(cA):
    k = [0.05, 38]  # min**(-1)
    V = [500, 20]   # ft**3
    F = 100         # ft**3/min
    cA_in = 2       # mol/ft**3

    dcA_tank1 = F / V[0] * (cA_in - cA[0]) - k[0] * cA[0]
    dcA_tank2 = F / V[1] * (cA[0] - cA[1]) - k[1] * cA[1]

    return np.array([dcA_tank1, dcA_tank2])


# Exact solution to CSTR system
def exact_cstr(t):
    cA1 = 1.6 - 0.4 * np.exp(-0.25 * t)
    cA2 = 0.186 - 0.04678 * np.exp(-0.25 * t) + 2.4e-4 * np.exp(-43 * t)    
    return np.array([cA1, cA2])


# Robertson reaction system
def robertson(X):
    x, y, z = X
    dx = -0.04 * x + 1e+04 * y * z
    dy = 0.04 * x - 1e+04 * y * z - 3e+07 * y**2
    dz = 3e+07 * y**2
    return np.array([dx, dy, dz])


# y' = Ay
def stiff(y):
    # A = np.array([[-1501, -1499],
    #               [-1499, -1501]])
    A = np.array([[-0.1, -49.9, 0],
                  [0, -50, 0],
                  [0, 70, -120]])
    return A @ y


def dae_index2(z, t):
    x, y = z
    alg = x - np.sin(t)
    dxdt = -y
    return [dxdt, alg]


def dae_index2_exact(t):
    x = np.sin(t)
    y = -np.cos(t)
    return np.array([x, y])


def dae_index3(X, t):
    x, y, z = X
    dxdt = y
    dydt = z
    alg = x - np.sin(t)
    return [dxdt, dydt, alg]


def dae_index3_exact(t):
    return np.array([np.sin(t), np.cos(t), -np.sin(t)])


# Stability plots
def plot(xspan, mu, ax_limits):
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_xlim(ax_limits)
    ax.set_ylim(ax_limits)
    
    # plot axes
    ax.plot([0, 0], ax_limits, 'k')
    ax.plot(ax_limits, [0, 0], 'k')
    
    # plot stability region
    # white = 1 / 256 * np.array([255, 255, 255, 0.9])
    # green = 1 / 256 * np.array([38, 194, 129, 1])
    # newcolors = np.vstack((white, green))
    # newcolors = np.empty([100, 4])
    # newcolors[10:, :] = green
    # cmap = ListedColormap(newcolors)
    X, Y = np.meshgrid(xspan, xspan)
    im = ax.contourf(Y, X, mu, [0, 1], colors=[[1, 0.5, 0.8]])
    # fig.colorbar(im, ax=ax)


