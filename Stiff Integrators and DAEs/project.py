# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:39:47 2020

@author: vdabadgh
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import seaborn as sns
sns.set_style('whitegrid')
import time
from functions import (exp_euler, imp_euler, RK4, bdf, cstr, robertson,
                       stiff, dae_index2, dae_index2_exact,
                       dae_index3, dae_index3_exact, plot)


# =============================================================================
# CSTR system
# =============================================================================
def solve_cstr():
    # Initial conditions for CSTR problem
    cA0 = [1.2, 0.1395]  # mol/ft3
    
    t_end = 10
    dt = 0.01
    npoints = int(t_end / dt + 1)
    t = np.linspace(0, t_end, npoints)
    cA = np.ones([len(t), len(cA0)])
    cA[0] = cA0
    
    for i in range(1, len(t)):
        # Explicit Euler
        cA[i] = exp_euler(cA[i-1], dt, func=cstr)
        
        # Implicit Euler
        cA[i] = imp_euler(cA[i-1], dt, func=cstr)
        
        # BDF
        order = 6
        if i < order:
            cA[i] = imp_euler(cA[i-1], dt, func=cstr)
        else:
            cA[i] = bdf(order, cA[i-order:i], dt, cstr)
    
    fig, axs = plt.subplots(2, 1, sharex=True)
    [ax.plot(t, cA[:, i]) for i, ax in enumerate(axs)]


# =============================================================================
# DAE system (index 2)
# =============================================================================
def solve_dae_index2():
    dt = 0.05
    t_end = 10
    npoints = int(t_end / dt + 1)
    t = np.linspace(0, t_end, npoints)
    # x0 = np.sin(0)
    # y0 = -np.cos(0) - dt
    # z0 = [x0, y0]
    Z0 = np.array([[np.sin(0), -np.cos(0)],
                   [np.sin(0), -np.cos(0) - dt],
                   [dt, -np.cos(0)]])
    fig, axs = plt.subplots(3, 1, sharex=True)#, figsize=(10, 10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for j, ax in enumerate(axs.flat):
        z0 = Z0[j]
        z = np.zeros([len(t), len(z0)])
        z[0] = z0
        for i in range(1, len(t)):
            # Midpoint Euler
            # z[i][0] = np.sin(t[i]) - z[i-1][0] + np.sin(t[i-1])
            # z[i][1] = -2 / dt * (z[i][0] - z[i-1][0]) - z[i-1][1] 
        
            # Implicit Euler (BDF1)
            z[i][0] = np.sin(t[i])
            z[i][1] = -(z[i][0] - z[i-1][0]) / dt
        
        ax.plot(t, z, '-', alpha=1)
        if j == len(axs.flat) - 1:
            ax.set_xlabel('Time (s)')
        ax.set_xlim([0, t_end])
        ax.legend([r'$x$', r'$y$'])
        ax.set_title(r'$x(0) = {}$, $y(0) = {}$'.format(*z0))
    
    fig.suptitle('Solutions to index-2 DAE system using Implicit Euler (BDF1)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig('dae_index2_imp.pdf')
    plt.savefig('dae_index2_imp.png')
    # plt.plot(t, dae_exact(t).T)


# =============================================================================
# DAE system (index 3)
# =============================================================================
def solve_dae_index3():
    dt = 0.05
    t_end = 10
    npoints = int(t_end / dt + 1)
    t = np.linspace(0, t_end, npoints)

    X0_all = np.array([[np.sin(0), np.cos(0), -np.sin(0)],
                       [np.sin(0), np.cos(0) - dt, -np.sin(0)],
                       [np.sin(0) + dt, np.cos(0), -np.sin(0) + dt]])
    fig, axs = plt.subplots(3, 1, sharex=True)#, figsize=(10, 10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for j, ax in enumerate(axs.flat):
        X0 = X0_all[j]
        x, y, z = np.zeros([len(t), len(X0)]).T
        x[0], y[0], z[0] = X0

        for i in range(1, len(t)):
            # Midpoint Euler
            # x[i] = np.sin(t[i]) - x[i-1] + np.sin(t[i-1])
            # y[i] = 2 / dt * (x[i] - x[i-1]) - y[i-1]
            # z[i] = 2 / dt * (y[i] - y[i-1]) - z[i-1]
        
            # Implicit Euler (BDF1)
            # x[i] = np.sin(t[i])
            # y[i] = (x[i] - x[i-1]) / dt
            # z[i] = (y[i] - y[i-1]) / dt
            
            # BDF2
            order = 2
            if i < order:
                x[i] = np.sin(t[i]) - x[i-1] + np.sin(t[i-1])
                y[i] = 2 / dt * (x[i] - x[i-1]) - y[i-1]
                z[i] = 2 / dt * (y[i] - y[i-1]) - z[i-1]
            else:
                x[i] = np.sin(t[i]) + np.sin(t[i-1]) + np.sin(t[i-2]) - x[i-1] - x[i-2]
                y[i] = 1 / dt * (3 / 2 * x[i] - 2 * x[i-1] + 1 / 2 * x[i-2])
                z[i] = 1 / dt * (3 / 2 * y[i] - 2 * y[i-1] + 1 / 2 * y[i-2])
        
            
        ax.plot(t, np.vstack((x, y, z)).T)
        if j == len(axs.flat) - 1:
            ax.set_xlabel('Time (s)')
        ax.set_xlim([0, t_end])
        # ax.set_ylim([-1.1, 1.1])
        ax.legend([r'$x$', r'$y$', r'$z$'])
        ax.set_title(r'$x(0) = {}$, $y(0) = {}$, $z(0) = {}$'.format(*X0))

    fig.suptitle('Solutions to index-3 DAE system using BDF2')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    # plt.savefig('dae_index3_bdf2.pdf')
    # plt.savefig('dae_index3_bdf2.png')


# =============================================================================
# Stiff system (hw3 p7)
# =============================================================================
def solve_stiff():
    y0 = [3, 1.5, 3]
    t_end = 60
    
    algs = ['exp', 'imp', 'rk4', 'bdf2', 'bdf3', 'bdf6']
    y = {alg: 0 for alg in algs}
    
    dtvals = [10**(-n) for n in range(5)]
    npoints = [int(t_end / dt + 1) for dt in dtvals]
    tspans = [np.linspace(0, t_end, n) for n in npoints]
    y_initialize = np.array([np.ones([len(t), len(y0)]) for t in tspans])
    solve_time = np.zeros(len(dtvals))
    
    fig, axs = plt.subplots(3, 2)#, sharex=True, sharey=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for j, dt, n, t, y_init, ax in zip(range(len(dtvals)), dtvals, npoints,
                                       tspans, y_initialize, axs.flat[:-1]):
        y_init[0] = y0
        
        # Explicit Euler
        # y['exp'] = y_init.copy()
        # t1 = time.time()
        # for i in range(1, len(t)):
        #     y['exp'][i] = exp_euler(y['exp'][i-1], dt, func=stiff)
        # t2 = time.time()
        # solve_time[j] = t2 - t1
        
        # Implicit Euler
        # y['imp'] = y_init.copy()
        # t1 = time.time()
        # for i in range(1, len(t)):
        #     y['imp'][i] = imp_euler(y['imp'][i-1], dt, func=stiff)
        # t2 = time.time()
        # solve_time[j] = t2 - t1
    
        # RK4
        # y['rk4'] = y_init.copy()
        # t1 = time.time()
        # for i in range(1, len(t)):
        #     y['rk4'][i] = RK4(y['rk4'][i-1], dt, func=stiff)
        # t2 = time.time()
        # solve_time[j] = t2 - t1
 
        # BDF
        y['bdf2'] = y_init.copy()
        order = 2
        t1 = time.time()
        for i in range(1, len(t)):
            if i < order:
                y['bdf2'][i] = imp_euler(y['bdf2'][i-1], dt, func=stiff)
            else:
                y['bdf2'][i] = bdf(order, y['bdf2'][i-order:i], dt, func=stiff)
        t2 = time.time()
        solve_time[j] = t2 - t1
        
        # ax.plot(t, y['bdf2'])
        ax.semilogx(t, y['bdf2'])
        ax.set_title(f'$\Delta t$ = {dt:.1e}')
        ax.legend(['A', 'B', 'C'], loc='upper right')
        if j >= len(dtvals) / 2:
            ax.set_xlabel('Time (s)')
        if j % 2 == 0:
            ax.set_ylabel('Concentrations (mol/L)')
    
    fig.delaxes(axs.flat[-1])
    fig.suptitle('Profiles of species concentrations using BDF2 (log scaled time axis)')
    # fig.suptitle('Profiles of species concentrations using BDF2')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    # plt.savefig('stiff_bdf2.pdf')
    # plt.savefig('stiff_bdf2.png')
    plt.savefig('stiff_BDF2_log.pdf')
    plt.savefig('stiff_BDF2_log.png')
    
    print(solve_time)


# =============================================================================
# Robertson reaction system
# =============================================================================
def solve_robertson():
    X0 = [1, 0, 0]
    t_end = 10**4
    dt = 1e-02
    npoints = int(t_end / dt + 1)
    t = np.linspace(0, t_end, npoints)
    X = np.ones([len(t), len(X0)])
    X[0] = X0
    
    fig, axs = plt.subplots(3, 1, sharex=True)
    
    # Explicit Euler
    # for i in range(1, len(t)):
    #     X[i] = exp_euler(X[i-1], dt=dt, func=robertson)
    # plt.plot(t, X)
    # plt.legend(['x', 'y', 'z'])
    
    # Implicit Euler
    # for i in range(1, len(t)):
    #     X[i] = imp_euler(X[i-1], dt=dt, func=robertson))
    
    # [ax.semilogx(t, X[:, i]) for i, ax in enumerate(axs)]
    
    # BDF2
    for i in range(1, len(t)):
        order = 2
        if i < order:
            X[i] = imp_euler(X[i-1], dt=dt, func=robertson)
        else:
            X[i] = bdf(order, X[i-order:i], dt=dt, func=robertson)

    [ax.semilogx(t, X[:, i]) for i, ax in enumerate(axs)]
    # plt.plot(t, y)


# =============================================================================
# Stability plots
# =============================================================================
# Explicit Euler, Midpoint Euler (Trapezoidal/IRK2), Adams-Bashforth
def plot_stability():
    w = np.exp(1j * np.linspace(0, 2 * np.pi, 200))
    
    Path = mpath.Path
    methods = {'ExpEuler': 'Explicit Euler',
               'trapezoidal': 'Trapezoidal (Implicit RK2)',
               'AB2': 'Adams-Bashforth Method'}
    for n, name, z in zip(methods.keys(),
                          methods.values(),
                          [w - 1, [5j, -5 + 5j, -5 - 5j, -5j, 5j], 2 * (w**2 - w) / (3 * w - 1)]):
        # fig.clf()
        fig, ax = plt.subplots()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    
        verts = list(map(lambda z: (np.real(z), np.imag(z)), z))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor=[0.8, 0.9, 1], edgecolor='black', alpha=1)
        ax.add_patch(patch)
        ax.plot([-3, 2], [0, 0],'k--', alpha=0.5)
        ax.plot([0, 0], [-2, 2],'k--', alpha=0.5)
        ax.set_xlim(-3, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel(r'Re($\lambda \Delta t$)')
        ax.set_ylabel(r'Im($\lambda \Delta t$)')
        fig.savefig("Stability_region_{}_.png".format(n))
        ax.set_title('Stability Region for {}'.format(name))
        fig.savefig("Stability_region_{}_.pdf".format(n))


# Plot Explicit RK2 and RK4 stability
def plot_stability_explicit_RK():
    # Specify x range and number of points
    x0 = -5
    x1 = 5
    Nx = 501
    # Specify y range and number of points
    y0 = -5
    y1 = 5
    Ny = 501
    # Construct mesh
    xv = np.linspace(x0,x1,Nx)
    yv = np.linspace(y0,y1,Ny)
    [x,y] = np.meshgrid(xv,yv)
    # Calculate z
    z = x + 1j*y
    # 2nd order Runge-Kutta growth factor
    g = 1 + z + 0.5*z**2
    # 4nd order Runge-Kutta growth factor
    g2 = 1 + z + 0.5*z**2 + (1/6)*z**3 + (1/24)*z**4
    # Calculate magnitude of g
    gmag = abs(g)
    g2mag = abs(g2)

    colors = sns.color_palette('Blues', 2)
    
    # Plot contours of gmag
    plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    c1 = plt.contour(x,y, gmag, [1], colors=[colors[0]])
    c2 = plt.contour(x,y, g2mag, [1], colors=[colors[1]])
    c1.collections[0].set_label('RK2')
    c2.collections[0].set_label('RK4')
    
    plt.plot([0, 0], [y0, y1], 'k--', alpha=0.5)
    plt.plot([x0, x1], [0, 0], 'k--', alpha=0.5)
    
    plt.legend()
    plt.xlabel(r'Re($\lambda \Delta t$)')
    plt.ylabel(r'Im($\lambda \Delta t$)')
    plt.savefig('Stability_Exp_RK2_RK4.png')
    plt.title('Stability Regions of Explicit RK Methods')
    plt.savefig('Stability_Exp_RK2_RK4.pdf')
    
    plt.figure()
    plt.contour(x,y, gmag, [1], colors='k')
    plt.contourf(x, y, gmag, [0, 1], colors=[[0.8, 0.9, 1]])
    plt.plot([0, 0], [y0, y1], 'k--', alpha=0.5)
    plt.plot([x0, x1], [0, 0], 'k--', alpha=0.5)
    plt.xlabel(r'Re($\lambda \Delta t$)')
    plt.ylabel(r'Im($\lambda \Delta t$)')
    plt.savefig('Stability_Exp_RK2.png')
    plt.title('Stability Regions of Explicit RK2')
    plt.savefig('Stability_Exp_RK2.pdf')

    plt.figure()
    plt.contour(x,y, g2mag, [1], colors='k')
    plt.contourf(x, y, g2mag, [0, 1], colors=[[0.8, 0.9, 1]])
    plt.plot([0, 0], [y0, y1], 'k--', alpha=0.5)
    plt.plot([x0, x1], [0, 0], 'k--', alpha=0.5)
    plt.xlabel(r'Re($\lambda \Delta t$)')
    plt.ylabel(r'Im($\lambda \Delta t$)')
    plt.savefig('Stability_Exp_RK4.png')
    plt.title('Stability Regions of Explicit RK4')
    plt.savefig('Stability_Exp_RK4.pdf')


BDFcoeffs = { 1: { 'alpha': [1, -1], 'beta': 1},
              2: { 'alpha': [3, -4, 1], 'beta': 2 },
              3: { 'alpha': [11, -18, 9, -2], 'beta': 6 },
              4: { 'alpha': [25, -48, 36, -16, 3], 'beta': 12 },
              5: { 'alpha': [137, -300, 300, -200, 75, -12], 'beta': 60 },
              6: { 'alpha': [147, -360, 450, -400, 225, -72, 10], 'beta': 60 } }

plotWindow = { 1: { 'realPart': [-2, 3], 'imagPart': [-2, 2] },
               2: { 'realPart': [-2, 5], 'imagPart': [-3, 3] },
               3: { 'realPart': [-4, 8], 'imagPart': [-5, 5] },
               4: { 'realPart': [-4, 14], 'imagPart': [-8, 8] },
               5: { 'realPart': [-10, 25], 'imagPart': [-15, 15] },
               6: { 'realPart': [-20, 40], 'imagPart': [-30, 30] } }


# Returns > 1 if argument is not in region of absolute stability
def stabilityFunction(hTimesLambda, s):
    stabPolyCoeffs = list(BDFcoeffs[s]['alpha'])
    stabPolyCoeffs[0] -= hTimesLambda * BDFcoeffs[s]['beta']
    return max(abs(np.roots(stabPolyCoeffs)))


# Plot all stability region boundaries of bdf on one plot
def plot_bdf_all():
    legend_string = r"$k = {}$"
    colors = sns.color_palette('Blues', len(BDFcoeffs))

    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

    for s in range(1, 7):
        x = np.linspace(*plotWindow[s]['realPart'], num=400)
        y = np.linspace(*plotWindow[s]['imagPart'], num=400)
        [X, Y] = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
    
        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                Z[m, n] = stabilityFunction(X[m, n] + 1j * Y[m, n], s)
    
        c = ax.contour(X, Y, Z, [1], colors=[colors[s-1]])
        c.collections[0].set_label(legend_string.format(s))

    ax.plot(plotWindow[s]['realPart'], [0, 0], 'k--', alpha=0.5)
    ax.plot([0, 0], plotWindow[s]['imagPart'], 'k--', alpha=0.5)

    ax.annotate(r'Im ($\lambda \Delta t$)', (0, 25), (1, 25), xycoords='data', fontsize=12)
    ax.annotate(r'Re($\lambda \Delta t$)', (35, 0), (33, 1), xycoords='data', fontsize=12)
    ax.legend()

    plt.savefig('Stability_all_BDF.png')
    ax.set_title('Stability Regions for Backward Difference Formulae')
    plt.savefig('Stability_all_BDF.pdf')


# Plot stability regions of bdf on separate plots
def plot_bdf():
    for s in range(1, 7):
        plt.figure()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        x = np.linspace(*plotWindow[s]['realPart'], num=400)
        y = np.linspace(*plotWindow[s]['imagPart'], num=400)
        [X, Y] = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
    
        for m in range(X.shape[0]):
            for n in range(X.shape[1]):
                Z[m, n] = stabilityFunction(X[m, n] + 1j * Y[m, n], s)
    
        plt.contour(X, Y, Z, [1], colors='k')
        plt.contourf(X, Y, Z, [0, 1], colors=[[0.8, 0.9, 1]])
        plt.plot(plotWindow[s]['realPart'], [0, 0], 'k--', alpha=0.5)
        plt.plot([0, 0], plotWindow[s]['imagPart'], 'k--', alpha=0.5)
        plt.xlabel(r'Re($\lambda \Delta t$)')
        plt.ylabel(r'Im($\lambda \Delta t$)')

        plt.savefig(f'Stability_BDF{s}.png')
        plt.title(f'Stability Region for BDF{s}')
        plt.savefig(f'Stability_BDF{s}.pdf')


