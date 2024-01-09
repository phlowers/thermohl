#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermohl import rain
from thermohl import solver
from thermohl import sun
from thermohl import utils


def dico():
    """Dataset similar to Achene-Lonny spans"""
    dc = {}
    dc['lat'] = 50.
    dc['alt'] = 300.
    dc['azm'] = 0.
    dc['month'] = 1
    dc['day'] = 1
    dc['hour'] = 0
    dc['Ta'] = 15.
    dc['Pa'] = 1.0E+05
    dc['rh'] = 0.67
    dc['pr'] = np.nan
    dc['ws'] = np.nan
    dc['wa'] = np.nan
    dc['I'] = np.nan
    dc['tb'] = 0.1
    dc['m'] = 1.571
    dc['d'] = 0.
    dc['D'] = 3.105E-02
    dc['a'] = 0.
    dc['A'] = 5.702E-04
    dc['l'] = 1.0
    dc['c'] = 900.
    dc['alpha'] = 0.9
    dc['epsilon'] = 0.8
    dc['RDC20'] = 5.886E-05
    dc['km'] = 1.0
    dc['ki'] = 0.0
    dc['kl'] = 4.0E-03
    dc['kq'] = 8.0E-07

    return dc


def dico_t(dc, dt):
    dc2 = dc.copy()
    for k in dt.keys():
        if k == 'time':
            continue
        else:
            dc2[k] = dt[k].copy()
    return dc2


def scn(I=900, pr=0., ws=0.):
    nh = 10
    dt = {}
    dt['time'] = np.linspace(0, nh * 3600, nh * 3600 + 1)
    dt['I'] = np.zeros_like(dt['time'])
    dt['I'][dt['time'] >= 600] = I
    dt['Ta'] = dc['Ta'] * np.ones_like(dt['time'])
    dt['wa'] = 90. * np.ones_like(dt['time'])
    dt['ws'] = np.zeros_like(dt['time'])
    dt['ws'][np.logical_and(dt['time'] >= 7 * 3600., dt['time'] < 9 * 3600.)] = ws
    dt['pr'] = np.zeros_like(dt['time'])
    dt['pr'][np.logical_and(dt['time'] >= 3 * 3600., dt['time'] < 5 * 3600.)] = pr
    return dt


def plot_scn(dt, tx, tc):
    key = []
    for k in dt.keys():
        if k == 'time' or k == 'thx':
            continue
        else:
            key.append(k)
    fig, ax = plt.subplots(nrows=len(key), ncols=1)
    for i, k in enumerate(key):
        ax[i].plot(dt['time'][tx] / tc, dt[k][tx], c='C0', label=k)
        ax[i].grid(True)
        ax[i].legend()

    plt.show()

    return


def plot_imax(dc):
    sl = solver.olla(dc)
    plt.figure()
    sl.dc['ws'] = np.linspace(0, 5, 501)
    sl.dc['wa'] = 90.
    for Tm in [30, 40, 50, 60, 75]:
        di = sl.steady_intensity(Tm, target='avg')
        plt.plot(sl.dc['ws'], di['I_max'], label='Tmax=%.f' % (Tm,))
    plt.ylabel('Imax (A)')
    plt.xlabel('Normal wind speed (m/s)')
    plt.title('Max transit to keep T < Tmax')
    plt.legend()
    plt.grid(True)

    plt.show()

    return


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    # input data
    dc = dico()
    dt = scn(I=900, pr=2.0 / 3.6E+06, ws=1.5)
    tc = 3600
    tx = range(len(dt['time']))[::10]

    # scn check plot and get Imax(ws) for given conditions
    _b = True
    if _b:
        plot_scn(dt, tx, tc)
        plot_imax(dc)

    Imax = 900.
    pr = np.array([2., 4., 8.]) / 3.6E+06
    ws = np.array([5., 10., 15.]) / 3.6

    fig, ax = plt.subplots()
    fih, ay = plt.subplots()
    for i in range(3):
        dt = scn(I=Imax, pr=pr[i], ws=ws[i])
        sls = solver.olla(dico_t(dc, dt))
        slt = solver.olla(dc)
        sls.pc = rain.PrecipitationCooling()
        slt.pc = rain.PrecipitationCooling()

        # --
        drs = sls.steady_temperature(return_avg=True, return_power=True)
        drt = slt.transient_temperature(T0=dc['Ta'], time=dt['time'], transit=dt['I'], Ta=dt['Ta'], wind_angle=dt['wa'],
                                        wind_speed=dt['ws'], pr=dt['pr'], return_core=True, return_avg=True,
                                        return_power=True)

        dt['thx'] = dt['time'][tx] / tc
        drs = drs.loc[tx, :]
        for k in drt:
            drt[k] = drt[k][tx]

        ci = 'C%1d' % (i,)
        ax.plot(dt['thx'], drs['T_avg'], c=ci, ls='--')
        ax.plot(dt['thx'], drt['T_avg'], c=ci, label='p=%.0f mm/h, u=%.1f m/s' % (pr[i] * 3.6E+06, ws[i]))

        if i == 2:
            ay.set_title('case p=%.0f mm/h, u=%.1f m/s' % (pr[i] * 3.6E+06, ws[i]))
            for j, cl in enumerate(['P_joule', 'P_convection', 'P_radiation', 'P_precipitation']):
                cj = 'C%1d' % (j,)
                ay.plot(dt['thx'], drs.loc[:, cl], c=cj, ls='--')
                ay.plot(dt['thx'], drt[cl], c=cj, ls='-', label=cl)

    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Temperature')

    plt.show()

    ay.grid(True)
    ay.legend()
    ay.set_xlabel('Time (h)')
    ay.set_ylabel('Power (W/m)')
    ay.set_ylim([-5, +60.])

    plt.show()
