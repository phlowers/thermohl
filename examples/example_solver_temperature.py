#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from thermohl import solver


def test_solve(dct, tol=1.0E-06, mxi=64):
    """Given an input dictionnary, a default tolerance and a maximum number of
     iterations, compute the surface temperature for the four available models.
     Print compute time, Return a DataFrame with both input and computed
     temperature."""

    mdl = [
        dict(label='cigr', model=solver.cigre(dct)),
        dict(label='ieee', model=solver.ieee(dct)),
        dict(label='olla', model=solver.olla(dct, multi=False)),
        dict(label='ollm', model=solver.olla(dct, multi=True)),
        dict(label='cner', model=solver.cner(dct)),
    ]
    dfi = pd.DataFrame(dct)

    for d in mdl:
        res = d['model'].steady_temperature(tol=tol, maxiter=mxi)
        print('[%4s] compute time = %7.3f s, ie %.2E s per element'
              % (d['label'], d['model'].ctime, d['model'].ctime / len(res)))
        dfi['T_' + d['label']] = res['T_surf']

    return dfi


if __name__ == '__main__':

    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    # Generate N entries with random (but realistic) parameters.
    N = 99999
    dct = dict(lat=np.random.uniform(42., 51., N),
               alt=np.random.uniform(0., 1600., N),
               azm=np.random.uniform(0., 360., N),
               month=np.random.randint(1, 13, N),
               day=np.random.randint(1, 31, N),
               hour=np.random.randint(0, 24, N),
               Ta=np.random.uniform(0., 30., N),
               ws=np.random.uniform(0., 7., N),
               wa=np.random.uniform(0., 90., N),
               I=np.random.uniform(40., 4000., N))

    # Test 1 : compute temperature
    df = test_solve(dct)

    # Test 3 : check the rel. differences between core and surface temperature and olla/olla-multi/cner models
    dol = solver.olla(dct, multi=False).steady_temperature(return_core=True, return_avg=True, return_err=False,
                                                           return_power=True)
    dom = solver.olla(dct, multi=True).steady_temperature(return_core=True, return_avg=True, return_err=False,
                                                          return_power=True)
    dcn = solver.cner(dct).steady_temperature(return_core=True, return_avg=True, return_err=False, return_power=True)

    fig, ax = plt.subplots(nrows=3, ncols=3)
    for j in range(3):
        c = ['T_surf', 'T_avg', 'T_core'][j]
        ax[0, j].hist(2. * np.abs((dol[c] - dom[c]) / (dol[c] + dom[c])), bins=100, density=True)
        ax[0, j].set_title('Relative error on %s (olla/ollamulti)' % (c,))
        ax[1, j].hist(2. * np.abs((dol[c] - dcn[c]) / (dol[c] + dcn[c])), bins=100, density=True)
        ax[1, j].set_title('Relative error on %s (olla/cner)' % (c,))
        ax[2, j].hist(2. * np.abs((dom[c] - dcn[c]) / (dom[c] + dcn[c])), bins=100, density=True)
        ax[2, j].set_title('Relative error on %s (olla-multi/cner)' % (c,))
        for i in range(3):
            ax[i, j].grid(True)

    plt.show()

    # Test 2 : plot energy balance for all models (row j of the big dict
    # input). The int() conversion is here since the to_dict() method converts
    # everything to float
    j = np.argmax(2. * np.abs((dol[c] - dcn[c]) / (dol[c] + dcn[c])))
    dcj = df.loc[j, :].to_dict()
    for k in ['day', 'month']:
        dcj[k] = int(dcj[k])

    mdl = [
        dict(label='cigr', model=solver.cigre(dcj)),
        dict(label='ieee', model=solver.ieee(dcj)),
        dict(label='olla', model=solver.olla(dcj, multi=False)),
        dict(label='ollm', model=solver.olla(dcj, multi=True)),
        dict(label='cner', model=solver.cner(dcj)),
    ]
    plt.figure()
    t = np.linspace(-20., +220., 241)
    for d in mdl:
        plt.plot(t, d['model']._rhs_value(t), label=d['label'])
    plt.grid(True)
    plt.legend()
    plt.xlabel('Temperature (C)')
    plt.ylabel('Power balance (W/m)')
    plt.title('Test 2')

    plt.show()
