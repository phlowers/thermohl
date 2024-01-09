#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from thermohl import solver


def test_solve(dct, Trep, tol=1.0E-06, mxi=64):
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
        res = d['model'].steady_intensity(Trep, tol=tol, maxiter=mxi)
        print('[%4s] compute time = %7.3f s, ie %.2E s per element'
              % (d['label'], d['model'].ctime, d['model'].ctime / len(res)))
        dfi['Im_' + d['label']] = res['I_max']

    return dfi


if __name__ == '__main__':

    import matplotlib

    matplotlib.use('TkAgg')
    plt.close('all')

    # Generate N entries with random (but realistic) parameters.
    N = 9999
    dct = dict(lat=np.random.uniform(42., 51., N),
               alt=np.random.uniform(0., 1600., N),
               azm=np.random.uniform(0., 360., N),
               month=np.random.randint(1, 13, N),
               day=np.random.randint(1, 31, N),
               hour=np.random.randint(0, 24, N),
               Ta=np.random.uniform(0., 30., N),
               ws=np.random.uniform(0., 7., N),
               wa=np.random.uniform(0., 90., N),
               I=np.random.uniform(20., 2000., N))

    # Test 1 : compute max intensity
    Trep = 99.
    df = test_solve(dct, Trep, tol=1.0E-03, mxi=16)

    # Test 2 : check balance and consistency
    tol = 1.0E-06
    mxi = 64

    mdl = [
        dict(label='cigr', model=solver.cigre(dct)),
        dict(label='ieee', model=solver.ieee(dct)),
        dict(label='olla', model=solver.olla(dct, multi=False)),
        dict(label='ollm', model=solver.olla(dct, multi=True)),
        dict(label='cner', model=solver.cner(dct)),
    ]

    fig, ax = plt.subplots(nrows=2, ncols=5)
    for i, d in enumerate(mdl):
        slv = d['model']
        df = slv.steady_intensity(Trep, tol=tol, maxiter=mxi, return_power=True)
        df['pb'] = df['P_joule'] + df['P_solar'] - df['P_convection'] - df['P_radiation']
        slv.dc['I'] = df['I_max'].values
        df['TIrep'] = slv.steady_temperature(return_power=False)['T_surf']

        ax[0, i].hist(df['pb'], bins=100)
        ax[0, i].grid(True)
        ax[1, i].hist(np.abs(1. - df['TIrep'] / Trep), bins=100)
        ax[1, i].grid(True)
        ax[0, i].set_title(d['label'])

    plt.show()
