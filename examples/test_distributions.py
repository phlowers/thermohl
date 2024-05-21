# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

from thermohl.distributions import *
from thermohl.distributions import _vonmises_kappa, _vonmises_circ_var

matplotlib.use('TkAgg')


_twopi = 2 * np.pi


def _test_circular_moments_wrapnorm():
    n = 100
    xx = np.linspace(0, _twopi, n)
    mu = np.sin(xx)
    sg = np.linspace(0, 4, n + 1)[1:]
    low = -np.pi

    cavg = np.zeros_like(sg)
    cvar = np.zeros_like(sg)
    cstd = np.zeros_like(sg)

    for i in range(len(sg)):
        ns = 9999
        wn = WrappedNormal(mu[i], sg[i], lwrb=low)
        s = wn.rvs(ns)
        # z = np.exp(1j * s)
        # R = np.mean(z)
        # circ_mean = np.angle(R)
        cavg[i] = circmean(s, high=wn.uprb, low=wn.lwrb)
        # circ_var = 1 - R
        cvar[i] = circvar(s, high=wn.uprb, low=wn.lwrb)
        # circ_std = np.sqrt(-2. * np.log(np.abs(R)))
        cstd[i] = circstd(s, high=wn.uprb, low=wn.lwrb)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(sg, cavg, label='circ. mean')
    ax[0].plot(sg, mu, '--', c='gray')
    ax[1].plot(sg, cvar, label='circ. variance')
    ax[1].plot(sg, 1 - np.exp(-0.5 * sg**2), label='analytic')
    ax[2].plot(sg, cstd, label='circ. std')
    ax[2].plot(sg, sg, '--', c='gray')
    ax[2].axhline(y=np.pi, xmin=sg[0], xmax=sg[-1], c='gray', ls='--')
    for i in range(3):
        ax[i].grid(True)
        ax[i].legend()

    return


def _test_circular_moments_vonmises():
    n = 10
    xx = np.linspace(0, _twopi, n)
    mu = np.sin(xx)
    sg = np.linspace(0, 5, n + 1)[1:]
    kp = _vonmises_kappa(sg)
    low = -np.pi
    high = +np.pi

    cavg = np.zeros_like(kp)
    cvar = np.zeros_like(kp)
    cstd = np.zeros_like(kp)

    savg = np.zeros_like(kp)
    svar = np.zeros_like(kp)
    sstd = np.zeros_like(kp)

    for i in range(len(kp)):
        ns = 9999
        vm = scipy.stats.vonmises_line(kp[i], loc=mu[i])
        s = vm.rvs(ns)
        # z = np.exp(1j * s)
        # R = np.mean(z)
        # circ_mean = np.angle(R)
        cavg[i] = circmean(s, high=high, low=low)
        # circ_var = 1 - R
        cvar[i] = circvar(s, high=high, low=low)
        # circ_std = np.sqrt(-2. * np.log(np.abs(R)))
        cstd[i] = circstd(s, high=high, low=low)

        # same thing but from distribution
        savg[i] = vm.mean()
        svar[i] = vm.var()
        sstd[i] = vm.std()

    plt.figure()
    plt.plot(sg, kp)
    plt.plot(sg, 1 / sg**2, '--', c='gray')
    plt.xlabel('$\sigma$')
    plt.ylabel('$\kappa$')
    plt.grid(True)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    # computed circular mean matches distribution's mean() method;
    ax[0].plot(sg, cavg, label='circ. mean')
    ax[0].plot(sg, savg, label='dist. mean')
    ax[0].plot(sg, mu, '--', c='gray')

    # computed circular variance does not match distribution's var() method;
    # analytic formula from wikipedia matches computed circular variance;
    # actually, d.var() = d.std()**2
    ax[1].plot(sg, cvar, label='circ. variance')
    ax[1].plot(sg, svar, label='dist. variance')
    ax[1].plot(sg, _vonmises_circ_var(kp), label='analytic')
    # ax[1].plot(sg, 0.5 / kp, ls='--', c='gray', label='1/$2\kappa$')
    # ax[1].plot(sg, 1.0 / kp, ls='--', c='gray', label='1/$2\kappa$')
    ax[1].set_ylim([0, 1.1])

    # computed circular std matches distribution's std() method;
    ax[2].plot(sg, cstd, label='circ. std')
    ax[2].plot(sg, sstd, label='dist. std')
    ax[2].plot(sg, np.sqrt(-2 * np.log(1 - _vonmises_circ_var(kp))), label='examples')
    ax[2].axhline(y=np.pi, xmin=sg[0], xmax=sg[-1], c='gray', ls='--')
    ax[2].axhline(y=_twopi / np.sqrt(12), xmin=sg[0], xmax=sg[-1], c='gray', ls='--')
    # ax[2].plot(sg, 1 / np.sqrt(kp), ls='--', c='gray', label='$1/\sqrt{\kappa}$')

    for i in range(3):
        ax[i].grid(True)
        ax[i].legend()

    return


def _test_truncnorm(a, b, mu, sigma, n=9999, ci=0.95, plot=False):
    d = truncnorm(a, b, mu, sigma)

    nb = max(20, (n + 1) // 50)
    s = d.rvs(n)
    x = np.linspace(a, b, 1001)

    xmin = max(a, d.mean() - 4 * d.std())
    xmax = min(b, d.mean() + 4 * d.std())
    qmin = 0.5 * (1 - ci)
    qmax = 0.5 * (1 + ci)

    if plot:
        fig, ax = plt.subplots()
        ax.hist(s, bins=nb, density=True, fc='C1')
        ax.plot(x, d.pdf(x), '-', c='C0')
        ax.grid(True)
        ax.set_xlim([xmin, xmax])
        ax.grid(True)

    print('dist: min=%+.3E, max=%+.3E, avg=%+.3E, std=%+.3E, ci=%+.3E (%.0f%%, std-norm.)'
          % (a, b, d.mean(), d.std(), 0.5 * (d.ppf(qmax) - d.ppf(qmin)) / d.std(), 100 * ci))
    print('splt: min=%+.3E, max=%+.3E, avg=%+.3E, std=%+.3E, ci=%+.3E (%.0f%%, std-norm.)'
          % (np.min(s), np.max(s), np.mean(s), np.std(s),
             0.5 * (np.quantile(s, qmax) - np.quantile(s, qmin)) / sigma, 100 * ci))

    return


def _test_wrapnorm(mu, sigma, n=999, ci=0.95, plot=False):
    a = 0.
    b = _twopi
    d = wrapnorm(mu, sigma)

    nb = max(20, (n + 1) // 50)
    s = d.rvs(n)
    # x = np.linspace(a, b, 1001)

    xmin = a
    xmax = b
    qmin = 0.5 * (1 - ci)
    qmax = 0.5 * (1 + ci)

    if plot:
        fig, ax = plt.subplots()
        ax.hist(s / np.pi, bins=nb, density=True, fc='C1')
        # ax.plot(x / np.pi, d.pdf(x), '-', c='C0')
        ax.grid(True)
        ax.set_xlim([xmin / np.pi, xmax / np.pi])
        ax.set_xlabel('$x/\pi$')
        ax.grid(True)

    print('dist: min=%+.3E, max=%+.3E, avg=%+.3E,  std=%+.3E,  ci=%+.3E (%.0f%%, std-norm.)'
          % (a, b, d.mean(), d.std(), np.nan, 100 * ci))
    print('splt: min=%+.3E, max=%+.3E, avg=%+.3E*, std=%+.3E*, ci=%+.3E (%.0f%%, std-norm.) *circ'
          % (np.min(s), np.max(s), circmean(s, high=b, low=a), circstd(s, high=b, low=a),
             0.5 * (np.quantile(s, qmax) - np.quantile(s, qmin)) / sigma, 100 * ci))

    return


def _test_vonmises(mu, sigma, n=999, ci=0.95, plot=False):
    a = -np.pi
    b = +np.pi
    d = vonmises(mu, sigma)

    nb = max(20, (n + 1) // 50)
    s = d.rvs(n)
    x = np.linspace(a, b, 1001)

    xmin = a
    xmax = b
    qmin = 0.5 * (1 - ci)
    qmax = 0.5 * (1 + ci)

    if plot:
        fig, ax = plt.subplots()
        ax.hist(s / np.pi, bins=nb, density=True, fc='C1')
        ax.plot(x / np.pi, d.pdf(x), '-', c='C0')
        ax.grid(True)
        ax.set_xlim([xmin / np.pi, xmax / np.pi])
        ax.set_xlabel('$x/\pi$')
        ax.grid(True)

    print('dist: min=%+.3E, max=%+.3E, avg=%+.3E,  std=%+.3E,  ci=%+.3E (%.0f%%, std-norm.)'
          % (a, b, d.mean(), d.std(), 0.5 * (d.ppf(qmax) - d.ppf(qmin)) / d.std(), 100 * ci))
    print('splt: min=%+.3E, max=%+.3E, avg=%+.3E*, std=%+.3E*, ci=%+.3E (%.0f%%, std-norm.) *circ'
          % (np.min(s), np.max(s), circmean(s, high=b, low=a), circstd(s, high=b, low=a),
             0.5 * (np.quantile(s, qmax) - np.quantile(s, qmin)) / sigma, 100 * ci))

    return


_test_circular_moments_wrapnorm()
_test_circular_moments_vonmises()
_test_truncnorm(0., 1., 0.6, 0.2)
_test_wrapnorm(1., 0.3)
_test_vonmises(1., 0.3)
