"""Functions and classes to obtain various probability distributions for scalar random variables."""
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.special import erf
from scipy.special import i0
from scipy.special import i1
from scipy.stats import circmean
from scipy.stats import circstd
from scipy.stats import circvar

_twopi = 2 * np.pi


def _phi(x: float) -> float:
    """PDF of standard normal distribution."""
    return np.exp(-0.5 * x**2) / np.sqrt(_twopi)


def _psi(x: float) -> float:
    """CDF of standard normal distribution."""
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def _truncnorm_header(a: float, b: float, mu: float, sigma: float) \
        -> tuple[float, float, float, float]:
    """Utility code factoring."""
    m = mu
    s = sigma
    al = (a - m) / s
    bt = (b - m) / s
    return al, bt, m, s


def _truncnorm_mean_std(a: float, b: float, mu: float, sigma: float) -> tuple[float, float]:
    """Real mean and std of truncated normal distribution."""
    al, bt, m, s = _truncnorm_header(a, b, mu, sigma)
    Z = _psi(bt) - _psi(al)
    mean = m + s * (_phi(al) - _phi(bt)) / Z
    std = s * np.sqrt(1 + (al * _phi(al) - bt * _phi(bt)) / Z - ((_phi(al) - _phi(bt)) / Z)**2)
    return mean, std


def truncnorm(a: float, b: float, mu: float, sigma: float,
              err_mu: float = 1.0E-03, err_sigma: float = 1.0E-02, rel_err: bool = True):
    """Truncated normal distribution. Wrapper from scipy.stats."""
    if a >= b:
        raise ValueError('Input a (%E) should be lower than b (%E).' % (a, b))
    if mu < a or mu > b:
        raise ValueError('Input mu (%E) should be in [a, b] range (%E, %E).' % (mu, a, b))
    if sigma < 0.:
        raise ValueError('Input sigma (%E) should be positive.' % (sigma,))

    al, bt, m, s = _truncnorm_header(a, b, mu, sigma)
    dist = scipy.stats.truncnorm(al, bt, m, s)

    mean = dist.mean()
    std = dist.std()

    err_mu_ = err_mu
    err_sigma_ = err_sigma
    if rel_err:
        err_mu_ *= mu
        err_sigma_ *= sigma
    if np.abs(sigma - std) >= err_sigma_:
        warnings.warn(
            'Required std cannot be achieved (%E instead of %E). Choose a lower std, extend your '
            'bounds or change your distribution.' % (sigma, std), RuntimeWarning)
    if np.abs(mu - mean) >= err_mu_:
        warnings.warn(
            'Required mean cannot be achieved (%E instead of %E). Move the mean towards the center of your '
            'bounds, extend your bounds or change your distribution.' % (mu, mean), RuntimeWarning)

    return dist


class WrappedNormal(object):
    """Wrapped-Normal distribution. Not as complete as a scipy.stat distribution."""

    def __init__(self, mu: float, sigma: float, lwrb: float = 0.):
        if sigma < 0:
            raise ValueError('Std should be positive.')
        if mu < lwrb or mu >= lwrb + _twopi:
            raise ValueError('Mean should be greater than lower bound and lower than lower bound + 2*pi.')
        self.lwrb = lwrb
        self.uprb = lwrb + _twopi
        self.mu = mu
        self.sigma = sigma
        return

    def rvs(self, size: int = 1, random_state: Union[int, np.random.Generator, np.random.RandomState] = None) \
            -> Union[float, np.ndarray]:
        smpl = scipy.stats.norm.rvs(loc=self.mu, scale=self.sigma, size=size, random_state=random_state)
        smpl = smpl % _twopi
        smpl[smpl < self.lwrb] += _twopi
        smpl[smpl > self.uprb] -= _twopi
        return smpl

    def mean(self) -> float:
        return self.mu

    def median(self) -> float:
        return self.mu

    def var(self) -> float:
        return 1 - np.exp(-0.5 * self.sigma**2)

    def std(self) -> float:
        return np.sqrt(self.var())

    def ppf(self, q) -> float:
        return np.quantile(self.rvs(9999), q)


def wrapnorm(mu: float, sigma: float):
    """Get Wrapped Normal distribution.
      -- in radians, in [0, 2*pi]
    """
    mu2 = mu % _twopi
    if mu2 != mu:
        warnings.warn('Changed mean from %.E to %.E to fit [0,2*pi] interval.'
                      % (mu, mu2), RuntimeWarning)
    if sigma >= _twopi:
        warnings.warn(
            'Required std cannot be achieved (%E > 2*pi). Choose a lower std '
            'or change your distribution.' % (sigma,), RuntimeWarning)
    return WrappedNormal(mu2, sigma)


def _vonmises_circ_var(kappa: float) -> float:
    """Von Mises distribution circular variance."""
    return 1 - i1(kappa) / i0(kappa)


def _vonmises_kappa(sigma: float) -> float:
    """Get von Mises parameter that matches std in input."""
    from scipy.optimize import newton
    vr = 1. - np.exp(-0.5 * sigma**2)
    k0 = 0.5 / vr

    def fun(x) -> float:
        return _vonmises_circ_var(x) - vr

    try:
        kappa = newton(fun, x0=k0)
    except RuntimeError:
        kappa = 1 / sigma**2

    return kappa


def vonmises(mu: float, sigma: float):
    """Get von Mises distribution.
     -- in radians, in [-pi,+pi]
    """
    mu2 = mu % _twopi
    if mu2 > np.pi:
        mu2 -= _twopi
    if mu2 != mu:
        warnings.warn('Changed mean from %.E to %.E to fit [-pi,+pi] interval.'
                      % (mu, mu2), RuntimeWarning)
    if sigma < 0.:
        raise ValueError('Input sigma (%E) should be positive.' % (sigma,))
    sigmax = _twopi / np.sqrt(12)
    if sigma >= sigmax:
        warnings.warn(
            'Required std cannot be achieved (%E > %E). Choose a lower std '
            'or change your distribution.' % (sigma, sigmax), RuntimeWarning)
    kappa = _vonmises_kappa(sigma)
    return scipy.stats.vonmises_line(kappa, loc=mu)
