# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Functions and classes to obtain various probability distributions for scalar random variables."""

import warnings
from typing import Union, Optional

import numpy as np

from thermohl.utils import depends_on_optional

try:
    import scipy
    from scipy.special import erf, i0, i1
    from scipy.stats._distn_infrastructure import rv_continuous_frozen as frozen_dist
except ImportError:
    warnings.warn(
        "scipy is not installed. Some functions will not be available.",
        RuntimeWarning,
    )
    frozen_dist = object


from thermohl import floatArrayLike

_twopi = 2 * np.pi


def _phi(value: float) -> float:
    """PDF of standard normal distribution."""
    return np.exp(-0.5 * value**2) / np.sqrt(_twopi)


@depends_on_optional("scipy")
def _psi(value: float) -> float:
    """CDF of standard normal distribution."""
    return 0.5 * (1 + erf(value / np.sqrt(2)))


def _truncnorm_header(
    lower_bound: float, upper_bound: float, mean: float, standard_deviation: float
) -> tuple[float, float, float, float]:
    """Utility code factoring."""
    alpha = (lower_bound - mean) / standard_deviation
    beta = (upper_bound - mean) / standard_deviation
    return alpha, beta, mean, standard_deviation


def _truncnorm_mean_std(
    lower_bound: float, upper_bound: float, mean: float, standard_deviation: float
) -> tuple[float, float]:
    """Real mean and std of truncated normal distribution."""
    alpha, beta, mean, standard_deviation = _truncnorm_header(
        lower_bound, upper_bound, mean, standard_deviation
    )
    normalizer = _psi(beta) - _psi(alpha)
    mean_value = mean + standard_deviation * (_phi(alpha) - _phi(beta)) / normalizer
    standard_value = standard_deviation * np.sqrt(
        1
        + (alpha * _phi(alpha) - beta * _phi(beta)) / normalizer
        - ((_phi(alpha) - _phi(beta)) / normalizer) ** 2
    )
    return mean_value, standard_value


@depends_on_optional("scipy")
def truncnorm(
    lower_bound: float,
    upper_bound: float,
    mean: float,
    standard_deviation: float,
    mean_tolerance: float = 1.0e-03,
    std_tolerance: float = 1.0e-02,
    relative_tolerance: bool = True,
) -> frozen_dist:
    """Truncated normal distribution. Wrapper from scipy.stats."""
    if lower_bound >= upper_bound:
        raise ValueError(
            "Input lower_bound (%.3E) should be lower than upper_bound (%.3E)."
            % (lower_bound, upper_bound)
        )
    if mean < lower_bound or mean > upper_bound:
        raise ValueError(
            "Input mean (%.3E) should be in [lower_bound, upper_bound] range (%.3E, %.3E)."
            % (mean, lower_bound, upper_bound)
        )
    if standard_deviation < 0.0:
        raise ValueError(
            "Input standard_deviation (%.3E) should be positive."
            % (standard_deviation,)
        )

    target_mean = mean
    target_standard_deviation = standard_deviation
    alpha, beta, mean, standard_deviation = _truncnorm_header(
        lower_bound, upper_bound, target_mean, target_standard_deviation
    )
    distribution = scipy.stats.truncnorm(alpha, beta, mean, standard_deviation)

    actual_mean = distribution.mean()
    actual_standard_deviation = distribution.std()

    mean_tol = mean_tolerance
    std_tol = std_tolerance
    if relative_tolerance:
        mean_tol *= target_mean
        std_tol *= target_standard_deviation
    if np.abs(target_standard_deviation - actual_standard_deviation) >= std_tol:
        warnings.warn(
            "Required std cannot be achieved (%.3E instead of %.3E). Choose a lower std, extend your "
            "bounds or change your distribution."
            % (target_standard_deviation, actual_standard_deviation),
            RuntimeWarning,
        )
    if np.abs(target_mean - actual_mean) >= mean_tol:
        warnings.warn(
            "Required mean cannot be achieved (%.3E instead of %.3E). Move the mean towards the center of your "
            "bounds, extend your bounds or change your distribution."
            % (target_mean, actual_mean),
            RuntimeWarning,
        )

    return distribution


@depends_on_optional("scipy")
class WrappedNormal(object):
    """Wrapped-Normal distribution. Not as complete as a scipy.stat distribution."""

    def __init__(
        self, mean: float, standard_deviation: float, lower_bound: float = 0.0
    ):
        if standard_deviation < 0:
            raise ValueError("Std should be positive.")
        if mean < lower_bound or mean >= lower_bound + _twopi:
            raise ValueError(
                "Mean should be greater than lower bound and lower than lower bound + 2*pi."
            )
        self.lower_bound = lower_bound
        self.upper_bound = lower_bound + _twopi
        self.mean_value = mean
        self.standard_deviation = standard_deviation
        return

    @depends_on_optional("scipy")
    def rvs(
        self,
        size: int = 1,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.RandomState]
        ] = None,
    ) -> floatArrayLike:
        sample = scipy.stats.norm.rvs(
            loc=self.mean_value,
            scale=self.standard_deviation,
            size=size,
            random_state=random_state,
        )
        sample = sample % _twopi
        sample[sample < self.lower_bound] += _twopi
        sample[sample > self.upper_bound] -= _twopi
        return sample

    def mean(self) -> float:
        return self.mean_value

    def median(self) -> float:
        return self.mean_value

    def var(self) -> float:
        return 1 - np.exp(-0.5 * self.standard_deviation**2)

    def std(self) -> float:
        return np.sqrt(self.var())

    def ppf(self, q: float) -> np.float64:
        return np.quantile(self.rvs(9999), q)


def wrapnorm(mean: float, standard_deviation: float) -> WrappedNormal:
    """Get Wrapped Normal distribution.
    -- in radians, in [0, 2*pi]
    """
    mean_wrapped = mean % _twopi
    if mean_wrapped != mean:
        warnings.warn(
            "Changed mean from %.3E to %.3E to fit [0,2*pi] interval."
            % (mean, mean_wrapped),
            RuntimeWarning,
        )
    if standard_deviation >= _twopi:
        warnings.warn(
            "Required std cannot be achieved (%.3E > 2*pi). Choose a lower std "
            "or change your distribution." % (standard_deviation,),
            RuntimeWarning,
        )
    return WrappedNormal(mean_wrapped, standard_deviation)


@depends_on_optional("scipy")
def _vonmises_circ_var(kappa: float) -> float:
    """Von Mises distribution circular variance."""
    return 1 - i1(kappa) / i0(kappa)


@depends_on_optional("scipy")
def _vonmises_kappa(standard_deviation: float) -> float:
    """Get von Mises parameter that matches std in input."""
    from scipy.optimize import newton

    circ_variance = 1.0 - np.exp(-0.5 * standard_deviation**2)
    kappa_guess = 0.5 / circ_variance

    def fun(x: float) -> float:
        return _vonmises_circ_var(x) - circ_variance

    try:
        kappa = newton(fun, x0=kappa_guess, tol=1.0e-06, maxiter=32)
    except RuntimeError:
        kappa = 1 / standard_deviation**2

    return kappa


@depends_on_optional("scipy")
def vonmises(mean: float, standard_deviation: float) -> frozen_dist:
    """Get von Mises distribution.
    -- in radians, in [-pi,+pi]
    """
    mean_wrapped = mean % _twopi
    if mean_wrapped > np.pi:
        mean_wrapped -= _twopi
    if mean_wrapped != mean:
        warnings.warn(
            "Changed mean from %.3E to %.3E to fit [-pi,+pi] interval."
            % (mean, mean_wrapped),
            RuntimeWarning,
        )
    if standard_deviation < 0.0:
        raise ValueError(
            "Input standard_deviation (%.3E) should be positive."
            % (standard_deviation,)
        )
    sigmax = _twopi / np.sqrt(12)
    if standard_deviation >= sigmax:
        warnings.warn(
            "Required std cannot be achieved (%.3E > %.3E). Choose a lower std "
            "or change your distribution." % (standard_deviation, sigmax),
            RuntimeWarning,
        )
    kappa = _vonmises_kappa(standard_deviation)
    return scipy.stats.vonmises_line(kappa, loc=mean_wrapped)
