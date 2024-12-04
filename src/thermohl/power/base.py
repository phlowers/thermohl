"""Generic radiative cooling term."""

from typing import Any
import numpy as np
from thermohl import floatArrayLike

_dT = 1.0e-03


class PowerTerm:
    """Base class for power term."""

    def __init__(self, **kwargs: Any):
        pass

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute power term value in function of temperature.

        Usually this function should be overridden in children classes; if it is
        not the case it will just return zero.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return np.zeros_like(T) if not np.isscalar(T) else 0.0

    def derivative(self, T: floatArrayLike, dT: float = _dT) -> floatArrayLike:
        r"""Compute power term derivative regarding temperature in function of temperature.

        Usually this function should be overriden in children classes; if it is
        not the case it will evaluate the derivative from the value method with
        a second-order approximation.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).
        dT : float, optional
            Temperature increment. The default is 1.0E-03.

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return (self.value(T + dT) - self.value(T - dT)) / (2.0 * dT)


class RadiativeCooling(PowerTerm):
    """Generic power term for radiative cooling."""

    def _celsius2kelvin(self, T: floatArrayLike) -> floatArrayLike:
        return T + self.zerok

    def __init__(
        self,
        Ta: floatArrayLike,
        D: floatArrayLike,
        epsilon: floatArrayLike,
        sigma: float = 5.67e-08,
        zerok: float = 273.15,
        **kwargs: Any,
    ):
        r"""Init with args.

        Parameters
        ----------
        Ta : float or np.ndarray
            Ambient temperature (C).
        D : float or np.ndarray
            External diameter (m).
        epsilon : float or np.ndarray
            Emissivity.
        sigma : float, optional
            Stefan-Boltzmann constant in W.m\ :sup:`-2`\ K\ :sup:`4`\ . The
            default is 5.67E-08.
        zerok : float, optional
            Value for zero kelvin.

        Returns
        -------

        """
        self.zerok = zerok
        self.Ta = self._celsius2kelvin(Ta)
        self.D = D
        self.epsilon = epsilon
        self.sigma = sigma

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute radiative cooling using the Stefan-Boltzmann law.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return (
            np.pi
            * self.sigma
            * self.epsilon
            * self.D
            * (self._celsius2kelvin(T) ** 4 - self.Ta**4)
        )

    def derivative(self, T: floatArrayLike) -> floatArrayLike:
        r"""Analytical derivative of value method.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return 4.0 * np.pi * self.sigma * self.epsilon * self.D * T**3
