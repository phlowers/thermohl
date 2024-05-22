"""Generic radiative cooling term."""

from typing import Union

import numpy as np

from thermohl.air import kelvin


class PowerTerm:
    """Base class for power term."""

    def value(self, T: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
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
        return np.zeros_like(T)

    def derivative(self, T: Union[float, np.ndarray], dT: float = 1.0E-03, **kwargs) -> Union[float, np.ndarray]:
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
        return (self.value(T + dT, **kwargs) - self.value(T - dT, **kwargs)) / (2. * dT)


class RadiativeCooling(PowerTerm):
    """Generic power term for radiative cooling."""

    def value(self, T: Union[float, np.ndarray], Ta: Union[float, np.ndarray],
              D: Union[float, np.ndarray], epsilon: Union[float, np.ndarray],
              sigma: float = 5.67E-08, **kwargs) -> Union[float, np.ndarray]:
        r"""Compute radiative cooling using the Stefan-Boltzmann law.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).
        Ta : float or np.ndarray
            Ambient temperature (C).
        D : float or np.ndarray
            External diameter (m).
        epsilon : float or np.ndarray
            Emissivity.
        sigma : float, optional
            Stefan-Boltzmann constant in W.m\ :sup:`-2`\ K\ :sup:`4`\ . The
            default is 5.67E-08.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        return np.pi * sigma * epsilon * D * (kelvin(T)**4 - kelvin(Ta)**4)

    def derivative(self, T: Union[float, np.ndarray], Ta: Union[float, np.ndarray],
                   D: Union[float, np.ndarray], epsilon: Union[float, np.ndarray],
                   sigma: float = 5.67E-08, **kwargs) -> Union[float, np.ndarray]:
        r"""Analytical derivative of value method.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature (C).
        Ta : float or np.ndarray
            Ambient temperature (C).
        D : float or np.ndarray
            External diameter (m).
        epsilon : float or np.ndarray
            Emissivity.
        sigma : float, optional
            Stefan-Boltzmann constant in W.m\ :sup:`-2`\ K\ :sup:`4`\ . The
            default is 5.67E-08.

        Returns
        -------
        float or np.ndarray
            Power term derivative (W.m\ :sup:`-1`\ K\ :sup:`-1`\ ).

        """
        return 4. * np.pi * sigma * epsilon * D * T**3
