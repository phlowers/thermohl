from typing import Any

import numpy as np

from thermohl import floatArrayLike
from thermohl.power.cner import Air
from thermohl.power.convective_cooling import ConvectiveCoolingBase


class ConvectiveCooling(ConvectiveCoolingBase):
    """Convective cooling term.

    Very similar to IEEE. The differences are in some coefficient values for air
    constants.
    """

    def __init__(
        self,
        alt: floatArrayLike,
        azm: floatArrayLike,
        Ta: floatArrayLike,
        ws: floatArrayLike,
        wa: floatArrayLike,
        D: floatArrayLike,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        alt : float or np.ndarray
            Altitude.
        azm : float or np.ndarray
            Azimuth.
        Ta : float or np.ndarray
            Ambient temperature.
        ws : float or np.ndarray
            Wind speed.
        wa : float or np.ndarray
            Wind angle regarding north.
        D : float or np.ndarray
            External diameter.

        """
        super().__init__(
            alt,
            azm,
            Ta,
            ws,
            wa,
            D,
            Air.volumic_mass,
            Air.dynamic_viscosity,
            Air.thermal_conductivity,
        )

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute convective cooling.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        Tf = 0.5 * (T + self.Ta)
        Td = T - self.Ta
        # very slight difference with air.IEEE.volumic_mass() in coefficient before alt**2
        vm = (1.293 - 1.525e-04 * self.alt + 6.38e-09 * self.alt**2) / (
            1 + 0.00367 * Tf
        )
        return np.maximum(self._value_forced(Tf, Td, vm), self._value_natural(Td, vm))
