"""Power terms implementation using RTE's olla project choices.

Based on NT-RD-CNER-DL-SLA-20-00215 by RTE.
"""
from typing import Union

import numpy as np

from thermohl.power import ieee, cner
from thermohl.power.base import RadiativeCooling as RadiativeCooling_


class JouleHeating(cner.JouleHeating):
    """Joule heating term."""
    pass


class SolarHeating(ieee.SolarHeating):
    """Solar heating term."""

    def __init__(
            self,
            lat: Union[float, np.ndarray],
            azm: Union[float, np.ndarray],
            month: Union[int, np.ndarray[int]],
            day: Union[int, np.ndarray[int]],
            hour: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            alpha: Union[float, np.ndarray],
            srad: Union[float, np.ndarray] = None,
            **kwargs,
    ):
        r"""Init with args.

        See ieee.SolarHeating; it is exactly the same with altitude and
        turbidity set to zero. If more than one input are numpy arrays, they
        should have the same size.

        Parameters
        ----------
        lat : float or np.ndarray
            Latitude.
        azm : float or np.ndarray
            Azimuth.
        month : int or np.ndarray
            Month number (must be between 1 and 12).
        day : int or np.ndarray
            Day of the month (must be between 1 and 28, 29, 30 or 31 depending on
            month).
        hour : float or np.ndarray
            Hour of the day (solar, must be between 0 and 23).
        D : float or np.ndarray
            external diameter.
        alpha : float or np.ndarray
            Solar absorption coefficient.
        srad : xxx
            xxx

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        for k in ['alt', 'tb']:
            kwargs.pop(k)
        super().__init__(
            lat=lat, alt=0., azm=azm, tb=0., month=month, day=day, hour=hour, D=D, alpha=alpha, srad=srad,
            **kwargs
        )


class ConvectiveCooling(ieee.ConvectiveCooling):
    pass


class RadiativeCooling(RadiativeCooling_):
    pass
