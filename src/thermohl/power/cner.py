"""Power terms implementation matching cner's Excel sheet.

See NT-RD-CNER-DL-SLA-20-00215.
"""

from typing import Any, Optional

import numpy as np

from thermohl import floatArrayLike, intArrayLike
from thermohl.power import ieee
from thermohl.power.base import PowerTerm


class JouleHeating(PowerTerm):
    """Joule heating term."""

    def __init__(
        self,
        I: floatArrayLike,
        D: floatArrayLike,
        d: floatArrayLike,
        A: floatArrayLike,
        a: floatArrayLike,
        km: floatArrayLike,
        ki: floatArrayLike,
        kl: floatArrayLike,
        kq: floatArrayLike,
        RDC20: floatArrayLike,
        T20: floatArrayLike = 20.0,
        f: floatArrayLike = 50.0,
        **kwargs: Any,
    ):
        r"""Init with args.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        I : float or np.ndarray
            Transit intensity.
        D : float or np.ndarray
            External diameter.
        d : float or np.ndarray
            core diameter.
        A : float or np.ndarray
            External (total) section.
        a : float or np.ndarray
            core section.
        km : float or np.ndarray
            Coefficient for magnetic effects.
        ki : float or np.ndarray
            Coefficient for magnetic effects.
        kl : float or np.ndarray
            Linear resistance augmentation with temperature.
        kq : float or np.ndarray
            Quadratic resistance augmentation with temperature.
        RDC20 : float or np.ndarray
            Electric resistance per unit length (DC) at 20°C.
        T20 : float or np.ndarray, optional
            Reference temperature. The default is 20.
        f : float or np.ndarray, optional
            Current frequency (Hz). The default is 50.

        """
        self.I = I
        self.D = D
        self.d = d
        self.kem = self._kem(A, a, km, ki)
        self.kl = kl
        self.kq = kq
        self.RDC20 = RDC20
        self.T20 = T20
        self.f = f

    def _rdc(self, T: floatArrayLike) -> floatArrayLike:
        """
        Compute resistance per unit length for direct current.

        Parameters
        ----------
        T : float or np.ndarray
            Temperature array or value at which to compute the resistance.

        Returns
        -------
        float or np.ndarray
            Resistance per unit length for direct current at the given temperature(s).
        """
        dt = T - self.T20
        return self.RDC20 * (1.0 + self.kl * dt + self.kq * dt**2)

    def _ks(self, rdc: floatArrayLike) -> floatArrayLike:
        """
        Compute skin-effect coefficient.

        This method calculates the skin-effect coefficient based on the given
        resistance (rdc) and the object's attributes. The calculation is an
        approximation as described in the document [NT-RD-CNER-DL-SLA-20-00215].

        Parameters:
        rdc (float or np.ndarray): The resistance value(s) for which the skin-effect
                              coefficient is to be computed.

        Returns:
        floatArrayLike: The computed skin-effect coefficient(s).
        """
        # Note: approx version as in [NT-RD-CNER-DL-SLA-20-00215]
        z = (
            8
            * np.pi
            * self.f
            * (self.D - self.d) ** 2
            / ((self.D**2 - self.d**2) * 1.0e07 * rdc)
        )
        a = 7 * z**2 / (315 + 3 * z**2)
        b = 56 / (211 + z**2)
        beta = 1.0 - self.d / self.D
        return 1.0 + a * (1.0 - 0.5 * beta - b * beta**2)

    def _kem(
        self,
        A: floatArrayLike,
        a: floatArrayLike,
        km: floatArrayLike,
        ki: floatArrayLike,
    ) -> floatArrayLike:
        """
        Compute magnetic coefficient.

        Parameters
        ----------
        A : float or np.ndarray
            External (total) section.
        a : float or np.ndarray
            Core section.
        km : float or np.ndarray
            Coefficient for magnetic effects.
        ki : float or np.ndarray
            Coefficient for magnetic effects.

        Returns
        -------
        floatArrayLike
            Computed magnetic coefficient.
        """
        s = (
            np.ones_like(self.I)
            * np.ones_like(A)
            * np.ones_like(a)
            * np.ones_like(km)
            * np.ones_like(ki)
        )
        z = s.shape == ()
        if z:
            s = np.array([1.0])
        I_ = self.I * s
        a_ = a * s
        A_ = A * s
        m = a_ > 0.0
        ki_ = ki * s
        kem = km * s
        kem[m] += ki_[m] * I_[m] / ((A_[m] - a_[m]) * 1.0e06)
        if z:
            kem = kem[0]
        return kem

    def value(self, T: floatArrayLike) -> floatArrayLike:
        r"""Compute joule heating.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        rdc = self._rdc(T)
        ks = self._ks(rdc)
        rac = self.kem * ks * rdc
        return rac * self.I**2


class SolarHeating(ieee.SolarHeatingBase):
    def __init__(
        self,
        lat: floatArrayLike,
        azm: floatArrayLike,
        month: intArrayLike,
        day: intArrayLike,
        hour: floatArrayLike,
        D: floatArrayLike,
        alpha: floatArrayLike,
        srad: Optional[floatArrayLike] = None,
        **kwargs: Any,
    ):
        r"""Build with args.

        If more than one input are numpy arrays, they should have the same size.

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
        alpha : np.ndarray
            Solar absorption coefficient.
        srad : xxx
            xxx

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        est = ieee._SRad(
            [-42.0, +63.8, -1.922, 0.03469, -3.61e-04, +1.943e-06, -4.08e-09],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        for k in ["alt", "tb"]:
            if k in kwargs.keys():
                kwargs.pop(k)
        super().__init__(
            lat=lat,
            alt=0.0,
            azm=azm,
            tb=0.0,
            month=month,
            day=day,
            hour=hour,
            D=D,
            alpha=alpha,
            est=est,
            srad=srad,
            **kwargs,
        )


class ConvectiveCooling(ieee.ConvectiveCooling):
    """Convective cooling term.

    Very similar to IEEE. The differences are in some coefficient values for air
    constants.
    """

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


class RadiativeCooling(ieee.RadiativeCooling):
    pass
