from typing import Union, Tuple

import numpy as np

from thermohl.power.base import PowerTerm


class JouleHeating(PowerTerm):

    def __init__(self, ol, rl: np.ndarray, ly: np.ndarray, ry: np.ndarray, iy: np.ndarray):
        pass

    def value(self, T: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        pass

    def derivative(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    def value_layer(self):
        pass

    def value_dicr(self):
        pass

    # def integrate_value_disc(jd, s):
    #     s_ = np.broadcast_to(s, jd.shape)
    #     js = jd * s_
    #     ij = 2. * np.pi * np.sum(0.5 * (js[..., 1:] + js[..., :-1]) * np.diff(s_, axis=-1), axis=-1)
    #     return ij


class _JouleHeating:
    """Joule heating."""

    @staticmethod
    def _RDC(
            T: Union[float, np.ndarray],
            kl: Union[float, np.ndarray],
            kq: Union[float, np.ndarray],
            RDC20: Union[float, np.ndarray],
            T20: Union[float, np.ndarray] = 20.,
    ) -> Union[float, np.ndarray]:
        """Compute resistance per unit length for direct current."""
        return RDC20 * (1. + kl * (T - T20) + kq * (T - T20)**2)

    @staticmethod
    def _ks(
            R: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            d: Union[float, np.ndarray],
            f: Union[float, np.ndarray] = 50.,
    ) -> Union[float, np.ndarray]:
        """Compute skin-effect coefficient."""
        # Note: approx version as in [NT-RD-CNER-DL-SLA-20-00215]
        z = 8 * np.pi * f * (D - d)**2 / ((D**2 - d**2) * 1.0E+07 * R)
        a = 7 * z**2 / (315 + 3 * z**2)
        b = 56 / (211 + z**2)
        beta = 1. - d / D
        return 1 + a * (1. - 0.5 * beta - b * beta**2)

    @staticmethod
    def _kem(
            I: Union[float, np.ndarray],
            A: Union[float, np.ndarray],
            a: Union[float, np.ndarray],
            km: Union[float, np.ndarray],
            ki: Union[float, np.ndarray],
            shape: Tuple[int, ...]
    ) -> Union[float, np.ndarray]:
        """Compute magnetic coefficient."""
        s = np.ones(shape)
        I_ = I * s
        a_ = a * s
        A_ = A * s
        m = a > 0.
        kem = km * s
        ki = ki * s
        kem[m] += ki[m] * I_[m] / ((A_[m] - a_[m]) * 1.0E+06)
        return kem

    @staticmethod
    def _RAC(
            T: Union[float, np.ndarray],
            I: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            d: Union[float, np.ndarray],
            A: Union[float, np.ndarray],
            a: Union[float, np.ndarray],
            km: Union[float, np.ndarray],
            ki: Union[float, np.ndarray],
            kl: Union[float, np.ndarray],
            kq: Union[float, np.ndarray],
            RDC20: Union[float, np.ndarray],
            T20: Union[float, np.ndarray] = 20.,
            f: Union[float, np.ndarray] = 50.,
            shape: Tuple[int, ...] = (1,),
    ) -> Union[float, np.ndarray]:
        """Compute resistance per unit length for alternative current."""
        R = _JouleHeating._RDC(T, kl, kq, RDC20, T20)
        ks = _JouleHeating._ks(R, D, d, f)
        km = _JouleHeating._kem(I, A, a, km, ki, shape)
        return km * ks * R

    @staticmethod
    def value(
            T: Union[float, np.ndarray],
            I: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            d: Union[float, np.ndarray],
            A: Union[float, np.ndarray],
            a: Union[float, np.ndarray],
            km: Union[float, np.ndarray],
            ki: Union[float, np.ndarray],
            kl: Union[float, np.ndarray],
            kq: Union[float, np.ndarray],
            RDC20: Union[float, np.ndarray],
            T20: Union[float, np.ndarray] = 20.,
            f: Union[float, np.ndarray] = 50.,
            shape: Tuple[int, ...] = (1,),
    ) -> Union[float, np.ndarray]:
        r"""Compute joule heating.

        If more than one input are numpy arrays, they should have the same size.

        Parameters
        ----------
        T : float or np.ndarray
            Conductor temperature.
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
        shape : tuple of ints, optional
            Shape of result. The default is (1,)

        Returns
        -------
        float or np.ndarray
            Power term value (W.m\ :sup:`-1`\ ).

        """
        RDC = _JouleHeating._RDC(T, kl, kq, RDC20, T20)
        ks = _JouleHeating._ks(RDC, D, d, f)
        km = _JouleHeating._kem(I, A, a, km, ki, shape)
        RAC = km * ks * RDC
        p = RAC * I**2
        if shape == (1,):
            return p[0]
        return p

    @staticmethod
    def value_layer(
            T: Union[float, np.ndarray],
            I: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            d: Union[float, np.ndarray],
            A: Union[float, np.ndarray],
            a: Union[float, np.ndarray],
            km: Union[float, np.ndarray],
            ki: Union[float, np.ndarray],
            kl: Union[float, np.ndarray],
            kq: Union[float, np.ndarray],
            ol: np.ndarray,
            T20: Union[float, np.ndarray] = 20.,
            f: Union[float, np.ndarray] = 50.,
            shape: Tuple[int, ...] = (1,),
    ) -> Union[float, np.ndarray]:
        RDC20 = 1. / np.sum(1. / ol)
        il = 1 / (ol * np.sum(1 / ol))

        if shape == (1,):
            shape2a = ol.shape
            shape2b = shape2a
        else:
            shape2a = ol.shape + shape
            shape2b = shape + ol.shape

        T_ = np.moveaxis(np.broadcast_to(T, shape2a), 0, -1)
        I_ = np.moveaxis(np.broadcast_to(I, shape2a), 0, -1)
        D_ = np.moveaxis(np.broadcast_to(D, shape2a), 0, -1)
        d_ = np.moveaxis(np.broadcast_to(d, shape2a), 0, -1)
        A_ = np.moveaxis(np.broadcast_to(A, shape2a), 0, -1)
        a_ = np.moveaxis(np.broadcast_to(a, shape2a), 0, -1)
        km_ = np.moveaxis(np.broadcast_to(km, shape2a), 0, -1)
        ki_ = np.moveaxis(np.broadcast_to(ki, shape2a), 0, -1)
        kl_ = np.moveaxis(np.broadcast_to(kl, shape2a), 0, -1)
        kq_ = np.moveaxis(np.broadcast_to(kq, shape2a), 0, -1)
        ol_ = np.broadcast_to(ol, shape2b)
        il_ = np.broadcast_to(il, shape2b)

        RDC_l = _JouleHeating._RDC(T_, kl_, kq_, ol_, T20)
        RDC = _JouleHeating._RDC(T_, kl_, kq_, RDC20, T20)

        ksk = _JouleHeating._ks(RDC, D_, d_, f)
        kem = _JouleHeating._kem(I_, A_, a_, km_, ki_, shape2b)
        RAC = kem * ksk * RDC_l

        return RAC * (il_ * I)**2

    @staticmethod
    def value_discr(
            T: float,
            I: Union[float, np.ndarray],
            D: Union[float, np.ndarray],
            d: Union[float, np.ndarray],
            A: Union[float, np.ndarray],
            a: Union[float, np.ndarray],
            km: Union[float, np.ndarray],
            ki: Union[float, np.ndarray],
            kl: Union[float, np.ndarray],
            kq: Union[float, np.ndarray],
            RDC20: Union[float, np.ndarray],
            rl: np.ndarray,
            ly: np.ndarray,
            ry: np.ndarray,
            iy: np.ndarray,
            T20: Union[float, np.ndarray] = 20.,
            f: Union[float, np.ndarray] = 50.,
            shape: Tuple[int, ...] = (1,),
    ) -> Union[float, np.ndarray]:

        bl = np.pi * (rl**2 - np.concatenate(([0.], rl[:-1]))**2)

        if shape == (1,):
            shape2a = ly.shape
            shape2b = shape2a
        else:
            shape2a = ly.shape + shape
            shape2b = shape + ly.shape

        T_ = np.moveaxis(np.broadcast_to(T, shape2a), 0, -1)
        I_ = np.moveaxis(np.broadcast_to(I, shape2a), 0, -1)
        D_ = np.moveaxis(np.broadcast_to(D, shape2a), 0, -1)
        d_ = np.moveaxis(np.broadcast_to(d, shape2a), 0, -1)
        A_ = np.moveaxis(np.broadcast_to(A, shape2a), 0, -1)
        a_ = np.moveaxis(np.broadcast_to(a, shape2a), 0, -1)
        R_ = np.moveaxis(np.broadcast_to(RDC20, shape2a), 0, -1)
        km_ = np.moveaxis(np.broadcast_to(km, shape2a), 0, -1)
        ki_ = np.moveaxis(np.broadcast_to(ki, shape2a), 0, -1)
        kl_ = np.moveaxis(np.broadcast_to(kl, shape2a), 0, -1)
        kq_ = np.moveaxis(np.broadcast_to(kq, shape2a), 0, -1)

        ry_ = np.broadcast_to(ry, shape2b)
        iy_ = np.broadcast_to(iy, shape2b)

        RDC_l = _JouleHeating._RDC(T_, kl_, kq_, ry_, T20)
        RDC = _JouleHeating._RDC(T_, kl_, kq_, R_, T20)

        ksk = _JouleHeating._ks(RDC, D_, d_, f)
        kem = _JouleHeating._kem(I_, A_, a_, km_, ki_, shape2b)
        RAC = kem * ksk * RDC_l

        return RAC * (iy_ * I)**2 / bl[ly]

    @staticmethod
    def integrate_value_disc(jd, s):
        s_ = np.broadcast_to(s, jd.shape)
        js = jd * s_
        ij = 2. * np.pi * np.sum(0.5 * (js[..., 1:] + js[..., :-1]) * np.diff(s_, axis=-1), axis=-1)
        return ij
