"""Rain cooling power term.

Based on *Modelling precipitation cooling of overhead conductors*, Pytlak et
al., 2011 (https://doi.org/10.1016/j.epsr.2011.06.004) and *Dynamic thermal
rating of power lines in raining conditions-model and measurements*, Maksic et
al., 2016 (https://doi.org/10.1109/PESGM.2016.7741611).
"""
import numpy as np

from thermohl import air
from thermohl import ieee
from thermohl import thermodynamics
from thermohl.utils import PowerTerm


class PrecipitationCooling(PowerTerm):
    """Precipitation cooling term."""

    def value(self, T, alt, Ta, ws, wa, D, Pa, rh, pr, **kwargs):
        # maksic = (PrecipitationCooling._evap(T, alt, Ta, ws, wa, D, Pa, rh, pr) +
        #           PrecipitationCooling._imp(T, Ta, ws, D, pr))
        pytlak = PrecipitationCooling._evap(T, alt, Ta, ws, wa, D, Pa, rh, pr)
        return pytlak

    @staticmethod
    def _ma(Ta, ws, D, pr, ps):
        # ! -- pr and ps in m.s**-1
        rho = thermodynamics.Water.volumic_mass(Ta)
        mar = np.sqrt((pr * rho)**2 + (ws * 23.589 * pr**0.8460)**2)
        mas = np.sqrt((ps * rho)**2 + (ws * 142.88 * ps**0.9165)**2)
        return D * (mar + mas)

    @staticmethod
    def _me(T, alt, Ta, ws, wa, D, Pa, rh):
        Tf = 0.5 * (T + Ta)
        Td = T - Ta
        vm = air.IEEE.volumic_mass(Tf, alt)
        cc = ieee.ConvectiveCooling._value_forced(Tf, Td, vm, ws, D, wa)
        h = cc / (np.pi * D * Td)
        k = 18.015 / 28.9647
        pm = np.pi * D
        cp = thermodynamics.Air.heat_capacity(T=air.kelvin(Tf))
        ec = thermodynamics.Water.vapor_pressure(T=air.kelvin(T))
        ea = thermodynamics.Water.vapor_pressure(T=air.kelvin(Ta))
        me = pm * h * k * (ec - rh * ea) / (cp * Pa)
        return np.where(Td != 0., me, np.zeros_like(T))

    @staticmethod
    def _mass_flux(T, alt, Ta, ws, wa, D, Pa, rh, pr, ps):
        return np.minimum(PrecipitationCooling._ma(Ta, ws, D, pr, ps),
                          PrecipitationCooling._me(T, alt, Ta, ws, wa, D, Pa, rh))

    @staticmethod
    def _evap(T, alt, Ta, ws, wa, D, Pa, rh, pr, ps=0.):
        m = PrecipitationCooling._mass_flux(T, alt, Ta, ws, wa, D, Pa, rh, pr, ps)
        Le = thermodynamics.Water.heat_of_vaporization()
        cw = thermodynamics.Water.heat_capacity(T=air.kelvin(T))
        Tb = thermodynamics.Water.boiling_point(p=Pa)
        Tb = air.celsius(Tb)
        Te = np.minimum(T, Tb)
        Lf = thermodynamics.Ice.heat_of_fusion()
        ci = thermodynamics.Ice.heat_capacity()
        rr = m * (Le + cw * (Te - Ta))
        rs = m * (Le + cw * T + Lf * ci * Ta)
        return np.maximum(rr + 0. * rs, np.zeros_like(T))

    @staticmethod
    def _imp(T, Ta, ws, D, pr, ps=0.):
        cw = thermodynamics.Water.heat_capacity(T=air.kelvin(T))
        return 0.71 * cw * (T - Ta) * PrecipitationCooling._ma(Ta, ws, D, pr, ps)
