"""Several models for different air properties with temperature dependency."""
import numpy as np
from typing import Union

_zerok = 273.15  # Temperature value used to convert temperatures from Celsius degrees to Kelvins.


def kelvin(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Celsius to Kelvin conversion.

    Parameters
    ----------
    Tc : float or numpy.ndarray
        Temperature in Celsius.

    Returns
    -------
    float or numpy.ndarray
        Temperature in Kelvin.

    """
    return Tc + _zerok


def celsius(Tk: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Kelvin to Celsius conversion.

    Parameters
    ----------
    Tk : float or numpy.ndarray
        Temperature in Kelvins.

    Returns
    -------
    float or numpy.ndarray
        Temperature in Celsius.

    """
    return Tk - _zerok


class Wikipedia:
    """`Wikipedia <https://fr.wikipedia.org/wiki/Air>`_ models."""

    @staticmethod
    def volumic_mass(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        r"""
        Compute air volumic mass.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius).
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        float or numpy.ndarray
             Volumic mass in kg.m\ :sup:`-3`\ .

        """
        Tk = kelvin(Tc)
        return 1.292 * _zerok * np.exp(-3.42E-02 * alt / Tk) / Tk

    @staticmethod
    def dynamic_viscosity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air dynamic viscosity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Dynamic viscosity in kg.m\ :sup:`-1`\ .s\ :sup:`-1`\ .

        """
        Tk = kelvin(Tc)
        return (8.8848E-15 * Tk**3 - 3.2398E-11 * Tk**2 +
                6.2657E-08 * Tk + 2.3543E-06)

    @staticmethod
    def kinematic_viscosity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air kinematic viscosity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Raises
        ------
        NotIpltementedError
            DESCRIPTION.

        Returns
        -------
        float or numpy.ndarray
             Kinematic viscosity in m\ :sup:`2`\ .s\ :sup:`-1`\ .

        """
        raise NotImplementedError()

    @staticmethod
    def thermal_conductivity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air thermal conductivity.

        The output is valid for input in [-150, 1300] range (in Celsius)

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Thermal conductivity in W.m\ :sup:`-1`\ .K\ :sup:`-1`\ .

        """
        Tk = kelvin(Tc)
        return (1.5207E-11 * Tk**3 - 4.8570E-08 * Tk**2 +
                1.0184E-04 * Tk - 3.9333E-04)


class CIGRE:
    """CIGRE models."""

    @staticmethod
    def volumic_mass(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        r"""Compute air volumic mass.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius).
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        float or numpy.ndarray
             Volumic mass in kg.m\ :sup:`-3`\ .

        """
        return 1.2925 * CIGRE.relative_density(Tc, alt)

    @staticmethod
    def relative_density(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        """Compute relative density, ie density ratio regarding density at zero altitude.

        This function has temperature and altitude as input for consistency
        regarding other functions in the module, but the temperature has no
        influence, only the altitude for this model.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius).
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.exp(-1.16E-04 * alt) * np.ones_like(Tc)

    @staticmethod
    def kinematic_viscosity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air kinematic viscosity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Kinematic viscosity in m\ :sup:`2`\ .s\ :sup:`-1`\ .

        """
        return 1.32E-05 + 9.5E-08 * Tc

    @staticmethod
    def dynamic_viscosity(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        r"""Compute air dynamic viscosity.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        float or numpy.ndarray
             Dynamic viscosity in kg.m\ :sup:`-1`\ .s\ :sup:`-1`\ .

        """
        return CIGRE.kinematic_viscosity(Tc) * CIGRE.volumic_mass(Tc, alt)

    @staticmethod
    def thermal_conductivity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air thermal conductivity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Thermal conductivity in W.m\ :sup:`-1`\ .K\ :sup:`-1`\ .

        """
        return 2.42E-02 + 7.2E-05 * Tc

    @staticmethod
    def prandtl(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute Prandtl number.

        The Prandtl number (Pr) is a dimensionless number, named after the German
        physicist Ludwig Prandtl, defined as the ratio of momentum diffusivity to
        thermal diffusivity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Prandtl number (no unit)

        """
        return 0.715 - 2.5E-04 * Tc


class IEEE:
    """IEEE models."""

    @staticmethod
    def volumic_mass(Tc: Union[float, np.ndarray], alt: Union[float, np.ndarray] = 0.) -> Union[float, np.ndarray]:
        r"""Compute air volumic mass.

        If both inputs are numpy arrays, they should have the same size.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius).
        alt : float or numpy.ndarray, optional
            Altitude above sea-level. The default is 0.

        Returns
        -------
        float or numpy.ndarray
             Volumic mass in kg.m\ :sup:`-3`\ .

        """
        return (1.293 - 1.525E-04 * alt + 6.379E-09 * alt**2) / (1. + 0.00367 * Tc)

    @staticmethod
    def dynamic_viscosity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air dynamic viscosity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Dynamic viscosity in kg.m\ :sup:`-1`\ .s\ :sup:`-1`\ .

        """
        return (1.458E-06 * kelvin(Tc)**1.5) / (Tc + 383.4)

    @staticmethod
    def thermal_conductivity(Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Compute air thermal conductivity.

        Parameters
        ----------
        Tc : float or numpy.ndarray
            Air temperature (in Celsius)

        Returns
        -------
        float or numpy.ndarray
             Thermal conductivity in W.m\ :sup:`-1`\ .K\ :sup:`-1`\ .

        """
        return 2.424E-02 + 7.477E-05 * Tc - 4.407E-09 * Tc**2
