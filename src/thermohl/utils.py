"""Misc. utility code for thermohl project."""
import os
from typing import Union, Optional

import numpy as np
import pandas as pd
import yaml
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
    """Power term for radiative cooling."""

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


def _dict_completion(dat: dict, filename: str, check: bool = True, warning: bool = False) -> dict:
    """Complete input dict with values from file.

    Read dict stored in filename (yaml format) and for each key in it, add it
    to input dict dat if the key is not already in dat.

    Parameters
    ----------
    dat : dict
        Input dict with parameters for power terms.
    warning : bool, optional
        Print a message if a parameter is missing. The default is False.

    Returns
    -------
    dict
        Completed input dict if some parameters were missing.

    """
    fil = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    dfl = yaml.safe_load(open(fil, 'r'))
    for k in dfl.keys():
        if k not in dat.keys() or dat[k] is None:
            dat[k] = dfl[k]
            if warning:
                print('Added key %s from default parameters' % (k,))
        elif not isinstance(dat[k], int) and not isinstance(dat[k], float) and \
                not isinstance(dat[k], np.ndarray) and check:
            raise TypeError('element in input dict (key [%s]) must be int, float or numpy.ndarray' % (k,))
    return dat


def add_default_parameters(dat: dict, warning: bool = False) -> dict:
    """Add default parameters if there is missing input.

    Parameters
    ----------
    dat : dict
        Input dict with parameters for power terms.
    warning : bool, optional
        Print a message if a parameter is missing. The default is False.

    Returns
    -------
    dict
        Completed input dict if some parameters were missing.

    """
    fil = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_values.yaml')
    return _dict_completion(dat, fil, warning=warning)


def add_default_uncertainties(dat: dict, warning: bool = False) -> dict:
    """Add default uncertainty parameters if there is missing input.

    Parameters
    ----------
    dat : dict
        Input dict with parameters for power terms.
    warning : bool, optional
        Print a message if a parameter is missing. The default is False.

    Returns
    -------
    dict
        Completed input dict if some parameters were missing.

    """
    fil = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_uncertainties.yaml')
    return _dict_completion(dat, fil, check=False, warning=warning)


def df2dct(df: pd.DataFrame) -> dict:
    """Convert a pandas.DataFrame to a dictionary.

    Would be an equivalent to df.to_dict(orient='numpy.ndarray') if it existed.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    dict
        DESCRIPTION.
    """
    q = df.to_dict(orient='list')
    for k in q.keys():
        q[k] = np.array(q[k])
    return q


def dict_max_len(dc: dict) -> int:
    """Get max length of all elements in a dict."""
    if len(dc) == 0:
        return 0
    n = 1
    for k in dc.keys():
        try:
            n = max(n, len(dc[k]))
        except TypeError:
            pass
    return n


def extend_to_max_len(dc: dict, n: Optional[int] = None) -> dict:
    """Put all elements in dc in size (n,)."""
    if n is None:
        n = dict_max_len(dc)
    dc2 = {}
    for k in dc.keys():
        if isinstance(dc[k], np.ndarray):
            t = dc[k].dtype
            c = len(dc[k]) == n
        else:
            t = type(dc[k])
            c = False
        if c:
            dc2[k] = dc[k][:]
        else:
            dc2[k] = dc[k] * np.ones((n,), dtype=t)
    return dc2
