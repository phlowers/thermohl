"""Misc. utility code for thermohl project."""
import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml


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
