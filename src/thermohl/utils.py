# Copyright 2023 Eurobios Mews Labs
# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


"""Misc. utility code for thermohl project."""

import os

import numpy as np
import pandas as pd
import yaml


def _dict_completion(
    dat: dict, filename: str, check: bool = True, warning: bool = False
) -> dict:
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
    dfl = yaml.safe_load(open(fil, "r"))
    for k in dfl.keys():
        if k not in dat.keys() or dat[k] is None:
            dat[k] = dfl[k]
            if warning:
                print("Added key %s from default parameters" % (k,))
        elif (
            not isinstance(dat[k], int)
            and not isinstance(dat[k], float)
            and not isinstance(dat[k], np.ndarray)
            and check
        ):
            raise TypeError(
                "element in input dict (key [%s]) must be int, float or numpy.ndarray"
                % (k,)
            )
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
    fil = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "default_values.yaml"
    )
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
    fil = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "default_uncertainties.yaml"
    )
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
    q = df.to_dict(orient="list")
    for k in q.keys():
        if len(q[k]) > 1:
            q[k] = np.array(q[k])
        else:
            q[k] = q[k][0]
    return q
