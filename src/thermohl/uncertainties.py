# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

# mypy: ignore-errors
"""Tools to perform Monte Carlo simulations using the thermOHL steady solvers with uncertain input parameters."""

import warnings
from typing import Union, Tuple

import numpy as np
import pandas as pd

from thermohl.solver.enums.cable_location import CableLocation, CableLocationListLike
from thermohl.solver.enums.cable_type import CableTypeListLike

from thermohl.utils import depends_on_optional

try:
    import scipy
    from scipy.stats._distn_infrastructure import rv_continuous_frozen as frozen_dist
except ImportError:
    warnings.warn(
        "scipy is not installed. Some functions will not be available.",
        RuntimeWarning,
    )
    frozen_dist = object

from thermohl import distributions
from thermohl import solver
from thermohl import utils


def default_uncertainties() -> dict:
    """
    Get default parameters for uncertainties.

    Returns
    -------
    dict
        Dictionnary of default distributions and parameters.

    """
    return utils.add_default_uncertainties({}, warning=False)


def cummean(values: np.ndarray) -> np.ndarray:
    """Cumulative mean."""
    return np.cumsum(values) / (1 + np.array(range(len(values))))


def cumstd(values: np.ndarray) -> np.ndarray:
    """Cumulative std."""
    return np.sqrt(cummean(values**2) - cummean(values) ** 2)


@depends_on_optional("scipy")
def _get_dist(uncertainty_spec: dict, mean_value: float) -> frozen_dist:
    """Get distribution
    -- based on parameters in dict uncertainty_spec and with mean mean_value"""
    mean = mean_value
    # set std
    standard_deviation = uncertainty_spec["std"]
    if uncertainty_spec["relative_std"]:
        standard_deviation *= mean
    # if std is 0., return uniform
    if standard_deviation == 0.0:
        return scipy.stats.uniform(mean, mean)

    # select ditribution
    if uncertainty_spec["dist"] == "truncnorm":
        lower_bound, upper_bound = uncertainty_spec["min"], uncertainty_spec["max"]
        dist = distributions.truncnorm(
            lower_bound, upper_bound, mean, standard_deviation
        )
    elif uncertainty_spec["dist"] == "vonmises":
        dist = distributions.vonmises(np.deg2rad(mean), np.deg2rad(standard_deviation))
    elif uncertainty_spec["dist"] == "wrapnorm":
        dist = distributions.wrapnorm(np.deg2rad(mean), np.deg2rad(standard_deviation))
    else:
        raise ValueError("Dist keyword not supported")

    return dist


@depends_on_optional("scipy")
def _generate_samples(
    input_params: dict,
    index: int,
    uncertainty_spec: dict,
    num_samples: int,
    include_check: bool = False,
) -> Union[dict, Tuple[dict, pd.DataFrame]]:
    """
    Generate random samples for all input parameters affected with a probability distribution.
    """

    # sample dict
    samples_dict = {}
    # check dataframe
    if include_check:
        columns = [
            "key",
            "dist",
            "mean",
            "std",
            "min",
            "max",
            "s_mean",
            "s_std",
            "s_min",
            "s_max",
            "circular",
        ]
        check_table = pd.DataFrame(
            columns=columns, data=np.zeros((len(input_params), len(columns))) * np.nan
        )
        check_table.loc[:, "circular"] = False

    # loop on dict
    for row_index, key in enumerate(input_params):
        if key not in uncertainty_spec.keys():
            continue
        dist = uncertainty_spec[key]["dist"]
        mean_value = input_params[key][index]
        if dist is None or np.isnan(mean_value):
            sample = mean_value * np.ones((num_samples,), dtype=type(mean_value))
        else:
            sample = _get_dist(uncertainty_spec[key], mean_value).rvs(num_samples)
            if dist == "vonmises" or dist == "wrapnorm":
                sample = np.rad2deg(sample) % 360.0
        samples_dict[key] = sample

        if include_check:
            check_table.loc[row_index, "key"] = key
            check_table.loc[row_index, "dist"] = dist
            check_table.loc[row_index, "mean"] = mean_value
            if "std" in uncertainty_spec[key].keys():
                check_table.loc[row_index, "std"] = uncertainty_spec[key]["std"]
                if uncertainty_spec[key]["relative_std"]:
                    check_table.loc[row_index, "std"] *= mean_value
            if "min" in uncertainty_spec[key].keys():
                check_table.loc[row_index, "min"] = uncertainty_spec[key]["min"]
            if "max" in uncertainty_spec[key].keys():
                check_table.loc[row_index, "max"] = uncertainty_spec[key]["max"]
            if dist in ["vonmises", "wrapnorm"]:
                check_table.loc[row_index, "s_mean"] = scipy.stats.circmean(
                    sample, high=360.0, low=0.0
                )
                check_table.loc[row_index, "s_std"] = scipy.stats.circstd(
                    sample, high=360.0, low=0.0
                )
                check_table.loc[row_index, "circular"] = True
            else:
                check_table.loc[row_index, "s_mean"] = sample.mean()
                check_table.loc[row_index, "s_std"] = sample.std()
            check_table.loc[row_index, "s_min"] = sample.min()
            check_table.loc[row_index, "s_max"] = sample.max()

    if include_check:
        return samples_dict, check_table

    return samples_dict


def _rdict(
    mode: str,
    target: CableLocation,
    include_surface: bool,
    include_core: bool,
    include_average: bool,
) -> dict:
    """Code factorization"""
    if mode == "temperature":
        rdc = dict(
            return_core=include_core,
            return_avg=include_average,
            return_power=False,
        )
    elif mode == "intensity":
        rdc = dict(
            target=target,
            return_core=include_core,
            return_avg=include_average,
            return_surf=include_surface,
            return_power=False,
        )
    else:
        raise ValueError("")
    return rdc


def _compute(
    mode: str,
    solver_instance: solver.Solver,
    target_temp: Union[float, np.ndarray],
    return_config: dict,
):
    """Code factorization"""
    if mode == "temperature":
        result = solver_instance.steady_temperature(**return_config)
    elif mode == "intensity":
        result = solver_instance.steady_intensity(target_temp, **return_config)
    else:
        raise ValueError()
    return result


def _steady_uncertainties(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    target_label: CableLocation,
    uncertainties: dict,
    num_samples: int,
    include_surface: bool,
    include_core: bool,
    include_average: bool,
    return_raw: bool,
    mode: str = "temperature",
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """Code factorization"""
    # return dict
    return_config = _rdict(
        mode, target_label, include_surface, include_core, include_average
    )

    # save solver dict
    saved_args = solver_instance.dc

    #  all to max_len size
    num_entries = utils.dict_max_len(solver_instance.dc)
    input_params = utils.extend_to_max_len(solver_instance.dc, num_entries)
    max_temp_vector = target_max_temp * np.ones(
        num_entries,
    )

    # add missing uncertainties parameters
    uncertainty_spec = utils.add_default_uncertainties(uncertainties)

    # init outputs
    raw_results = []
    solver_instance.dc = solver.default_values()
    sample_result = _compute(mode, solver_instance, 99.0, return_config)
    columns = []
    for column_name in sample_result.columns:
        columns.append(column_name + "_mean")
        columns.append(column_name + "_std")
    stats_df = pd.DataFrame(data=np.zeros((num_entries, len(columns))), columns=columns)

    # for each entry, generate sample then compute
    for index in range(num_entries):
        solver_instance.dc = _generate_samples(
            input_params, index, uncertainty_spec, num_samples, include_check=False
        )
        result = _compute(mode, solver_instance, max_temp_vector[index], return_config)
        if return_raw:
            raw_results.append(result)
        mean_values = result.mean()
        std_values = result.std()
        for column_name in result.columns:
            stats_df.loc[index, column_name + "_mean"] = mean_values[column_name]
            stats_df.loc[index, column_name + "_std"] = std_values[column_name]

    # restore solver dict
    solver_instance.dc = saved_args

    if return_raw:
        return stats_df, raw_results
    else:
        return stats_df


def temperature(
    solver_instance: solver.Solver,
    uncertainties: dict = {},
    num_samples: int = 4999,
    return_core: bool = False,
    return_avg: bool = False,
    return_raw: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """
    Perform Monte Carlo simulation using the steady temperature solver.
    """
    return _steady_uncertainties(
        solver_instance,
        np.nan,
        None,
        uncertainties,
        num_samples,
        None,
        return_core,
        return_avg,
        return_raw,
        mode="temperature",
    )


def intensity(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    target_label: CableLocation = CableLocation.SURFACE,
    cable_type: CableTypeListLike = None,
    uncertainties: dict = {},
    num_samples: int = 4999,
    return_core: bool = False,
    return_avg: bool = False,
    return_surf: bool = False,
    return_raw: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, list]]:
    """
    Perform Monte Carlo simulation using the steady intensity solver.
    """
    target_label = solver.infer_target_label(target_label, cable_type)
    return _steady_uncertainties(
        solver_instance,
        target_max_temp,
        target_label,
        uncertainties,
        num_samples,
        return_surf,
        return_core,
        return_avg,
        return_raw,
        mode="intensity",
    )


def _diff_method(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    target_label: str,
    uncertainties: dict,
    quantile: float = 0.95,
    return_surf: bool = False,
    return_core: bool = False,
    return_avg: bool = False,
    perturbation_step: float = 1.0e-06,
    mode: str = "temperature",
) -> pd.DataFrame:
    """."""
    # return dict
    return_config = _rdict(
        mode,
        target_label,
        include_surface=return_surf,
        include_core=return_core,
        include_average=return_avg,
    )

    # save solver dict
    saved_args = solver_instance.dc

    #  all to max_len size
    num_entries = utils.dict_max_len(solver_instance.dc)
    input_params = utils.extend_to_max_len(solver_instance.dc, num_entries)

    # add missing uncertainties parameters
    uncertainty_spec = utils.add_default_uncertainties(uncertainties)

    baseline = _compute(mode, solver_instance, target_max_temp, return_config)
    delta_results = baseline * 0.0
    for key in input_params:
        if key not in uncertainty_spec.keys() or uncertainty_spec[key]["dist"] is None:
            continue
        if np.all(np.isnan(input_params[key])):
            continue

        mean_values = input_params[key]
        delta_param = np.zeros_like(mean_values)
        for index in range(num_entries):
            if np.isnan(mean_values[index]):
                delta_param[index] = 0.0
            else:
                dist = _get_dist(uncertainty_spec[key], mean_values[index])
                try:
                    delta_param[index] = (
                        0.5
                        * np.diff(
                            dist.ppf([0.5 * (1 - quantile), 0.5 * (1 + quantile)])
                        )[0]
                    )
                except ValueError:
                    delta_param[index] = 0.0
                if np.isnan(delta_param[index]):
                    delta_param[index] = 0.0
        solver_instance.dc[key] = mean_values + perturbation_step
        perturbed = _compute(mode, solver_instance, target_max_temp, return_config)
        solver_instance.dc[key] = mean_values
        delta_output = np.abs(perturbed - baseline) / perturbation_step
        for column_name in delta_output.columns:
            delta_output.loc[:, column_name] *= delta_param
        delta_results += delta_output
        if np.any(delta_output.isna()):
            print("Nans with key %s" % (key,))

    results = pd.DataFrame()
    for column_name in baseline.columns:
        results.loc[:, column_name] = baseline.loc[:, column_name]
        results.loc[:, column_name + "_delta"] = delta_results.loc[:, column_name]

    # restore solver dict
    solver_instance.dc = saved_args

    return results


def temperature_diff(
    solver_instance: solver.Solver,
    uncertainties: dict,
    quantile: float = 0.95,
    return_core: bool = False,
    return_avg: bool = False,
    perturbation_step: float = 1.0e-06,
) -> pd.DataFrame:
    """."""
    return _diff_method(
        solver_instance,
        np.nan,
        None,
        uncertainties,
        quantile=quantile,
        return_core=return_core,
        return_avg=return_avg,
        perturbation_step=perturbation_step,
        mode="temperature",
    )


def intensity_diff(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    uncertainties: dict,
    target_label: str = None,
    cable_type: CableTypeListLike = None,
    quantile: float = 0.95,
    return_surf: bool = False,
    return_core: bool = False,
    return_avg: bool = False,
    perturbation_step: float = 1.0e-06,
) -> pd.DataFrame:
    """."""
    target_label = solver.infer_target_label(target_label, cable_type)
    return _diff_method(
        solver_instance,
        target_max_temp,
        target_label,
        uncertainties,
        quantile=quantile,
        return_core=return_core,
        return_avg=return_avg,
        return_surf=return_surf,
        perturbation_step=perturbation_step,
        mode="intensity",
    )


def sensitivity(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    uncertainties: dict,
    num_samples: int,
    include_surface: bool,
    include_core: bool,
    include_average: bool,
    target_label: CableLocationListLike = None,
    cable_type: CableTypeListLike = None,
    mode: str = "temperature",
) -> Tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """
    Perform a sensitivity analysis with Sobol indices (first order and total indices) using the Monte Carlo method.
    """
    target_label = solver.infer_target_label(target_label, cable_type)
    # return dict
    return_config = _rdict(
        mode, target_label, include_surface, include_core, include_average
    )

    # save solver dict
    saved_args = solver_instance.dc

    #  all to max_len size
    num_entries = utils.dict_max_len(solver_instance.dc)
    input_params = utils.extend_to_max_len(solver_instance.dc, num_entries)
    max_temp_vector = target_max_temp * np.ones(
        num_entries,
    )

    # add missing uncertainties parameters
    uncertainty_spec = utils.add_default_uncertainties(uncertainties)

    # init outputs
    first_order_list = []
    total_order_list = []

    # for each entry, compute
    for index in range(num_entries):
        # first sample
        samples_a = _generate_samples(
            input_params, index, uncertainty_spec, num_samples, include_check=False
        )
        solver_instance.dc = samples_a
        result_a = _compute(
            mode, solver_instance, max_temp_vector[index], return_config
        )

        # second sample
        samples_b = _generate_samples(
            input_params, index, uncertainty_spec, num_samples, include_check=False
        )
        solver_instance.dc = samples_b
        result_b = _compute(
            mode, solver_instance, max_temp_vector[index], return_config
        )

        pqs = ((result_a - result_b) ** 2).sum()

        #
        first_order = pd.DataFrame(
            columns=["var"] + result_a.columns.tolist(),
            data=np.zeros((len(uncertainty_spec), 1 + len(result_a.columns))),
        )
        first_order.loc[:, "var"] = uncertainty_spec.keys()
        total_order = pd.DataFrame(
            columns=["var"] + result_a.columns.tolist(),
            data=np.zeros((len(uncertainty_spec), 1 + len(result_a.columns))),
        )
        total_order.loc[:, "var"] = uncertainty_spec.keys()

        # mix samples, run and compute 1st and total indexes
        for row_index, key in enumerate(uncertainty_spec):
            solver_instance.dc = samples_a.copy()
            solver_instance.dc[key] = samples_b[key]
            result_a_mix = _compute(
                mode, solver_instance, max_temp_vector[index], return_config
            )

            solver_instance.dc = samples_b.copy()
            solver_instance.dc[key] = samples_a[key]
            result_b_mix = _compute(
                mode, solver_instance, max_temp_vector[index], return_config
            )

            denom = pqs + ((result_a_mix - result_b_mix) ** 2).sum()
            first_order.iloc[row_index, 1:] = (
                2.0
                * ((result_b_mix - result_b) * (result_a - result_a_mix)).sum()
                / denom
            )
            total_order.iloc[row_index, 1:] = (
                (result_b - result_b_mix) ** 2 + (result_a - result_a_mix) ** 2
            ).sum() / denom

        first_order_list.append(first_order)
        total_order_list.append(total_order)

    # restore solver dict
    solver_instance.dc = saved_args

    return first_order_list, total_order_list
