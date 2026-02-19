<!--
SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
SPDX-License-Identifier: MPL-2.0
-->

# Migration Guide: Variable Renaming

This guide helps you migrate your code from the old version with short variable names, to the new version which introduces clearer, more descriptive variable names throughout the ThermOHL library. In this new version, variables are renamed to make the codebase more readable and maintainable. All abbreviated parameter names have been replaced with self-documenting names that include units where applicable.

## Quick Migration Steps

1. Update all parameter names in your dictionaries (see tables below)
2. Update function parameter names if you use keyword arguments
3. Update configuration files (YAML files with default values/uncertainties)
4. Run your tests to ensure everything works correctly

## Parameter Name Changes

### Location and Time Parameters

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `lat` | `latitude` | Latitude | degrees |
| `lon` | `longitude` or `longitude` depending on context | Longitude | degrees or rad |
| `alt` | `altitude` | Altitude | meters |
| `azm` | `azimuth` | Azimuth angle | degrees |

**Example:**
```python
# Before
params = {
    'lat': 45.0,
    'lon': 2.5,
    'alt': 100.0,
    'azm': 90.0
}

# After
params = {
    'latitude': 45.0,
    'longitude': 2.5,
    'altitude': 100.0,
    'azimuth': 90.0
}
```

### Weather and Environmental Parameters

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `Ta` | `ambient_temperature` | Ambient temperature | °C |
| `Pa` | `ambient_pressure` | Ambient pressure | Pa |
| `rh` | `relative_humidity` | Relative humidity | dimensionless [0-1] |
| `pr` | `precipitation_rate` | Precipitation rate | m/s |
| `ws` | `wind_speed` | Wind speed | m/s |
| `wa` | `wind_angle` | Wind angle | degrees |
| `al` | `albedo` | Albedo | dimensionless |
| `tb` and `trb` | `turbidity` | Turbidity coefficient | dimensionless [0-1] |

**Example:**
```python
# Before
weather = {
    'Ta': 20.0,
    'Pa': 1.0e5,
    'rh': 0.8,
    'ws': 2.0,
    'wa': 45.0,
    'al': 0.8,
    'tb': 0.1
}

# After
weather = {
    'ambient_temperature': 20.0,
    'ambient_pressure': 1.0e5,
    'relative_humidity': 0.8,
    'wind_speed': 2.0,
    'wind_angle': 45.0,
    'albedo': 0.8,
    'turbidity': 0.1
}
```

### Cable Physical Parameters

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `m` | `linear_mass` | Mass per unit length | kg/m |
| `d` | `core_diameter` | Core diameter | m |
| `D` | `outer_diameter` | Outer/external diameter | m |
| `a` | `core_area` | Core cross-sectional area | m² |
| `A` | `outer_area` | Outer cross-sectional area | m² |
| `R` | `roughness_ratio` | Surface roughness | dimensionless |
| `l` | `radial_thermal_conductivity` | Radial thermal conductivity | W/(m·K) |
| `c` | `heat_capacity` | Specific heat capacity | J/(kg·K) |

**Example:**
```python
# Before
cable = {
    'm': 1.5,
    'd': 1.9e-2,
    'D': 3.0e-2,
    'a': 2.84e-4,
    'A': 7.07e-4,
    'R': 4.0e-2,
    'l': 1.0,
    'c': 500.0
}

# After
cable = {
    'linear_mass': 1.5,
    'core_diameter': 1.9e-2,
    'outer_diameter': 3.0e-2,
    'core_area': 2.84e-4,
    'outer_area': 7.07e-4,
    'roughness_ratio': 4.0e-2,
    'radial_thermal_conductivity': 1.0,
    'heat_capacity': 500.0
}
```

### Thermal Radiation Parameters

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `alpha` | `solar_absorptivity` | Solar absorption coefficient | dimensionless |
| `epsilon` | `emissivity` | Emissivity | dimensionless |
| `srad` | `precomputed_solar_radiation` | Direct solar radiation (precomputed) | W/m² |
| `Qs` | `measured_solar_irradiance` | Measured solar irradiance | W/m² |
| `sigma` | `stefan_boltzmann_constant` | Stefan-Boltzmann constant | W/m²/$K^4$ |
| `x` | `solar_altitude` | Solar altitude | degrees |

**Example:**
```python
# Before
optical = {
    'alpha': 0.5,
    'epsilon': 0.5,
    'srad': np.nan  # or measured value
}

# After
optical = {
    'solar_absorptivity': 0.5,
    'emissivity': 0.5,
    'precomputed_solar_radiation': np.nan  # or measured value
}
```

### Electrical Parameters

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `transit` | `transit` | Transit current intensity | A |
| `RDC20` | `linear_resistance_dc_20c` | DC resistance per unit length at 20°C | Ω/m |
| `km` | `magnetic_coeff` | Magnetic effects coefficient | dimensionless |
| `ki` | `magnetic_coeff_per_a` | Magnetic effects coefficient (per ampere) | A⁻¹ |
| `kl` | `temperature_coeff_linear` | Linear temperature coefficient | K⁻¹ |
| `kq` | `temperature_coeff_quadratic` | Quadratic temperature coefficient | K⁻² |
| `RDCHigh` | `linear_resistance_temp_high` | DC resistance at high temperature | Ω/m |
| `RDCLow` | `linear_resistance_temp_low` | DC resistance at low temperature | Ω/m |
| `THigh` | `temp_high` | High reference temperature | °C |
| `TLow` | `temp_low` | Low reference temperature | °C |
| `T20` | `reference_temperature` | Reference temperature, default is 20 °C | °C |

**Example:**
```python
# Before
electrical = {
    'transit': 1000.0,
    'RDC20': 2.5e-5,
    'km': 1.006,
    'ki': 0.016,
    'kl': 3.8e-3,
    'kq': 8.0e-7,
    'RDCHigh': 3.05e-5,
    'RDCLow': 2.66e-5,
    'THigh': 60.0,
    'TLow': 20.0
}

# After
electrical = {
    'transit': 1000.0,
    'linear_resistance_dc_20c': 2.5e-5,
    'magnetic_coeff': 1.006,
    'magnetic_coeff_per_a': 0.016,
    'temperature_coeff_linear': 3.8e-3,
    'temperature_coeff_quadratic': 8.0e-7,
    'linear_resistance_temp_high': 3.05e-5,
    'linear_resistance_temp_low': 2.66e-5,
    'temp_high': 60.0,
    'temp_low': 20.0
}
```

### Misc.

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `Tc` | `conductor_temperature` or `air_temperature` depending on context | Conductor temperature or air temperature | °C |
| `Tf` | `film_temperature` | Air film temperature | °C |
| `Td` | `temperature_delta` | Air film temperature | °C |
| `T` | `conductor_temperature` or `max_conductor_temperature` depending on context | Conductor temperature or maximum conductor temperature | °C |
| `ts` | `surface_temperature` | Conductor surface temperature | °C |
| `tc` | `core_temperature` | Conductor core temperature | °C |
| `dT` | `temperature_increment` | Temperature increment | °C |
| `nu` | `kinematic_viscosity` | Air film temperature | dimensionless |
| `time` | `time` | Time | s |

## Function Parameter Changes

Several function signatures have been updated with clearer parameter names:

### `utils.py` Functions

```python
# Before
def _dict_completion(dat: dict, filename: str, check: bool = True, warning: bool = False)
def add_default_parameters(dat: dict, warning: bool = False)
def add_default_uncertainties(dat: dict, warning: bool = False)
def bisect_v(fun, a, b, shape, tol=1e-6, maxiter=128, print_err=False)

# After
def _dict_completion(params: dict, filename: str, validate_types: bool = True, warning: bool = False)
def add_default_parameters(params: dict, warning: bool = False)
def add_default_uncertainties(params: dict, warning: bool = False)
def bisect_v(func, lower_bound, upper_bound, output_shape, tolerance=1e-6, max_iterations=128, print_error=False)
```

### `distributions.py` Functions

```python
# Before
def truncnorm(a, b, mu, sigma, err_mu=1e-3, err_sigma=1e-2, rel_err=True)
def _phi(x: float)
def _psi(x: float)
def _truncnorm_header(a: float, b: float, mu: float, sigma: float)
def _truncnorm_mean_std(a: float, b: float, mu: float, sigma: float)
def wrapnorm(mu: float, sigma: float)
def _vonmises_kappa(sigma: float)
def vonmises(mu: float, sigma: float)

# After
def truncnorm(lower_bound, upper_bound, mean, standard_deviation, 
              mean_tolerance=1e-3, std_tolerance=1e-2, relative_tolerance=True)
def _phi(value: float)
def _psi(value: float)
def _truncnorm_header(lower_bound: float, upper_bound: float, mean: float, standard_deviation: float)
def _truncnorm_mean_std(lower_bound: float, upper_bound: float, mean: float, standard_deviation: float)
def wrapnorm(mean: float, standard_deviation: float)
def _vonmises_kappa(standard_deviation: float)
def vonmises(mean: float, standard_deviation: float)
```

### `solver/base.py` Args Class

```python
# Before
def __init__(self, dic: Optional[dict[str, Any]] = None)

# After
def __init__(self, input_dict: Optional[dict[str, Any]] = None)
```

### `src/thermohl/sun.py` Function

```python
# Before
def utc2solar_hour(hour, minute=0.0, second=0.0, lon=0.0)
def hour_angle(hour: floatArrayLike, minute: floatArrayLike = 0.0, second: floatArrayLike = 0.0)
def solar_declination(month: intArrayLike, day: intArrayLike)
def solar_altitude(
    lat: floatArrayLike,
    month: intArrayLike,
    day: intArrayLike,
    hour: floatArrayLike,
    minute: floatArrayLike = 0.0,
    second: floatArrayLike = 0.0,
)
def solar_azimuth(
    lat: floatArrayLike,
    month: intArrayLike,
    day: intArrayLike,
    hour: floatArrayLike,
    minute: floatArrayLike = 0.0,
    second: floatArrayLike = 0.0,
)

# After
def utc2solar_hour(
    utc_hour: floatArrayLike,
    utc_minute: floatArrayLike = 0.0,
    utc_second: floatArrayLike = 0.0,
    longitude: floatArrayLike = 0.0,
)
def hour_angle(
    solar_hour: floatArrayLike,
    solar_minute: floatArrayLike = 0.0,
    solar_second: floatArrayLike = 0.0,
)
def solar_declination(
    month_index: intArrayLike, day_of_month: intArrayLike
)
def solar_altitude(
    latitude: floatArrayLike,
    month_index: intArrayLike,
    day_of_month: intArrayLike,
    solar_hour: floatArrayLike,
    solar_minute: floatArrayLike = 0.0,
    solar_second: floatArrayLike = 0.0,
)
def solar_azimuth(
    latitude: floatArrayLike,
    month_index: intArrayLike,
    day_of_month: intArrayLike,
    solar_hour: floatArrayLike,
    solar_minute: floatArrayLike = 0.0,
    solar_second: floatArrayLike = 0.0,
)
```

### `src/thermohl/uncertainties` Functions

```python
# Before
def cummean(x: np.ndarray)
def cumstd(x: np.ndarray)
def _get_dist(du: dict, mean: float)
def _generate_samples(
    dc: dict, i: int, du: dict, ns: int, check: bool = False
)
def _rdict(
    mode: str, target: str, return_surf: bool, return_core: bool, return_avg: bool
)
def _compute(mode: str, s: solver.Solver, tmx: Union[float, np.ndarray], rdc: dict)
def _steady_uncertainties(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str,
    u: dict,
    ns: int,
    return_surf: bool,
    return_core: bool,
    return_avg: bool,
    return_raw: bool,
    mode: str = "temperature",
)
def temperature(
    s: solver.Solver,
    u: dict = {},
    ns: int = 4999,
    return_core: bool = False,
    return_avg: bool = False,
    return_raw: bool = False,
)
def intensity(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str = "surf",
    u: dict = {},
    ns: int = 4999,
    return_core: bool = False,
    return_avg: bool = False,
    return_surf: bool = False,
)
def _diff_method(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str,
    u: dict,
    q: float = 0.95,
    return_surf: bool = False,
    return_core: bool = False,
    return_avg: bool = False,
    ep: float = 1.0e-06,
)
def temperature_diff(
    s: solver.Solver,
    u: dict,
    q: float = 0.95,
    return_core: bool = False,
    return_avg: bool = False,
    ep: float = 1.0e-06,
)
def intensity_diff(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    target: str,
    u: dict,
    q: float = 0.95,
    return_core: bool = False,
    return_avg: bool = False,
    return_surf: bool = False,
    ep: float = 1.0e-06,
)
def sensitivity(
    s: solver.Solver,
    tmax: Union[float, np.ndarray],
    u: dict,
    ns: int,
    target: str,
    return_surf: bool,
    return_core: bool,
    return_avg: bool,
    mode: str = "temperature",
)

# After
def cummean(values: np.ndarray)
def cumstd(values: np.ndarray)
def _get_dist(uncertainty_spec: dict, mean_value: float)
def _generate_samples(
    input_params: dict,
    index: int,
    uncertainty_spec: dict,
    num_samples: int,
    include_check: bool = False,
)
def _rdict(
    mode: str,
    target: str,
    include_surface: bool,
    include_core: bool,
    include_average: bool,
)
def _compute(
    mode: str,
    solver_instance: solver.Solver,
    target_temp: Union[float, np.ndarray],
    return_config: dict,
)
def _steady_uncertainties(

    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    target_label: str,
    uncertainties: dict,
    num_samples: int,
    include_surface: bool,
    include_core: bool,
    include_average: bool,
)
def temperature(
    solver_instance: solver.Solver,
    uncertainties: dict = {},
    num_samples: int = 4999,
)
def intensity(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    target_label: str = "surf",
    uncertainties: dict = {},
    num_samples: int = 4999,
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
)
def temperature_diff(
    solver_instance: solver.Solver,
    uncertainties: dict,
    quantile: float = 0.95,
    return_core: bool = False,
    return_avg: bool = False,
    perturbation_step: float = 1.0e-06,
)
def intensity_diff(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    target_label: str,
    uncertainties: dict,
    quantile: float = 0.95,
    return_surf: bool = False,
    return_core: bool = False,
    return_avg: bool = False,
    perturbation_step: float = 1.0e-06,
)
def sensitivity(
    solver_instance: solver.Solver,
    target_max_temp: Union[float, np.ndarray],
    uncertainties: dict,
    num_samples: int,
    target_label: str,
    include_surface: bool,
    include_core: bool,
    include_average: bool,
    mode: str = "temperature",
)
```
### `src/thermohl/solver/slv3t.py` Function

```python
# Before
def _profile_mom(ts: float, tc: float, r: floatArrayLike, re: float)

# After
def _profile_mom(
    surface_temperature: float,
    core_temperature: float,
    radius: floatArrayLike,
    outer_radius: float,
)
```

## Variable and Attributes Changes

Several attributes have been renamed, please adapt your code if you use them directly. The former and new names are listed in the tables below.

### Global Variables Change

Following global variables have been renamed:

| Old Name | New Name |
-----------|-----------
| `power_term._dT` | `power_term_dT._TEMP_INCREMENT` |
| `_T0` | `_STD_TEMP_K` |
| `_p0` | `_STD_PRESSURE_PA` |
| `_kb` | `_BOLTZMANN_CONSTANT` |
| `_Na` | `_AVOGADRO_NUMBER` |
| `_R` | `_GAS_CONSTANT` |

### Classes ConvectiveCooling

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `alt` | `altitude` | Altitude | meters |
| `Ta` | `ambient_temperature` | Ambient temperature | °C |
| `ws` | `wind_speed` | Wind speed | m/s |
| `D` | `outer_diameter` | Cable external diameter | m |
| `R` | `roughness_ratio` | Cable roughness | dimensionless |
| `g` | `gravity` | Gravitational acceleration | m/s² |
| `da` | `attack_angle` | Attack angle | m/s² |
| `rho` | `air_density` | Air density | dimensionless |
| `mu` | `dynamic_viscosity` | Dynamic viscosity |  |
| `lambda_` | `thermal_conductivity` | Thermal conductivity |  |

### Classes JouleHeating

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `transit` | `transit` | Transit_intensity | A |
| `km` | `magnetic_coeff` | Coefficient for magnetic effects | dimensionless |
| `kl` | `temperature_coeff_linear` | Linear resistance augmentation with temperature | K-1 |
| `kem` | `magnetic_coeff` | Coefficient for magnetic effects | |
| `kq` | `temp_coeff_quadratic` | Quadratic resistance augmentation with temperature | K-2 |
| `T20` | `reference_temperature` | Reference temperature | °C |
| `TLow` | `temp_low` | Temperature for linear_resistance_temp_low measurement | °C |
| `THigh` | `temp_high` | Temperature for linear_resistance_temp_high measurement | °C |
| `RDC20` | `linear_resistance_dc_20c` | Coefficient for magnetic effects | ohm/m |
| `RDCLow` | `linear_resistance_temp_low` | Electric resistance per unit length at temp_low | ohm/m |
| `RDCHigh` | `linear_resistance_temp_high` | Electric resistance per unit length at temp_high | ohm/m |
| `c` | `temp_coeff_linear` | Temperature coefficients | |
| `D` | `outer_diameter` | Cable external diameter | m |
| `d` | `core_diameter` | Cable core diameter | m |
| `f` | `frequency` | Current frequency | Hz |

### Classes SolarHeating

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `alpha` | `solar_absorptivity` | Solar absorption coefficient | |
| `srad` | `solar_irradiance` | Solar irradiance | |
| `D` | `outer_diameter` | Cable external diameter | m |

### Classes RadiativeCooling

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `Ta` | `ambient_temperature` or `ambient_temperature` depending on implementation | Ambient_temperature | °C or K |
| `D` | `outer_diameter` | Cable external diameter | m |
| `epsilon` | `emissivity` | Emissivity | dimensionless |
| `sigma` | `stefan_boltzmann` | Stefan Boltzmann constant | W/m²/K-4 |
| `zerok` | `kelvin_offset` | Kelvin offset | K |

### Solver classes

| Old Name | New Name | Description | Unit |
|----------|----------|-------------|------|
| `Names.transit` | `Names.transit` | Transit intensity | A |
| `jh` | `joule_heating` | Joule heating | |
| `sh` | `solar_heating` | Solar heating | |
| `cc` | `convective_cooling` | Convective cooling | |
| `rc` | `radiative_cooling` | Radiative cooling | |
| `pc` | `precipitation_cooling` | Precipitation cooling | |
| `mgc` | `morgan_coefficients` | Morgan coefficients | |

### WrappedNormal classe

| Old Name | New Name | Description |
|----------|----------|-------------|
| `lwrb` | `lower_bound` | Lower bound |
| `uprb` | `upper_bound` | Upper bound |
| `mu` | `mean_value` | Mean value |
| `sigma` | `standard_deviation` | Standard deviation |

## Transient Solver Method Changes

When calling transient temperature methods, update the parameter name:

```python
# Before
result = solver.transient_temperature(t, T0=initial_temp, transit=current_array)

# After
result = solver.transient_temperature(t, T0=initial_temp, transit=current_array)
```

## Complete Migration Example

Here's a complete example showing before and after:

```python
# ========== BEFORE ==========
import numpy as np
from thermohl.solver import slv1t

# Define parameters
params = dict(
    lat=45.0,
    lon=2.5,
    alt=100.0,
    azm=90.0,
    month=3,
    day=21,
    hour=12,
    Ta=20.0,
    Pa=1.0e5,
    ws=2.0,
    wa=45.0,
    transit=1500.0,
    m=1.5,
    D=3.0e-2,
    alpha=0.5,
    epsilon=0.5,
    RDC20=2.5e-5
)

# Create solver
solver = slv1t.Slv1tCigre(params)

# Compute temperature
df = solver.steady_temperature()

# Transient analysis
t = np.linspace(0, 3600, 361)
transit_profile = 1000.0 + 500.0 * np.sin(2 * np.pi * t / 3600)
result = solver.transient_temperature(t, T0=25.0, transit=transit_profile)


# ========== AFTER ==========
import numpy as np
from thermohl.solver import slv1t

# Define parameters
params = dict(
    latitude=45.0,
    longitude=2.5,
    altitude=100.0,
    azimuth=90.0,
    month=3,
    day=21,
    hour=12,
    ambient_temperature=20.0,
    ambient_pressure=1.0e5,
    wind_speed=2.0,
    wind_angle=45.0,
    transit=1500.0,
    linear_mass=1.5,
    outer_diameter=3.0e-2,
    solar_absorptivity=0.5,
    emissivity=0.5,
    linear_resistance_dc_20c=2.5e-5
)

# Create solver
solver = slv1t.Slv1tCigre(params)

# Compute temperature
df = solver.steady_temperature()

# Transient analysis
t = np.linspace(0, 3600, 361)
transit_profile = 1000.0 + 500.0 * np.sin(2 * np.pi * t / 3600)
result = solver.transient_temperature(t, T0=25.0, transit=transit_profile)
```

## YAML Configuration Files

If you have custom YAML configuration files for default values or uncertainties, update them as well:

```yaml
# Before
lat:
  dist: 'truncnorm'
  min: 35.
  max: 55.
  std: 4.4E-04
  relative_std: false
Ta:
  dist: 'truncnorm'
  min: -40.
  max: +50.
  std: 2.
  relative_std: false

# After
latitude:
  dist: 'truncnorm'
  min: 35.
  max: 55.
  std: 4.4E-04
  relative_std: false
ambient_temperature:
  dist: 'truncnorm'
  min: -40.
  max: +50.
  std: 2.
  relative_std: false
```

## Breaking Changes Summary

⚠️ **All changes are breaking changes** - the old parameter names are no longer supported. You must update:

1. All dictionaries passed to solver constructors
2. All keyword arguments when calling functions
3. All YAML configuration files
4. All references to `transit` in transient solver method calls → `transit`
5. All references to `srad` for solar radiation → `precomputed_solar_radiation`

## Validation Checklist

After migration, verify that:

- All parameter dictionaries use new names
- All solver instantiations work correctly
- All steady-state calculations produce expected results
- All transient calculations use `transit` parameter
- All custom YAML files are updated
- All tests pass
- Documentation references are updated

## Benefits of Migration

The new naming convention provides:

- **Self-documenting code**: Variable names clearly indicate what they represent
- **Unit clarity**: Many names now include units (e.g., `_deg`, `_ms`, `_m`, `_pa`)
- **Reduced ambiguity**: No more guessing what `d`, `D`, `a`, `A` mean
- **Better IDE support**: Autocomplete and IntelliSense work better with descriptive names
- **Easier onboarding**: New users can understand the code without constantly referring to documentation
