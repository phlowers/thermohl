# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


class RadiationIncompatibleWithParametersError(ValueError):
    """Raised when attempting to estimate a nebulosity for a radiation
    which is not compatible with the other parameters (solar_altitude, datetime_utc...).

    This means the described situation is physically impossible."""

    def __init__(self, *args, **kwargs):
        self.message = (
            "It's impossible to have this radiation level with given parameters."
        )
        super().__init__(*args, **kwargs)
