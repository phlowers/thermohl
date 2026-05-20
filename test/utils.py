# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import os.path

import pandas as pd


def get_cable_data(cable_name: str) -> dict:
    """Get cable/conductor data from file."""
    f = os.path.join("test", "functional_test", "cable_catalog.csv")
    df = pd.read_csv(f)
    if cable_name in df["conductor"].values:
        return df[df["conductor"] == cable_name].to_dict(orient="records")[0]
    else:
        raise ValueError(f"Conductor {cable_name} not found in file {f}.")
