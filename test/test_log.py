# SPDX-FileCopyrightText: 2025 RTE (https://www.rte-france.com)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from importlib import reload

import thermohl
import thermohl.utils


def test_log(caplog) -> None:
    """Test logger initialization and helper function."""
    caplog.set_level(logging.INFO)
    logging.info("Test log is working")
    assert "Test log is working" in caplog.text

    with caplog.at_level(logging.DEBUG):
        reload(thermohl)  # noqa
        assert "Thermohl package initialized." in caplog.text

    with caplog.at_level(logging.DEBUG, logger="thermohl.utils"):
        thermohl.utils.add_stderr_logger(logging.DEBUG)
    assert "Added a stderr logging handler to logger: thermohl" in caplog.text
