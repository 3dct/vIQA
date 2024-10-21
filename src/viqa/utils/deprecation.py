"""Module containing deprecation statements for vIQA package."""
# Authors
# -------
# Author: Lukas Behammer
# Research Center Wels
# University of Applied Sciences Upper Austria, 2023
# CT Research Group
#
# Modifications
# -------------
# Original code, 2024, Lukas Behammer
#
# License
# -------
# BSD-3-Clause License


class RemovedInViqa30Warning(DeprecationWarning):
    """Warn about features that will be removed in ViQa 3.0.x."""

    pass


class RemovedInViqa40Warning(PendingDeprecationWarning):
    """Warn about features that will be removed in ViQa 4.0.x."""

    pass


RemovedInNextVersionWarning = RemovedInViqa30Warning
RemovedInFutureVersionWarning = RemovedInViqa40Warning
