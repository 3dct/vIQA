"""Warnings for deprecated features."""

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


class RemovedInViqa20Warning(DeprecationWarning):
    """Warn about features that will be removed in ViQa 2.0.x."""

    pass


class RemovedInViqa30Warning(PendingDeprecationWarning):
    """Warn about features that will be removed in ViQa 3.0.x."""

    pass


RemovedInNextVersionWarning = RemovedInViqa20Warning
