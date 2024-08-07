"""Warnings for deprecated features."""


class RemovedInViqa20Warning(DeprecationWarning):
    """Warn about features that will be removed in ViQa 2.0.x."""

    pass


class RemovedInViqa30Warning(PendingDeprecationWarning):
    """Warn about features that will be removed in ViQa 3.0.x."""

    pass


RemovedInNextVersionWarning = RemovedInViqa20Warning
