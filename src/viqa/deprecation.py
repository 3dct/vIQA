"""Warnings for deprecated features."""


class RemovedInViqa10Warning(DeprecationWarning):
    """Warn about features that will be removed in ViQa 1.0."""

    pass


class RemovedInViqa20Warning(PendingDeprecationWarning):
    """Warn about features that will be removed in ViQa 2.0."""

    pass


RemovedInNextVersionWarning = RemovedInViqa10Warning
