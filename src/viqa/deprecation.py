"""Warnings for deprecated features."""


class RemovedInViqa10Warning(DeprecationWarning):
    pass


class RemovedInViqa20Warning(PendingDeprecationWarning):
    pass


RemovedInNextVersionWarning = RemovedInViqa10Warning
