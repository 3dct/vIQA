"""Module for utility functions for importing."""

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

import builtins
from importlib import import_module
from typing import Any, Tuple, Union
from warnings import warn


def try_import(module: str, name: Union[str, None] = None) -> Tuple[Any, bool]:
    """
    Import a module by name.

    Parameters
    ----------
    module : str
        The name of the module to import.
    name : str, optional
        The name of the function, class, etc. to import from the module,
        default is None.

    Returns
    -------
    Tuple[Any, bool]
        The imported module and a boolean indicating whether the import was successful.

    """
    try:
        imported_module = import_module(module)
        if name:
            imported_module = getattr(imported_module, name)
    except ImportError:
        imported_module = None
        import_successful = False
        warn(
            f"Could not import {module}. Some functionality may be limited.",
            UserWarning,
        )
        return imported_module, import_successful
    else:
        import_successful = True
        return imported_module, import_successful


def is_ipython() -> bool:
    """
    Check if running in an IPython environment.

    Returns
    -------
    bool
        True if the current environment is an IPython environment, False otherwise.

    """
    return hasattr(builtins, "__IPYTHON__")


def check_interactive_vis_deps(has_ipywidgets: bool, has_ipython: bool) -> None:
    """
    Check if the necessary dependencies for interactive visualization are installed.

    Parameters
    ----------
    has_ipywidgets : bool
        Whether ipywidgets is installed.
    has_ipython : bool
        Whether IPython is installed.
    """
    if not has_ipywidgets and not has_ipython:
        raise ImportError(
            "ipywidgets and IPython are not installed. Please install them to use "
            "this method."
        )
    elif not has_ipywidgets:
        raise ImportError(
            "ipywidgets is not installed. Please install it to use this method."
        )
    elif not has_ipython:
        raise ImportError(
            "IPython is not installed. Please install it to use this method."
        )
