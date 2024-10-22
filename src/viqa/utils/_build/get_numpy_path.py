#!/usr/bin/env python
"""Get numpy include path."""

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

import os

try:
    import numpy
except ImportError:
    numpy = None
    print("numpy is not installed")
    exit(20)

os.chdir("../..")
print(numpy.get_include().replace("\\", "/"))
