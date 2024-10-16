#!/usr/bin/env python
"""Extract version number from __init__.py."""

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

viqa_init = os.path.join(os.path.dirname(__file__), "../../__init__.py")

data = open(viqa_init).readlines()
version_line = next(line for line in data if line.startswith("__version__"))

version = version_line.strip().split(" = ")[1].replace('"', "").replace("'", "")

print(version)
