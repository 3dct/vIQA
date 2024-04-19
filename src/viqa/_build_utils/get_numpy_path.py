#!/usr/bin/env python
"""Get numpy include path."""

import os

import numpy

os.chdir("..")
print(numpy.get_include())
