#!/usr/bin/env python
"""Get numpy include path."""

import os

try:
    import numpy
except ImportError:
    print("numpy is not installed")
    exit(20)

os.chdir("..")
print(numpy.get_include().replace("\\", "/"))
