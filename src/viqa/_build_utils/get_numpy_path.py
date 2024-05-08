#!/usr/bin/env python
"""Get numpy include path."""

# import os
#
# import numpy
#
# os.chdir("..")
# print(os.path.relpath(numpy.get_include().replace("\\", "/")))

# incdir_numpy = 'C:/ProgramData/anaconda3/Lib/site-packages/numpy/core/include'
incdir_numpy = '/opt/conda/lib/python3.11/site-packages/numpy/core/include'
print(incdir_numpy)
