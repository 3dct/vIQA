#!/usr/bin/env python
"""Get numpy include path."""

# import os
#
# import numpy
#
# os.chdir("..")
# print(os.path.relpath(numpy.get_include().replace("\\", "/")))

import os
from pathlib import Path

home_path = Path.home()

incdir_numpy = os.path.join(home_path, 'AppData/Local/anaconda3/Lib/site-packages/numpy/core/include')
# incdir_numpy = 'C:/ProgramData/anaconda3/Lib/site-packages/numpy/core/include'
# incdir_numpy = '/opt/conda/lib/python3.11/site-packages/numpy/core/include'
print(incdir_numpy)
