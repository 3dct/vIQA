"""
vIQA (voulmetric Image Quality Assessment) is a Python package for the assessment of volumetric image quality.
It provides a set of metrics to assess the quality of volumetric images. The package can be used for 2D and 3D images.
"""

__version__ = "0.6.0"
__author__ = "Lukas Behammer"

from .metrics import *
from ._kernels import *
from .utils import load_data, normalize_data
