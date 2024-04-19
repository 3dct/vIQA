"""
vIQA (volumetric Image Quality Assessment) is a Python package for the assessment of
volumetric image quality.

The package provides a set of metrics to assess the quality of volumetric images.
It can be used for 2D and 3D images.
"""

__version__ = "0.13.14"
__author__ = "Lukas Behammer"
__all__ = ["FSIM", "GSM", "MAD", "MSSSIM", "PSNR", "PSNR", "RMSE", "SSIM", "VIFp",
           "VSI", "CNR", "SNR", "load_data", "normalize_data", "export_csv",
           "fuse_metrics_linear_combination", "kernels"]

import kernels

from .fr_metrics import *
from .fusion import fuse_metrics_linear_combination
from .nr_metrics import *
from .utils import export_csv, load_data, normalize_data
