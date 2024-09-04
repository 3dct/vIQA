"""
vIQA is a Python package for the assessment of volumetric image quality.

The package provides a set of metrics to assess the quality of volumetric images.
It can be used for 2D and 3D images.
"""

__version__ = "1.12.0"
__author__ = "Lukas Behammer"
__all__ = [
    "FSIM",
    "GSM",
    "MAD",
    "UQI",
    "MSSSIM",
    "PSNR",
    "PSNR",
    "RMSE",
    "SSIM",
    "VIFp",
    "VSI",
    "CNR",
    "SNR",
    "QMeasure",
    "load_data",
    "normalize_data",
    "export_metadata",
    "export_results",
    "export_image",
    "fuse_metrics_linear_combination",
    "visualize_3d",
    "BatchMetrics",
    "MultipleMetrics",
    "ImageArray",
]

from .fr_metrics import *
from .fusion import fuse_metrics_linear_combination
from .load_utils import ImageArray, load_data, normalize_data
from .multiple import BatchMetrics, MultipleMetrics
from .nr_metrics import *
from .utils import export_image, export_metadata, export_results
from .visualization_utils import visualize_3d


def get_version():
    """
    Return the version of the package.

    Returns
    -------
    str
        The version of the package
    """
    return __version__
