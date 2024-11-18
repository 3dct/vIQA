"""
vIQA is a Python package for the assessment of volumetric image quality.

The package provides a set of metrics to assess the quality of volumetric images.
It can be used for 2D and 3D images.
"""

__version__ = "2.0.5"
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
    "crop_image",
    "export_metadata",
    "export_results",
    "export_image",
    "fuse_metrics_linear_combination",
    "visualize_2d",
    "visualize_3d",
    "BatchMetrics",
    "MultipleMetrics",
    "ImageArray",
]

from .fr_metrics import *
from .fusion import fuse_metrics_linear_combination
from .multiple import BatchMetrics, MultipleMetrics
from .nr_metrics import *
from .utils import (
    ImageArray,
    crop_image,
    export_image,
    export_metadata,
    export_results,
    load_data,
    normalize_data,
    visualize_2d,
    visualize_3d,
)


def get_version():
    """
    Return the version of the package.

    Returns
    -------
    str
        The version of the package
    """
    return __version__
