"""Module for calculating the peak signal-to-noise ratio (PSNR) between two images.

Examples
--------
TODO: add examples
"""

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
# TODO: add license

import metrics
from utils import _check_imgs
from skimage.metrics import peak_signal_noise_ratio
from warnings import warn


class PSNR(metrics.FullReferenceMetricsInterface):
    """
    Class to calculate the peak signal-to-noise ratio (PSNR) between two images.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Can be omitted if `normalize` is False.
    normalize : bool, default False
        If True, the input images are normalized to the `data_range` argument.
    batch : bool, default False
        If True, the input images are expected to be given as path to a folder containing the images.
        .. note:: Currently not supported. Added for later implementation.
    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to `utils.load_data`.
        See below for details.

    Attributes
    ----------
    score_val : float
        PSNR score value of the last calculation.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.
        .. note:: Currently not supported.

    Notes
    -----
    The parameter `data_range` for image loading is also used for the PSNR calculation and therefore must be set.
    The parameter is set through the constructor of the class and is passed to the `score` method.
    """

    def __init__(self, data_range=255, normalize=True, batch=False, **kwargs):
        """Constructor method"""
        super().__init__(data_range=data_range, normalize=normalize, batch=batch)
        self._parameters.update(**kwargs)

    def score(self, img_r, img_m):
        """Calculates the peak signal-to-noise ratio (PSNR) between two images.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.

        Returns
        -------
        score_val : float
            PSNR score value.
        """

        # Check images
        img_r, img_m = _check_imgs(img_r, img_m, data_range=self._parameters['data_range'],
                                   normalize=self._parameters['normalize'], batch=self._parameters['batch'])
        # Calculate score
        score_val = peak_signal_noise_ratio(img_r, img_m, data_range=self._parameters['data_range'])
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Prints the PSNR score value of the last calculation.

        Parameters
        ----------
        decimals : int, default=2
            Number of decimal places to print the score value.

        Warns
        -----
        RuntimeWarning
            If no score value is available. Run score() first.
        """
        if self.score_val is not None:
            print('PSNR: {}'.format(round(self.score_val, decimals)))
        else:
            warn('No score value for PSNR. Run score() first.', RuntimeWarning)
