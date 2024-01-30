"""Module for the root mean squared error (RMSE) metric.

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

from warnings import warn

import numpy as np
from skimage.metrics import mean_squared_error

from vIQA._metrics import FullReferenceMetricsInterface
from vIQA.utils import _check_imgs


class RMSE(FullReferenceMetricsInterface):
    """Class to calculate the root mean squared error (RMSE) between two images.

    Parameters
    ----------
    data_range : {1, 255, 65535}, optional
        Data range of the returned data in data loading. Can be omitted if `normalize` is False.
    normalize : bool, default False
        If True, the input images are normalized to the `data_range` argument.
    batch : bool, default False
        If True, the input images are expected to be given as path to a folder containing the images.
        .. note:: Currently not supported. Added for later implementation.
    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to `vIQA.utils.load_data`.
        See below for details.

    Attributes
    ----------
    score_val : float
        RMSE score value of the last calculation.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.
        .. note:: Currently not supported.
    """

    def __init__(self, data_range=None, normalize=None, batch=None, **kwargs) -> None:
        """Constructor method"""

        super().__init__(data_range=data_range, normalize=normalize, batch=batch)
        self._parameters.update(**kwargs)

    def score(self, img_r, img_m):
        """Calculates the RMSE score between two images.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.

        Returns
        -------
        score_val : float
            RMSE score value.
        """

        # Check images
        img_r, img_m = _check_imgs(img_r, img_m, data_range=self._parameters['data_range'],
                                   normalize=self._parameters['normalize'], batch=self._parameters['batch'])
        # Calculate score
        score_val = np.sqrt(mean_squared_error(img_r, img_m))
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Prints the RMSE score value of the last calculation.

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
            print('RMSE: {}'.format(round(self.score_val, decimals)))
        else:
            warn('No score value for RMSE. Run score() first.', RuntimeWarning)