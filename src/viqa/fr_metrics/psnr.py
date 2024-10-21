"""Module for calculating the peak signal-to-noise ratio (PSNR) between two images.

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import PSNR
        >>> img_r = np.zeros((256, 256))
        >>> img_m = np.ones((256, 256))
        >>> psnr = PSNR(data_range=1, normalize=False)
        >>> psnr
        PSNR(score_val=None)
        >>> score = psnr.score(img_r, img_m)
        >>> score
        0.0
        >>> psnr.print_score()
        PSNR: 0.0
        >>> img_r = np.zeros((256, 256))
        >>> img_m = np.zeros((256, 256))
        >>> psnr.score(img_r, img_m)
        inf
        >>> img_r = np.random.rand(256, 256)
        >>> img_m = np.random.rand(128, 128)
        >>> psnr.score(img_r, img_m)
        Traceback (most recent call last):
            ...
        ValueError: Image shapes do not match
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
# BSD-3-Clause License

from warnings import warn

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from viqa._metrics import FullReferenceMetricsInterface


class PSNR(FullReferenceMetricsInterface):
    """Class to calculate the peak signal-to-noise ratio (PSNR) between two images.

    Attributes
    ----------
    score_val : float
        PSNR score value of the last calculation.
    parameters : dict
        Dictionary containing the parameters for PSNR calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the PSNR calculation. Passed to
        :py:func:`viqa.utils.load_data` and :py:meth:`score`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.utils.load_data`.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.
        If False, the input images are converted to grayscale images if necessary.

    Raises
    ------
    ValueError
        If ``data_range`` is not set.

    Notes
    -----
    ``data_range`` for image loading is also used for the PSNR calculation and therefore
    must be set. The parameter is set through the constructor of the class and is passed
    to :py:meth:`score`.
    """

    def __init__(self, data_range=255, normalize=False, **kwargs) -> None:
        """Construct method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        if self.parameters["chromatic"]:
            self._name = "PSNRc"
        else:
            self._name = "PSNR"

    def score(self, img_r, img_m):
        """Calculate the peak signal-to-noise ratio (PSNR) between two images.

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
        img_r, img_m = self.load_images(img_r, img_m)
        # Calculate score
        if np.array_equal(img_r, img_m):
            score_val = np.inf  # PSNR of identical images is infinity
        else:
            score_val = peak_signal_noise_ratio(
                img_r, img_m, data_range=self.parameters["data_range"]
            )
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the PSNR score value of the last calculation.

        Parameters
        ----------
        decimals : int, default=2
            Number of decimal places to print the score value.

        Warns
        -----
        RuntimeWarning
            If :py:attr:`score_val` is not available.
        """
        if self.score_val is not None:
            print("PSNR: {}".format(np.round(self.score_val, decimals)))
        else:
            warn("No score value for PSNR. Run score() first.", RuntimeWarning)
