"""Module for calculating the signal-to-noise ratio (SNR) between two images.

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import SNR
        >>> img = np.random.rand(256, 256)
        >>> snr = SNR(data_range=1, normalize=False)
        >>> snr
        SNR(score_val=None)
        >>> score = snr.score(img,
        ...                   background_center=(128, 128),
        ...                   signal_center=(32, 32),
        ...                   radius=16)
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

from viqa._metrics import NoReferenceMetricsInterface
from viqa.utils import load_data


class SNR(NoReferenceMetricsInterface):
    """Class to calculate the signal-to-noise ratio (SNR) between two images.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when `normalize` is True.
    normalize : bool, default False
        If True, the input images are normalized to the `data_range` argument.
    batch : bool, default False
        If True, the input images are expected to be given as path to a folder containing the images.

        .. note::
            Currently not supported. Added for later implementation.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to `viqa.utils.load_data`.
        See below for details.

    Attributes
    ----------
    score_val : float
        SNR score value of the last calculation.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. note::
            Currently not supported.

    """

    def __init__(self, data_range=255, normalize=False, batch=False, **kwargs) -> None:
        """Constructor method"""

        super().__init__(data_range=data_range, normalize=normalize, batch=batch, **kwargs)

    def score(self, img, **kwargs):
        """Calculate the signal-to-noise ratio (SNR) between two images.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to calculate score of.
        **kwargs : optional
            Additional parameters for SNR calculation. The keyword arguments are passed to
            `viqa.snr.signal_to_noise_ratio()`.

        Returns
        -------
        score_val : float
            SNR score value.
        """

        # Check images
        img = load_data(
            img,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            batch=self._parameters["batch"],
        )

        score_val = signal_to_noise_ratio(img, **kwargs)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the SNR score value of the last calculation.

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
            print("SNR: {}".format(round(self.score_val, decimals)))
        else:
            warn("No score value for SNR. Run score() first.", RuntimeWarning)


def signal_to_noise_ratio(img, signal_center, radius):
    """Calculate the signal-to-noise ratio (SNR) between two images.

    Parameters
    ----------
    img : np.ndarray or Tensor or str or os.PathLike
        Image to calculate score of.
    signal_center : Tuple(int)
        Center of the signal.
    radius : int
        Width of the regions.

    Returns
    -------
    score_val : float
        SNR score value.

    Notes
    -----
    This implementation uses a cubic region to calculate the SNR. The calculation is based on the following formula:

    .. math::
        SNR = \\frac{\\mu}{\\sigma}

    where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.
    """

    # Define regions
    if img.ndim == 2:  # 2D image
        signal = img[signal_center[0] - radius:signal_center[0] + radius,
                     signal_center[1] - radius:signal_center[1] + radius]
    elif img.ndim == 3:  # 3D image
        signal = img[signal_center[0] - radius:signal_center[0] + radius,
                     signal_center[1] - radius:signal_center[1] + radius,
                     signal_center[2] - radius:signal_center[2] + radius]
    else:
        raise ValueError("Image has to be either 2D or 3D.")

    # Calculate SNR
    snr_val = np.mean(signal) / np.std(signal)

    return snr_val
