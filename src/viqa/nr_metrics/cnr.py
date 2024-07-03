"""Module for calculating the contrast-to-noise ratio (CNR) for an image.

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import CNR
        >>> img = np.random.rand(256, 256)
        >>> cnr = CNR(data_range=1, normalize=False)
        >>> cnr
        CNR(score_val=None)
        >>> score = cnr.score(img,
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
from viqa.load_utils import load_data
from viqa.visualization_utils import _visualize_cnr_2d, _visualize_cnr_3d


class CNR(NoReferenceMetricsInterface):
    """Class to calculate the contrast-to-noise ratio (CNR) for an image.

    Attributes
    ----------
    score_val : float
        CNR score value of the last calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True. Passed to :py:func:`viqa.utils.load_data`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`.viqa.utils.load_data`.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. note::
            Currently not supported.

    """

    def __init__(self, data_range=255, normalize=False, **kwargs) -> None:
        """Constructor method."""
        super().__init__(
            data_range=data_range, normalize=normalize, **kwargs
        )
        self._name = "CNR"

    def score(self, img, **kwargs):
        """Calculate the contrast-to-noise ratio (CNR) for an image.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to calculate score of.
        **kwargs : optional
            Additional parameters for CNR calculation. The keyword arguments are passed
            to :py:func:`.viqa.nr_metrics.cnr.contrast_to_noise_ratio`.

        Returns
        -------
        score_val : float
            CNR score value.
        """
        # write kwargs to ._parameters attribute
        self._parameters.update(kwargs)

        # Load image
        img = load_data(
            img,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
        )

        score_val = contrast_to_noise_ratio(img, **kwargs)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the CNR score value of the last calculation.

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
            print("CNR: {}".format(round(self.score_val, decimals)))
        else:
            warn("No score value for CNR. Run score() first.", RuntimeWarning)

    def visualize_centers(
            self,
            img,
            signal_center=None,
            background_center=None,
            radius=None
    ):
        """Visualize the centers for CNR calculation.

        The visualization shows the signal and background regions in a matplotlib plot.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to visualize.
        signal_center : Tuple(int), optional
            Center of the signal.
            Order is ``(y, x)`` for 2D images and ``(z, y, x)`` for 3D images.
        background_center : Tuple(int), optional
            Center of the background. Order is ``(y, x)`` for 2D images and
            ``(z, y, x)`` for 3D images.
        radius : int, optional
            Width of the regions.
        """
        if not signal_center or not background_center or not radius:
            if (not self._parameters["signal_center"]
                    or not self._parameters["background_center"]
                    or not self._parameters["radius"]):
                raise ValueError("No center or radius provided.")

            signal_center = self._parameters["signal_center"]
            background_center = self._parameters["background_center"]
            radius = self._parameters["radius"]

        if img.ndim != len(signal_center) or img.ndim != len(background_center):
            raise ValueError("Centers have to be in the same dimension as img.")

        if img.ndim == 2:
            _visualize_cnr_2d(img=img, signal_center=signal_center,
                              background_center=background_center, radius=radius)
        elif img.ndim == 3:
            _visualize_cnr_3d(img=img, signal_center=signal_center,
                              background_center=background_center, radius=radius)
        else:
            raise ValueError("No visualization possible for non 2d or non 3d images.")


def contrast_to_noise_ratio(img, background_center, signal_center, radius):
    """Calculate the contrast-to-noise ratio (CNR) for an image.

    Parameters
    ----------
    img : np.ndarray or Tensor or str or os.PathLike
        Image to calculate score of.
    background_center : Tuple(int)
        Center of the background. Order is ``(y, x)`` for 2D images and ``(z, y, x)``
        for 3D images.
    signal_center : Tuple(int)
        Center of the signal. Order is ``(y, x)`` for 2D images and ``(z, y, x)`` for
        3D images.
    radius : int
        Width of the regions.

    Returns
    -------
    score_val : float
        CNR score value.

    Raises
    ------
    ValueError
        If the input image is not 2D or 3D. \n
        If the input center is not a tuple of integers. \n
        If the input center is too close to the border. \n
        If the input radius is not an integer.

    Notes
    -----
    This implementation uses cubic regions to calculate the CNR. The calculation is
    based on the following formula:

    .. math::
        CNR = \\frac{\\mu_{signal} - \\mu_{background}}{\\sigma_{background}}

    where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.

    .. important::
        The background region should be chosen in a homogeneous area, while the signal
        region should be chosen in an area with a high contrast.

    References
    ----------
    .. [1] Desai, N., Singh, A., & Valentino, D. J. (2010). Practical evaluation of
        image quality in computed radiographic (CR) imaging systems. Medical Imaging
        2010: Physics of Medical Imaging, 7622, 76224Q. https://doi.org/10.1117/12.844640
    """
    # check if signal_center and background_center are tuples of integers and radius is
    # an integer
    for center in signal_center:
        if not isinstance(center, int):
            raise TypeError("Signal center has to be a tuple of integers.")
        if abs(center) - radius < 0:  # check if center is too close to the border
            raise ValueError(
                "Signal center has to be at least the radius away from the border."
            )

    for center in background_center:
        if not isinstance(center, int):
            raise TypeError("Background center has to be a tuple of integers.")
        if abs(center) - radius < 0:
            raise ValueError(
                "Background center has to be at least the radius away from the border."
            )

    if not isinstance(radius, int) or radius <= 0:
        raise TypeError("Radius has to be an integer.")

    # Check if img and centers have the same dimension
    if img.ndim != len(signal_center) or img.ndim != len(background_center):
        raise ValueError("Centers have to be in the same dimension as img.")

    # Define regions
    if img.ndim == 2:  # 2D image
        background = img[
            background_center[0] - radius : background_center[0] + radius,
            background_center[1] - radius : background_center[1] + radius,
        ]
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
        ]
    elif img.ndim == 3:  # 3D image
        background = img[
            background_center[0] - radius : background_center[0] + radius,
            background_center[1] - radius : background_center[1] + radius,
            background_center[2] - radius : background_center[2] + radius,
        ]
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
            signal_center[2] - radius : signal_center[2] + radius,
        ]
    else:
        raise ValueError("Image has to be either 2D or 3D.")

    # Calculate CNR
    if np.std(background) == 0:
        cnr_val = 0
    else:
        cnr_val = (np.mean(signal) - np.mean(background)) / np.std(background)

    return cnr_val
