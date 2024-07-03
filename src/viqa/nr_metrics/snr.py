"""Module for calculating the signal-to-noise ratio (SNR) for an image.

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
from viqa.load_utils import load_data
from viqa.utils import _to_grayscale, _rgb_to_yuv
from viqa.visualization_utils import _visualize_snr_2d, _visualize_snr_3d


class SNR(NoReferenceMetricsInterface):
    """Class to calculate the signal-to-noise ratio (SNR) for an image.

    Attributes
    ----------
    score_val : float
        SNR score value of the last calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True. Passed to :py:func:`viqa.utils.load_data`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.utils.load_data`.

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
        self._name = "SNR"

    def score(self, img, **kwargs):
        """Calculate the signal-to-noise ratio (SNR) for an image.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to calculate score of.
        **kwargs : optional
            Additional parameters for SNR calculation. The keyword arguments are passed
            to :py:func:`viqa.nr_metrics.snr.signal_to_noise_ratio`.

        Returns
        -------
        score_val : float
            SNR score value.
        """
        # write kwargs to ._parameters attribute
        self._parameters.update(kwargs)

        # Load image
        img = load_data(
            img,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
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
            If :py:attr:`score_val` is not available.
        """
        if self.score_val is not None:
            print("SNR: {}".format(round(self.score_val, decimals)))
        else:
            warn("No score value for SNR. Run score() first.", RuntimeWarning)

    def visualize_centers(self, img, signal_center=None, radius=None):
        """Visualize the centers for SNR calculation.

        The visualization shows the signal region in a matplotlib plot.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to visualize.
        signal_center : Tuple(int), optional
            Center of the signal.
            Order is ``(y, x)`` for 2D images and ``(z, y, x)`` for 3D images.
        radius : int, optional
            Width of the regions.
        """
        if not signal_center or not radius:
            if not self._parameters["signal_center"] or not self._parameters["radius"]:
                raise ValueError("No center or radius provided.")

            signal_center = self._parameters["signal_center"]
            radius = self._parameters["radius"]

        # Check if img and signal_center have the same dimension
        if img.shape[-1] == 3:
            if img.ndim != len(signal_center) + 1:
                raise ValueError("Center has to be in the same dimension as img.")
        else:
            if img.ndim != len(signal_center):
                raise ValueError("Center has to be in the same dimension as img.")

        # Visualize centers
        if img.ndim == 3 and (img.shape[-1] != 3):
            _visualize_snr_3d(img=img, signal_center=signal_center, radius=radius)
        elif img.ndim == 3 and (img.shape[-1] == 3):
            img = _to_grayscale(img)
            _visualize_snr_2d(img=img, signal_center=signal_center, radius=radius)
        elif img.ndim == 2:
            _visualize_snr_2d(img=img, signal_center=signal_center, radius=radius)
        else:
            raise ValueError("No visualization possible for non 2d or non 3d images.")


def signal_to_noise_ratio(img, signal_center, radius, yuv=True):
    """Calculate the signal-to-noise ratio (SNR) for an image.

    Parameters
    ----------
    img : np.ndarray or Tensor or str or os.PathLike
        Image to calculate score of.
    signal_center : Tuple(int)
        Center of the signal. Order is ``(y, x)`` for 2D images and ``(z, y, x)`` for
        3D images.
    radius : int
        Width of the regions.
    yuv : bool, default True

        .. important::
            Only applicable for color images.

        If True, the input images are expected to be RGB images and are converted to YUV
        color space. If False, the input images are kept as RGB images.

    Returns
    -------
    snr_lum : float
        SNR score value for grayscale image.
    snr_val[...] : float, optional
        SNR score values per channel for color image. The order is Y, U, V for YUV
        images and R, G, B for RGB images.

        .. note::
            For RGB images the first return value is the SNR for the luminance channel.

    Raises
    ------
    ValueError
        If the center is not a tuple of integers. \n
        If center is too close to the border. \n
        If the radius is not an integer. \n
        If the image is not 2D or 3D.

    Notes
    -----
    This implementation uses a cubic region to calculate the SNR. The calculation for
    grayscale images is based on the following formula:

    .. math::
       SNR = \\frac{\\mu}{\\sigma}

    where :math:`\\mu` is the mean and :math:`\\sigma` is the standard deviation.

    For color images, the calculation is a lot more complicated. The image is first
    converted to YUV color space by matrix multiplication with the weighting matrix [1]_:

    .. math::
        \\begin{bmatrix}
            Y \\\\
            U \\\\
            V \\\\
        \\end{bmatrix}
        =
        \\begin{bmatrix}
            0.2126 & 0.7152 & 0.0722 \\\\
            -0.09991 & -0.33609 & 0.436 \\\\
            0.615 & -0.55861 & -0.05639 \\\\
        \\end{bmatrix}
        \\begin{bmatrix}
            R \\\\
            G \\\\
            B \\\\
        \\end{bmatrix}

    Then the SNR is calculated for each channel separately [2]_:

    .. math::
        SNR_{channel} = \\frac{\\mu_{Y}}{\\sigma_{channel}}

    where :math:`\\mu_{Y}` is the mean of the Y channel and :math:`\\sigma_{channel}` is
    the standard deviation of the channel for YUV images and:

    .. math::
        SNR_{channel} = \\frac{\\mu_{channel}}{\\sigma_{channel}}

    for RGB images.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/YUV
    .. [2] https://www.imatest.com/docs/color-tone-esfriso-noise/#chromanoise
    """
    # check if signal_center is a tuple of integers and radius is an integer
    for center in signal_center:
        if not isinstance(center, int):
            raise TypeError("Center has to be a tuple of integers.")
        if abs(center) - radius < 0:  # check if center is too close to the border
            raise ValueError(
                "Center has to be at least the radius away from the border."
            )

    if not isinstance(radius, int) or radius <= 0:
        raise TypeError("Radius has to be a positive integer.")

    # Check if img and signal_center have the same dimension
    if img.shape[-1] == 3:
        if img.ndim != len(signal_center) + 1:
            raise ValueError("Center has to be in the same dimension as img.")
    else:
        if img.ndim != len(signal_center):
            raise ValueError("Center has to be in the same dimension as img.")

    # Color images
    if img.ndim == 3 and (img.shape[-1] == 3):  # 2D RGB image
        if yuv:
            img = _rgb_to_yuv(img)

        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
            :,
        ]

        sdev = np.std(signal, axis=(0, 1))

        if yuv:
            snr_val = [np.mean(signal[..., 0]) / sdev[i] if sdev[i] != 0 else 0
                       for i in range(3)]
            return snr_val[0], snr_val[1], snr_val[2]
        else:
            snr_lum = signal_to_noise_ratio(
                _to_grayscale(img),
                signal_center,
                radius,
                yuv=False
            )
            snr_val = [np.mean(signal[..., i]) / sdev[i] if sdev[i] != 0 else 0
                       for i in range(3)]

        return snr_lum, snr_val[0], snr_val[1], snr_val[2]

    # Define regions
    if img.ndim == 2:  # 2D image
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
        ]
    elif img.ndim == 3 and (img.shape[-1] != 3):  # 3D image
        signal = img[
            signal_center[0] - radius : signal_center[0] + radius,
            signal_center[1] - radius : signal_center[1] + radius,
            signal_center[2] - radius : signal_center[2] + radius,
        ]
    else:
        raise ValueError("Image has to be either 2D or 3D.")

    # Calculate SNR
    if np.std(signal) == 0:
        snr_val = 0
    else:
        snr_val = np.mean(signal) / np.std(signal)

    return snr_val
