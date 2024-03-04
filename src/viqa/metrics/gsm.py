"""Module for calculating the Gradient Similarity Metric (GSM) metric.

Examples
--------
    .. todo:: Add examples
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

import scipy.ndimage as ndi

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_imgs, _to_float
from viqa.kernels import *

# Load the kernels as constants
KERNELS = [
            gsm_kernel_x(),
            gsm_kernel_y(),
            gsm_kernel_z(),
            gsm_kernel_xy1(),
            gsm_kernel_xy2(),
            gsm_kernel_yz1(),
            gsm_kernel_yz2(),
            gsm_kernel_xz1(),
            gsm_kernel_xz2(),
        ]
# TODO: Add the other 3 kernels


class GSM(FullReferenceMetricsInterface):
    """Calculate the gradient similarity (GSM) between two images.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when `normalize` is True and for the
        GSM calculation.
    normalize : bool, default=False
        If True, the input images are normalized to the `data_range` argument.
    batch : bool, default=False
        If True, the input images are expected to be given as path to a folder containing the images.

        .. caution::
            Currently not supported. Added for later implementation.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to `viqa.utils.load_data`.
        See below for details.

    Attributes
    ----------
    score_val : float
        GSM score value of the last calculation.

    Raises
    ------
    ValueError
        If the parameter `data_range` is not set.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. caution::
            Currently not supported.

    Notes
    -----
    .. todo:: Add notes
    The parameter `data_range` for image loading is also used for the GSM calculation and therefore must be set.
    The parameter is set through the constructor of the class and is passed to the `score` method.

    References
    ----------
    .. [1] Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on gradient similarity. IEEE
           Transactions on Image Processing, 21(4), 1500–1512. https://doi.org/10.1109/TIP.2011.2175935
    """

    def __init__(self, data_range=255, normalize=False, batch=False, **kwargs):
        """Constructor method"""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, batch=batch, **kwargs)
        self._parameters.update(**kwargs)

    def score(self, img_r, img_m, **kwargs):
        """Calculate the gradient similarity (GSM) between two images.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.
        **kwargs : optional
            Additional parameters for GSM calculation. The keyword arguments are passed to
            `viqa.gsm.gradient_similarity()`.

        Returns
        -------
        score_val : float
            GSM score value.

        Notes
        -----
        This implementation is adapted for 3D images. Therefore, 12 kernels are used instead of the original 4. Also,
        the gradient is calculated by max{convolve(img, kernel)} instead of max{mean2(abs(x * kernel))}.
        """
        img_r, img_m = _check_imgs(
            img_r,
            img_m,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            batch=self._parameters["batch"],
        )

        score_val = gradient_similarity(img_r, img_m, data_range=self._parameters["data_range"], **kwargs)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the GSM score value of the last calculation.

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
            print("GSM: {}".format(round(self.score_val, decimals)))
        else:
            warn("No score value for GSM. Run score() first.", RuntimeWarning)


def gradient_similarity(img_r, img_m, data_range=255, c=200, p=0.1):
    """Calculate the gradient similarity between two images.

    Parameters
    ----------
    img_r : np.ndarray
        Reference image to calculate score against
    img_m : np.ndarray
        Distorted image to calculate score of
    data_range : {1, 255, 65535}
        Data range of the input images
    c : int, default=200
        Constant as masking parameter. Typically, 200 <= c <= 1000. See [1] for details.
    p : float, default=0.1
        Constant for weighting between luminance and structure similarity. 0 <= p <= 1. Higher p means more accentuation
        of luminance. Should be signficantly smaller than 0.5. See [1] for details.

    Returns
    -------
    gsm_score : float
        GSM score value.

    References
    ----------
    .. [1] Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on gradient similarity. IEEE
           Transactions on Image Processing, 21(4), 1500–1512. https://doi.org/10.1109/TIP.2011.2175935
    """

    gradients_r = []
    gradients_m = []
    for kernel in KERNELS:
        gradients_r.append(ndi.correlate(img_r, kernel))
        gradients_m.append(ndi.correlate(img_m, kernel))

    # key for sorting the gradients by their mean of absolute values
    def _mean_of_abs(input_array):
        return np.mean(np.abs(input_array))

    img_r_gradient = sorted(gradients_r, key=_mean_of_abs, reverse=True)[0]
    img_m_gradient = sorted(gradients_m, key=_mean_of_abs, reverse=True)[0]

    img_r_gradient = _to_float(img_r_gradient)
    img_m_gradient = _to_float(img_m_gradient)
    k = c / max(np.max(img_r_gradient), np.max(img_m_gradient))
    r = np.abs(img_r_gradient - img_m_gradient) / max(
        img_r_gradient, img_m_gradient, key=_mean_of_abs
    )
    con_struc_sim = ((2 * (1 - r)) + k) / (1 + (1 - r) ** 2 + k)
    lum_sim = 1 - ((img_r - img_m) / data_range) ** 2
    weight = p * con_struc_sim
    quality = (1 - weight) * con_struc_sim + weight * lum_sim

    gsm_score = np.nanmean(quality)
    return gsm_score
