"""Module for the structural similarity index (SSIM) metric.

Notes
-----
This code is adapted from skimage.metrics.structural_similarity available under [1].

References
----------
.. [1] scikit-image team (2023). https://github.com/scikit-image/scikit-image

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import SSIM
        >>> img_r = np.zeros((256, 256))
        >>> img_m = np.ones((256, 256))
        >>> ssim = SSIM(data_range=1, normalize=False)
        >>> ssim
        SSIM(score_val=None)
        >>> score = ssim.score(img_r, img_m)
        >>> score
        0.0
        >>> ssim.print_score()
        SSIM: 1.0
        >>> img_r = np.zeros((256, 256))
        >>> img_m = np.zeros((256, 256))
        >>> ssim.score(img_r, img_m)
        1.0
        >>> img_r = np.random.rand(256, 256)
        >>> img_m = np.random.rand(128, 128)
        >>> ssim.score(img_r, img_m)
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
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util.arraycrop import crop

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_imgs


class SSIM(FullReferenceMetricsInterface):
    """Calculates the structural similarity index (SSIM) between two images.

    Attributes
    ----------
    score_val : float or None
        Score value of the SSIM metric.

    Parameters
    ----------
    data_range : {1, 255, 65535}, optional
        Data range of the returned data in data loading. Can be omitted if `normalize` is False.
    normalize : bool, default False
        If True, the input images are normalized to the `data_range` argument.
    batch : bool, default False
        If True, the input images are expected to be given as path to a folder containing the images.

        .. note::
            Currently not supported. Added for later implementation.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to `viqa.utils.load_data`.
        See below for details.

        .. seealso::
            [`viqa.utils.load_data`]

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. note::
            Currently not supported.

    Raises
    ------
    ValueError
        If `data_range` is not set.

    Notes
    -----
    The parameter `data_range` for image loading is also used for the SSIM calculation if the image type is integer and
    therefore must be set. The parameter is set through the constructor of the class and is passed to the `score`
    method. SSIM [1] is a full-reference IQA metric. It is based on the human visual system and is designed to predict
    the perceived quality of an image.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error
           visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600–612.
           https://doi.org/10.1109/TIP.2003.819861
    """

    def __init__(self, data_range=255, normalize=False, batch=False, **kwargs):
        """Constructor method"""

        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, batch=batch, **kwargs)

    def score(self, img_r, img_m, **kwargs):
        """Calculates the structural similarity index (SSIM) between two images.

        Parameters
        ----------
        img_r : np.ndarray
            Reference image to calculate score against.
        img_m : np.ndarray
            Modified image to calculate score of.
        **kwargs : optional
            Additional parameters for the SSIM calculation. The keyword arguments are passed to `structural_similarity`.

            .. seealso::
                [`structural_similarity`]

        Returns
        -------
        score_val : float
            SSIM score value.

        Notes
        -----
        .. note::
            The metric is currently not usable for color images.
        """
        img_r, img_m = _check_imgs(
            img_r,
            img_m,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            batch=self._parameters["batch"],
        )
        score_val = structural_similarity(
            img_r, img_m, data_range=self._parameters["data_range"], **kwargs
        )
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the SSIM score value of the last calculation.

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
            print("SSIM: {}".format(round(self.score_val, decimals)))
        else:
            print("No score value for SSIM. Run score() first.")


def structural_similarity(
    img_r,
    img_m,
    win_size=None,
    data_range=None,
    gaussian_weights=True,
    alpha=1,
    beta=1,
    gamma=1,
    **kwargs,
):
    """Compute the structural similarity index between two images.

    Parameters
    ----------
    img_r : np.ndarray
        Reference image to calculate score against.
    img_m : np.ndarray
        Modified image to calculate score of.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    data_range : int, default=255
        Data range of the input images.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    alpha : float, default=1
        Weight of the luminance comparison. Should be alpha >=1.
    beta : float, default=1
        Weight of the contrast comparison. Should be beta >=1.
    gamma : float, default=1
        Weight of the structure comparison. Should be gamma >=1.

    Other Parameters
    ----------------
    K1 : float, default=0.01
        Algorithm parameter, K1 (small constant, see [1]_).
    K2 : float, default=0.03
        Algorithm parameter, K2 (small constant, see [1]_).
    sigma : float, default=1.5
        Standard deviation for the Gaussian when `gaussian_weights` is True.
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        The mode parameter determines how the array borders are handled. See
        Numpy documentation for detail.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'. Default is 0.

    Returns
    -------
    ssim : float
        The mean structural similarity index over the image.

    Raises
    ------
    ValueError
        If `K1`, `K2` or `sigma` are negative.
        If `win_size` exceeds image or is not an odd number.

    Warns
    -----
    RuntimeWarning
        If `alpha`, `beta` or `gamma` are not integers.

    Notes
    -----
    To match the implementation in [1], set `gaussian_weights` to True and `sigma` to 1.5.
    This code is adapted from skimage.metrics.structural_similarity available under [2].

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error
           visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600–612.
           https://doi.org/10.1109/TIP.2003.819861
    .. [2] scikit-image team (2023). https://github.com/scikit-image/scikit-image

    """
    # Authors
    # -------
    # Author: scikit-image team
    #
    # Adaption: Lukas Behammer
    # Research Center Wels
    # University of Applied Sciences Upper Austria, 2023
    # CT Research Group
    #
    # Modifications
    # -------------
    # Original code, 2009-2022, scikit-image team
    # Adapted, 2024, Lukas Behammer
    #
    # License
    # -------
    # BSD-3-Clause

    K1 = kwargs.pop("K1", 0.01)
    K2 = kwargs.pop("K2", 0.03)
    sigma = kwargs.pop("sigma", 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004
        else:
            win_size = 7  # backwards compatibility

    if np.any((np.asarray(img_r.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent. "
            "Either ensure that your images are "
            "at least 7x7; or pass win_size explicitly "
            "in the function call, with an odd value "
            "less than or equal to the smaller side of your "
            "images."
        )

    if not (win_size % 2 == 1):
        raise ValueError("Window size must be odd.")

    ndim = img_r.ndim

    mode = kwargs.pop("mode", "reflect")
    cval = kwargs.pop("cval", 0)

    if gaussian_weights:
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate, 'mode': mode, 'cval': cval}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size, 'mode': mode, 'cval': cval}

    if not isinstance(alpha, int):
        alpha = int(alpha)
        warn("alpha is not an integer. Cast to int.", RuntimeWarning)
    if not isinstance(beta, int):
        beta = int(beta)
        warn("beta is not an integer. Cast to int.", RuntimeWarning)
    if not isinstance(gamma, int):
        gamma = int(gamma)
        warn("gamma is not an integer. Cast to int.", RuntimeWarning)

    # ndimage filters need floating point data
    img_r = img_r.astype(np.float64, copy=False)
    img_m = img_m.astype(np.float64, copy=False)

    NP = win_size**ndim
    if not cov_norm:
        cov_norm = NP / (NP - 1)  # sample covariance

    # compute (weighted) means
    ux = filter_func(img_r, **filter_args)
    uy = filter_func(img_m, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(img_r * img_r, **filter_args)
    uyy = filter_func(img_m * img_m, **filter_args)
    uxy = filter_func(img_r * img_m, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    C3 = C2 / 2

    A1, A2, B1, B2 = (
        2 * ux * uy + C1,
        2 * vxy + C2,
        ux**2 + uy**2 + C1,
        vx + vy + C2,
    )

    lum = A1 / B1
    con = (2 * np.sqrt(vx) * np.sqrt(vy) + C2) / B2
    stru = (vxy + C3) / (np.sqrt(vx) * np.sqrt(vy) + C3)

    S = (lum**alpha) * (con**beta) * (stru**gamma)

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    ssim = crop(S, pad).mean(dtype=np.float64)

    return ssim
