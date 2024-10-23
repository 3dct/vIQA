"""Module for the structural similarity index (SSIM) metric.

Notes
-----
This code is adapted from :py:func:`skimage.metrics.structural_similarity` available
under [1]_.

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
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.util.arraycrop import crop

from viqa._metrics import FullReferenceMetricsInterface


class SSIM(FullReferenceMetricsInterface):
    """Calculate the structural similarity index (SSIM) between two images.

    Attributes
    ----------
    score_val : float or None
        Score value of the SSIM metric.
    parameters : dict
        Dictionary containing the parameters for SSIM calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, optional
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the SSIM calculation. Passed to
        :py:func:`viqa.utils.load_data` and
        :py:func:`viqa.fr_metrics.ssim.structural_similarity`.
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
    ``data_range`` for image loading is also used for the SSIM calculation if the image
    type is integer and therefore must be set. The parameter is set through the
    constructor of the class and is passed to :py:meth:`score`. SSIM [1]_ is a
    full-reference IQA metric. It is based on the human visual system and is designed to
    predict the perceived quality of an image.

    See Also
    --------
    viqa.fr_metrics.uqi.UQI : Universal quality index.
    viqa.fr_metrics.msssim.MSSSIM : Multi-scale structural similarity index.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
        quality assessment: From error visibility to structural similarity. IEEE
        Transactions on Image Processing, 13(4), 600–612.
        https://doi.org/10.1109/TIP.2003.819861
    """

    def __init__(self, data_range=255, normalize=False, **kwargs):
        """Construct method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        self._name = "SSIM"

    def score(self, img_r, img_m, color_weights=None, **kwargs):
        """Calculate the structural similarity index (SSIM) between two images.

        Parameters
        ----------
        img_r : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
            Modified image to calculate score of.
        color_weights : np.ndarray, optional
            Weights for the color channels. The array must have the same length as the
            number of color channels in the images. Is only effective if
            ``chromatic=True`` is set during initialization.
        **kwargs : optional
            Additional parameters for the SSIM calculation. The keyword arguments are
            passed to :py:func:`viqa.fr_metrics.ssim.structural_similarity`.

        Returns
        -------
        score_val : float
            SSIM score value.

        Raises
        ------
        ValueError
            If ``color_weights`` are not set for chromatic images.

        Notes
        -----
        For color images, the metric is calculated channel-wise and the mean after
        weighting with the color weights is returned.
        """
        img_r, img_m = self.load_images(img_r, img_m)

        if self.parameters["chromatic"]:
            if color_weights is None:
                raise ValueError("Color weights must be set for chromatic images.")
            scores = []
            for channel in range(img_r.shape[-1]):
                score = structural_similarity(
                    img_r[..., channel],
                    img_m[..., channel],
                    data_range=self.parameters["data_range"],
                    **kwargs,
                )
                scores.append(score)
            score_val = (color_weights * np.array(scores)).mean()
        else:
            score_val = structural_similarity(
                img_r, img_m, data_range=self.parameters["data_range"], **kwargs
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
            If :py:attr:`score_val` is not available.
        """
        if self.score_val is not None:
            print("SSIM: {}".format(np.round(self.score_val, decimals)))
        else:
            warn("No score value for SSIM. Run score() first.", RuntimeWarning)


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
        odd value. If ``gaussian_weights`` is True, this is ignored and the
        window size will depend on ``sigma``.
    data_range : int, default=None
        Data range of the input images.
    gaussian_weights : bool, default=True
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
        Standard deviation for the Gaussian when ``gaussian_weights`` is True.
    mode : str, default='reflect'
        Determines how the array borders are handled. 'constant', 'edge', 'symmetric',
        'reflect' or 'wrap'.

        .. seealso::
            See Scipy documentation for :py:func:`scipy.ndimage.gaussian_filter` or
            :py:func:`scipy.ndimage.uniform_filter` for more information on the modes.

    cval : float, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default is 0.

    Returns
    -------
    ssim : float
        The mean structural similarity index over the image.

    Raises
    ------
    ValueError
        If ``K1``, ``K2`` or ``sigma`` are negative. \n
        If ``win_size`` exceeds image or is not an odd number.

    Warns
    -----
    RuntimeWarning
        If ``alpha``, ``beta`` or ``gamma`` are not integers.

    Notes
    -----
    To match the implementation in [1]_, set ``gaussian_weights`` to True and ``sigma``
    to 1.5. This code is adapted from :py:func:`skimage.metrics.structural_similarity`
    available under [2]_. The metric would possibly result in a value of nan in specific
    cases. To avoid this, the function replaces nan values with 1.0 before computing the
    final score.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
        quality assessment: From error visibility to structural similarity. IEEE
        Transactions on Image Processing, 13(4), 600–612.
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

    cov_norm = None
    k_1 = kwargs.pop("K1", 0.01)
    k_2 = kwargs.pop("K2", 0.03)
    sigma = kwargs.pop("sigma", 1.5)
    if k_1 < 0:
        raise ValueError("K1 must be positive")
    if k_2 < 0:
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
        raise ValueError("win_size must be odd.")

    ndim = img_r.ndim

    mode = kwargs.pop("mode", "reflect")
    cval = kwargs.pop("cval", 0)

    if gaussian_weights:
        filter_func = gaussian_filter
        filter_args = {"sigma": sigma, "truncate": truncate, "mode": mode, "cval": cval}
    else:
        filter_func = uniform_filter
        filter_args = {"size": win_size, "mode": mode, "cval": cval}

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
    img_r = img_r.astype(np.float32, copy=False)
    img_m = img_m.astype(np.float32, copy=False)

    n = win_size**ndim
    if not cov_norm:
        cov_norm = n / (n - 1)  # sample covariance

    # compute (weighted) means
    ux = filter_func(img_r, **filter_args)
    uy = filter_func(img_m, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(img_r * img_r, **filter_args)
    uyy = filter_func(img_m * img_m, **filter_args)
    uxy = filter_func(img_r * img_m, **filter_args)

    del img_r, img_m

    vx = cov_norm * (uxx - ux * ux)
    del uxx

    vy = cov_norm * (uyy - uy * uy)
    del uyy

    vxy = cov_norm * (uxy - ux * uy)
    del uxy

    c_1 = (k_1 * data_range) ** 2
    c_2 = (k_2 * data_range) ** 2
    c_3 = c_2 / 2

    stru = (vxy + c_3) / (np.sqrt(vx) * np.sqrt(vy) + c_3)
    del vxy
    # remove nan
    stru = np.nan_to_num(stru, nan=1.0)

    con = (2 * np.sqrt(vx) * np.sqrt(vy) + c_2) / (vx + vy + c_2)
    del vx, vy
    # remove nan
    con = np.nan_to_num(con, nan=1.0)

    lum = (2 * ux * uy + c_1) / (ux**2 + uy**2 + c_1)
    del ux, uy
    # remove nan
    lum = np.nan_to_num(lum, nan=1.0)

    ssim = (lum**alpha) * (con**beta) * (stru**gamma)
    del lum, con, stru

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    ssim = crop(ssim, pad).mean(dtype=np.float64)

    return ssim
