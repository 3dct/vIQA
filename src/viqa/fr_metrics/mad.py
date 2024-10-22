"""Module for the most apparent distortion (MAD) metric.

Notes
-----
The code is adapted from the original MATLAB code available under [1]_.

References
----------
.. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad (version 2011_10_07)

Examples
--------
    .. doctest-skip::

        >>> import numpy as np
        >>> from viqa import MAD
        >>> img_r = np.random.rand(256, 256)
        >>> img_m = np.random.rand(256, 256)
        >>> mad = MAD()
        >>> mad.score(img_r, img_m, data_range=1)

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

from typing import Any
from warnings import warn

import numpy as np
from scipy.ndimage import convolve
from tqdm.autonotebook import trange

from viqa._metrics import FullReferenceMetricsInterface
from viqa.fr_metrics.stat_utils import statisticscalc
from viqa.utils import (
    _extract_blocks,
    _fft,
    _ifft,
    _to_float,
    gabor_convolve,
)

# Global preinitialized variables
M = 0
N = 0
BLOCK_SIZE = 0
STRIDE = 0


class MAD(FullReferenceMetricsInterface):
    """Class to calculate the most apparent distortion (MAD) between two images.

    Attributes
    ----------
    score_val : float
        MAD score value of the last calculation.
    parameters : dict
        Dictionary containing the parameters for MAD calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, optional
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the MAD calculation. Passed to
        :py:func:`viqa.utils.load_data` and
        :py:func:`viqa.fr_metrics.mad.most_apparent_distortion`.
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
        If ``data_range`` is None.

    Warnings
    --------
    The metric is not yet validated. Use with caution.

    .. todo:: validate

    Notes
    -----
    ``data_range`` for image loading is also used for the MAD calculation if the image
    type is integer and therefore must be set. The parameter is set through the
    constructor of the class and is passed to :py:meth:`score`. MAD [1]_ is a
    full-reference IQA metric. It is based on the human visual system and is designed to
    predict the perceived quality of an image.

    References
    ----------
    .. [1] Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion:
        full-reference image quality assessment and the role of strategy. Journal of
        Electronic Imaging, 19(1), 011006. https://doi.org/10.1117/1.3267105
    """

    def __init__(self, data_range=255, normalize=False, **kwargs) -> None:
        """Construct method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        self._name = "MAD"

    def score(self, img_r, img_m, dim=None, im_slice=None, **kwargs):
        """Calculate the MAD between two images.

        The metric can be calculated for 2D and 3D images. If the images are 3D, the
        metric can be calculated for the full volume or for a given slice of the image
        by setting ``dim`` to the desired dimension and ``im_slice`` to the desired
        slice number.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.
        dim : {0, 1, 2}, optional
            MAD for 3D images is calculated as mean over all slices of the given
            dimension.
        im_slice : int, optional
            If given, MAD is calculated only for the given slice of the 3D image.
        **kwargs : optional
            Additional parameters for MAD calculation. The keyword arguments are passed
            to :py:func:`viqa.fr_metrics.mad.most_apparent_distortion`.

        Returns
        -------
        score_val : float
            MAD score value.

        Raises
        ------
        ValueError
            If invalid dimension given in ``dim``. \n
            If images are neither 2D nor 3D. \n
            If images are 3D, but ``dim`` is not given. \n
            If ``im_slice`` is given, but not an integer.

        Warns
        -----
        RuntimeWarning
            If dim or im_slice is given for 2D images. \n
            If im_slice is not given, but dim is given for 3D images, MAD is calculated
            for the full volume.

        Notes
        -----
        For 3D images if ``dim`` is given, but ``im_slice`` is not, the MAD is
        calculated for the full volume of the 3D image. This is implemented as `mean` of
        the MAD values of all slices of the given dimension. If ``dim`` is given and
        ``im_slice`` is given, the MAD is calculated for the given slice of the given
        dimension (represents a 2D metric of the given slice).
        """
        img_r, img_m = self.load_images(img_r, img_m)

        # Check if images are 2D or 3D
        if img_r.ndim == 3:
            if (
                dim is not None and type(im_slice) is int
            ):  # if dim and im_slice are given
                # Calculate MAD for given slice of given dimension
                match dim:
                    case 0:
                        score_val = most_apparent_distortion(
                            img_r[im_slice, :, :],
                            img_m[im_slice, :, :],
                            data_range=self.parameters["data_range"],
                            **kwargs,
                        )
                    case 1:
                        score_val = most_apparent_distortion(
                            img_r[:, im_slice, :],
                            img_m[:, im_slice, :],
                            data_range=self.parameters["data_range"],
                            **kwargs,
                        )
                    case 2:
                        score_val = most_apparent_distortion(
                            img_r[:, :, im_slice],
                            img_m[:, :, im_slice],
                            data_range=self.parameters["data_range"],
                            **kwargs,
                        )
                    case _:
                        raise ValueError(
                            "Invalid dim value. Must be integer of 0, 1 or 2."
                        )
            elif (
                dim is not None and im_slice is None
            ):  # if dim is given, but im_slice is not, calculate MAD for full volume
                warn(
                    "im_slice is not given. Calculating MAD for full volume.",
                    RuntimeWarning,
                )
                score_val = most_apparent_distortion_3d(
                    img_r,
                    img_m,
                    data_range=self.parameters["data_range"],
                    dim=dim,
                    **kwargs,
                )
            elif (
                dim is not None
                and type(im_slice) is not int
                or type(im_slice) is not None
            ):
                raise ValueError("im_slice must be an integer or None.")
            else:
                raise ValueError(
                    "If images are 3D, dim and im_slice (optional) must be given."
                )
        elif img_r.ndim == 2:
            if dim or im_slice:
                warn("dim and im_slice are ignored for 2D images.", RuntimeWarning)
            # Calculate MAD for 2D images
            score_val = most_apparent_distortion(
                img_r, img_m, data_range=self.parameters["data_range"], **kwargs
            )
        else:
            raise ValueError("Images must be 2D or 3D.")

        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the MAD score value of the last calculation.

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
            print("MAD: {}".format(np.round(self.score_val, decimals)))
        else:
            warn("No score value for MAD. Run score() first.", RuntimeWarning)


def most_apparent_distortion_3d(
    img_r: np.ndarray, img_m: np.ndarray, dim: int = 2, **kwargs: Any
) -> float:
    """Calculate the MAD for a 3D image.

    Parameters
    ----------
    img_r : np.ndarray
        Reference image to calculate score against
    img_m : np.ndarray
        Distorted image to calculate score of
    dim : {0, 1, 2}, default=2
        Dimension on which the slices are iterated.
    **kwargs : optional
            Additional parameters for MAD calculation. The keyword arguments are passed
            to :py:func:`viqa.fr_metrics.mad.most_apparent_distortion`.

    Returns
    -------
    score_val : float
        MAD score value as mean of MAD values of all slices of the given dimension.

    Raises
    ------
    ValueError
        If invalid dimension given in ``dim``.

    Warnings
    --------
    The metric is not yet validated. Use with caution.

    .. todo:: validate

    See Also
    --------
    viqa.fr_metrics.mad.most_apparent_distortion : Calculate the MAD between two images.

    References
    ----------
    .. [1] Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion:
        full-reference image quality assessment and the role of strategy. Journal of
        Electronic Imaging, 19(1), 011006. https://doi.org/10.1117/1.3267105
    """
    x, y, z = img_r.shape  # get image dimensions
    scores = []
    # Calculate MAD for all slices of the given dimension
    match dim:
        case 0:
            for slice_ in trange(x):
                scores.append(
                    most_apparent_distortion(
                        img_r[slice_, :, :], img_m[slice_, :, :], **kwargs
                    )
                )
        case 1:
            for slice_ in trange(y):
                scores.append(
                    most_apparent_distortion(
                        img_r[:, slice_, :], img_m[:, slice_, :], **kwargs
                    )
                )
        case 2:
            for slice_ in trange(z):
                scores.append(
                    most_apparent_distortion(
                        img_r[:, :, slice_], img_m[:, :, slice_], **kwargs
                    )
                )
        case _:
            raise ValueError("Invalid dim value. Must be integer of 0, 1 or 2.")
    return np.mean(np.array(scores))


def most_apparent_distortion(
    img_r: np.ndarray,
    img_m: np.ndarray,
    data_range: int = 255,
    block_size: int = 16,
    block_overlap: float = 0.75,
    beta_1: float = 0.467,
    beta_2: float = 0.130,
    thresh_1: float | None = None,
    thresh_2: float | None = None,
    **kwargs,
) -> float:
    """Calculate the most apparent distortion (MAD) between two images.

    Parameters
    ----------
    img_r : np.ndarray
        Reference image to calculate score against.
    img_m : np.ndarray
        Distorted image to calculate score of.
    data_range : int, default=255
        Data range of the input images.
    block_size : int, default=16
        Size of the blocks in the MAD calculation. Must be positive. Use 1 for no
        blocks.
    block_overlap : float, default=0.75
        Overlap of the blocks in the MAD calculation. Given as a fraction of
        ``block_size``.
    beta_1 : float, default=0.467
        Parameter for single metrics combination.
    beta_2 : float, default=0.130
        Parameter for single metrics combination.
    thresh_1 : float, optional
        Threshold for single metrics combination.
    thresh_2 : float, optional
        Threshold for single metrics combination.
    **kwargs : optional
        Additional parameters for MAD calculation.

    Returns
    -------
    mad_index : float
        MAD score value.

    Other Parameters
    ----------------
    account_monitor : bool, default False
        If True, the ``display_function`` of the monitor is taken into account.
    display_function : dict, optional
        Parameters of the display function of the monitor. Must be given if
        ``account_monitor`` is True.

        .. admonition:: Dictionary layout for ``display_function``

            disp_res : float, Display resolution. \n
            view_dis : float, Viewing distance. Same unit as ``disp_res``.

    luminance_function : dict, optional
        Parameters of the luminance function. If not given, default values for sRGB
        displays are used.

        .. admonition:: Dictionary layout for ``luminance_function``

            b : float, default=0.0 \n
            k : float, default=0.02874 \n
            gamma : float, default=2.2

    ms_scale : float, default=1
        Additional normalization parameter for the high-quality index.
    orientations_num : int, default 4
        Number of orientations for the log-Gabor filters. Passed to
        `.viqa.utils.gabor_convolve`.
    scales_num : int, default 5
        Number of scales for the log-Gabor filters. Passed to
        `.viqa.utils.gabor_convolve`.
    weights : list, default [0.5, 0.75, 1, 5, 6]
        Weights for the different scales of the log-Gabor filters. Must be of length
        ``scales_num``.
    csf_function : dict, optional
        Parameters for the contrast sensitivity function. If not given, default values
        for sRGB displays are used.

        .. admonition:: Dictionary layout for ``csf_function``

            lambda_csf : float, default=0.114 \n
            f_peak : float, default=7.8909

    Raises
    ------
    ValueError
        If ``block_size`` is larger than the image dimensions. \n
        If ``block_overlap`` is not between 0 and smaller than 1. \n
        If ``block_size`` is not positive. \n
        If ``weights`` is not of length ``scales_num``.

    Warns
    -----
    RuntimeWarning
        If either ``thresh_1`` or ``thresh_2`` and not both are given. \n
        If ``thresh_1`` and ``thresh_2`` and ``beta_1`` or ``beta_2`` are given.

    Warnings
    --------
    The metric is not yet validated. Use with caution.

    .. todo:: validate

    See Also
    --------
    viqa.fr_metrics.mad.most_apparent_distortion_3d : Calculate the MAD for a 3D image.

    Notes
    -----
    The metric is calculated as combination of two single metrics. One for high quality
    and one for low quality of the image. The parameters ``beta_1``, ``beta_2``,
    ``thresh_1`` and ``thresh_2`` determine the weighting of the two combined single
    metrics. If ``thresh_1`` and ``thresh_2`` are given, ``beta_1`` and ``beta_2`` are
    calculated from them, else ``beta_1`` and ``beta_2`` or their default values are
    used. The values to be set for ``thresh_1`` and ``thresh_2`` that lead to the
    default values ``beta_1=0.467`` and ``beta_2=0.130`` are ``thresh_1=2.55`` and
    ``thresh_2=3.35``. These need not to be set, since automatic values for ``beta_1``
    and ``beta_2`` are used when they are not given as parameter. For more information
    see [1]_. The code is adapted from the original MATLAB code available under [2]_.

    References
    ----------
    .. [1] Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion:
        full-reference image quality assessment and the role of strategy. Journal of
        Electronic Imaging, 19(1), 011006. https://doi.org/10.1117/1.3267105
    .. [2] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad
        (version 2011_10_07)
    """
    # Authors
    # -------
    # Author: Eric Larson
    # Department of Electrical and Computer Engineering
    # Oklahoma State University, 2008
    # University Of Washington Seattle, 2009
    # Image Coding and Analysis Lab
    #
    # Translation and Adaption: Lukas Behammer
    # Research Center Wels
    # University of Applied Sciences Upper Austria, 2023
    # CT Research Group
    #
    # Modifications
    # -------------
    # Original code, 2008, Eric Larson
    # Translated to Python and Adapted, 2024, Lukas Behammer
    #
    # License
    # -------
    # No License attached to the original code.
    # Permission to use the code was granted by Eric Larson.

    # Set global variables
    global M, N
    M, N = img_r.shape
    global BLOCK_SIZE, STRIDE
    if block_size > M or block_size > N:
        raise ValueError("block_size must be smaller than the image dimensions.")
    elif block_size > 0:
        BLOCK_SIZE = block_size
        if block_overlap >= 1 or block_overlap < 0:
            raise ValueError("block_overlap must be between 0 and smaller than 1.")
        STRIDE = BLOCK_SIZE - int(block_overlap * BLOCK_SIZE)
    else:
        raise ValueError("block_size must be positive.")

    # Parameters for single metrics combination
    if (thresh_1 or thresh_2) and not (thresh_1 and thresh_2):
        warn(
            "thresh_1 and thresh_2 must be given together. Using default "
            "values for beta_1 and beta_2.",
            RuntimeWarning,
        )
    elif thresh_1 and thresh_2:
        beta_1 = np.exp(-thresh_1 / thresh_2)
        beta_2 = 1 / (np.log(10) * thresh_2)
        if beta_1 or beta_2:
            warn(
                "thresh_1 and thresh_2 are given. Ignoring beta_1 and beta_2.",
                RuntimeWarning,
            )

    # Hiqh quality index
    d_detect = _high_quality(img_r, img_m, data_range=data_range, **kwargs)
    # Low quality index
    d_appear = _low_quality(img_r, img_m, **kwargs)

    # Combine single metrics with weighting
    alpha = 1 / (1 + beta_1 * d_detect**beta_2)  # weighting factor
    mad_index = d_detect**alpha * d_appear ** (1 - alpha)
    return mad_index


def _high_quality(img_r: np.ndarray, img_m: np.ndarray, **kwargs) -> float:
    """Calculate the high-quality index of MAD.

    Notes
    -----
    The code is adapted from the original MATLAB code available under [1]_.

    References
    ----------
    .. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad
        (version 2011_10_07)
    """
    # Authors
    # -------
    # Author: Eric Larson
    # Department of Electrical and Computer Engineering
    # Oklahoma State University, 2008
    # University Of Washington Seattle, 2009
    # Image Coding and Analysis Lab
    #
    # Translation and Adaption: Lukas Behammer
    # Research Center Wels
    # University of Applied Sciences Upper Austria, 2023
    # CT Research Group
    #
    # Modifications
    # -------------
    # Original code, 2008, Eric Larson
    # Translated to Python and Adapted, 2024, Lukas Behammer
    #
    # License
    # -------
    # No License attached to the original code.
    # Permission to use the code was granted by Eric Larson.

    account_monitor = kwargs.pop("account_monitor", False)
    # Account for display function of monitor
    if account_monitor:
        if "display_function" not in kwargs:
            raise ValueError(
                "If account_monitor is True, display_function must be given."
            )
        display_function = kwargs.pop("display_function")
        cycles_per_degree = (
            display_function["disp_res"]
            * display_function["view_dis"]
            * np.tan(np.pi / 180)
        ) / 2
    else:
        cycles_per_degree = 32

    csf_function = kwargs.pop("csf_function", {"lambda_csf": 0.114, "f_peak": 7.8909})
    # Calculate contrast sensitivity function
    csf = _contrast_sensitivity_function(
        M,
        N,
        cycles_per_degree,
        lambda_csf=csf_function["lambda_csf"],
        f_peak=csf_function["f_peak"],
    )

    # Convert to perceived lightness
    luminance_function = kwargs.pop("luminance_function", {"k": 0.02874, "gamma": 2.2})

    data_range = kwargs.pop("data_range", 255)
    lum_r = _pixel_to_lightness(img_r, data_range=data_range, **luminance_function)
    lum_m = _pixel_to_lightness(img_m, data_range=data_range, **luminance_function)

    # Fourier transform
    lum_r_fft = _fft(lum_r)
    lum_m_fft = _fft(lum_m)

    i_org = np.real(_ifft(csf * lum_r_fft))
    i_dst = np.real(_ifft(csf * lum_m_fft))

    i_err = i_dst - i_org  # error image

    # Contrast masking
    # TODO: change to generator
    # i_org_blocks = np.fromiter(
    #     _extract_blocks(
    #         i_org,
    #         block_size=BLOCK_SIZE,
    #         stride=STRIDE
    #     ),
    #     dtype=np.float32,
    #     count=-1
    # )
    # i_err_blocks = np.fromiter(
    #     _extract_blocks(
    #         i_err,
    #         block_size=BLOCK_SIZE,
    #         stride=STRIDE
    #     ),
    #     dtype=np.float32,
    #     count=-1
    # )
    i_org_blocks = _extract_blocks(i_org, block_size=BLOCK_SIZE, stride=STRIDE)
    i_err_blocks = _extract_blocks(i_err, block_size=BLOCK_SIZE, stride=STRIDE)

    # Calculate local statistics
    mu_org_p = np.mean(i_org_blocks, axis=(1, 2))
    std_err_p = np.std(i_err_blocks, axis=(1, 2), ddof=1)

    # std_org = _min_std(i_org, block_size=BLOCK_SIZE, stride=STRIDE)  # Legacy function
    std_org = statisticscalc.minstd(i_org, BLOCK_SIZE, STRIDE)

    mu_org = np.zeros(i_org.shape)
    std_err = np.zeros(i_err.shape)

    # Assign local statistics to image blocks of size stride x stride
    block_n = 0
    for x in range(0, i_org.shape[0] - STRIDE * 3, STRIDE):
        for y in range(0, i_org.shape[1] - STRIDE * 3, STRIDE):
            mu_org[x : x + STRIDE, y : y + STRIDE] = mu_org_p[block_n]
            std_err[x : x + STRIDE, y : y + STRIDE] = std_err_p[block_n]
            block_n += 1
    del mu_org_p, std_err_p  # free memory

    # Calculate contrast of reference and error image
    c_org = std_org / mu_org
    c_err = np.zeros(std_err.shape)
    _ = np.divide(std_err, mu_org, out=c_err, where=mu_org > 0.5)

    # Create mask
    # log(Contrast of ref-dst) vs. log(Contrast of reference)
    #               /
    #              /
    #             / _| <- c_slope
    #            /
    # ----------+ < - cd_thresh(y axis height)
    #           /\
    #           ||
    #        ci_thresh(x axis value)
    ci_org = np.log(c_org)
    ci_err = np.log(c_err)
    c_slope = 1
    ci_thresh = -5
    cd_thresh = -5
    tmp = c_slope * (ci_org - ci_thresh) + cd_thresh
    cond_1 = np.logical_and(ci_err > tmp, ci_org > ci_thresh)
    cond_2 = np.logical_and(ci_err > cd_thresh, ci_thresh >= ci_org)

    ms_scale = kwargs.pop("ms_scale", 1)  # additional normalization parameter
    msk = np.zeros(c_err.shape)
    _ = np.subtract(
        ci_err, tmp, out=msk, where=cond_1
    )  # contrast of heavy distortion - (0.75 * contrast of ref)
    _ = np.subtract(
        ci_err, cd_thresh, out=msk, where=cond_2
    )  # contrast of low distortion - threshold
    msk /= ms_scale

    # Use lum_mse and weight by distortion mask
    win = np.ones((BLOCK_SIZE, BLOCK_SIZE)) / BLOCK_SIZE**2
    lum_mse = convolve((_to_float(img_r) - _to_float(img_m)) ** 2, win, mode="reflect")

    # Kill the edges
    mp = msk * lum_mse
    mp2 = mp[BLOCK_SIZE:-BLOCK_SIZE, BLOCK_SIZE:-BLOCK_SIZE]

    # Calculate high quality index by using the 2-norm
    d_detect = (
        np.linalg.norm(mp2) / np.sqrt(np.prod(mp2.shape)) * 200
    )  # fixed factor of 200
    return d_detect


def _low_quality(img_r: np.ndarray, img_m: np.ndarray, **kwargs) -> float:
    """Calculate the low-quality index of MAD.

    Notes
    -----
    The code is adapted from the original MATLAB code available under [1]_.

    References
    ----------
    .. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad
        (version 2011_10_07)
    """
    # Authors
    # -------
    # Author: Eric Larson
    # Department of Electrical and Computer Engineering
    # Oklahoma State University, 2008
    # University Of Washington Seattle, 2009
    # Image Coding and Analysis Lab
    #
    # Translation: Lukas Behammer
    # Research Center Wels
    # University of Applied Sciences Upper Austria, 2023
    # CT Research Group
    #
    # Modifications
    # -------------
    # Original code, 2008, Eric Larson
    # Translated to Python, 2024, Lukas Behammer
    #
    # License
    # -------
    # No License attached to the original code.
    # Permission to use the code was granted by Eric Larson.

    orientations_num = kwargs.pop("orientations_num", 4)
    scales_num = kwargs.pop("scales_num", 5)
    weights = kwargs.pop("weights", [0.5, 0.75, 1, 5, 6])
    weights /= np.sum(weights)
    if len(weights) != scales_num:
        raise ValueError(f"weights must be of length scales_num ({scales_num}).")

    # Decompose using log-Gabor filters
    gabor_org = gabor_convolve(
        img_m,
        scales_num=scales_num,
        orientations_num=orientations_num,
        min_wavelength=3,
        wavelength_scaling=3,
        bandwidth_param=0.55,
        d_theta_on_sigma=1.5,
    )
    gabor_dst = gabor_convolve(
        img_r,
        scales_num=scales_num,
        orientations_num=orientations_num,
        min_wavelength=3,
        wavelength_scaling=3,
        bandwidth_param=0.55,
        d_theta_on_sigma=1.5,
    )

    # Calculate statistics for each filterband
    stats = np.zeros((M, N))
    for scale_n in range(scales_num):
        for orientation_n in range(orientations_num):
            # Legacy function
            # std_ref, skw_ref, krt_ref = _get_statistics(
            #     np.abs(gabor_org[scale_n, orientation_n]),
            #     block_size=BLOCK_SIZE,
            #     stride=STRIDE,
            # )
            # std_dst, skw_dst, krt_dst = _get_statistics(
            #     np.abs(gabor_dst[scale_n, orientation_n]),
            #     block_size=BLOCK_SIZE,
            #     stride=STRIDE,
            # )

            std_ref, skw_ref, krt_ref = statisticscalc.getstatistics(
                np.abs(gabor_org[scale_n, orientation_n]),
                BLOCK_SIZE,
                STRIDE,
            )
            std_dst, skw_dst, krt_dst = statisticscalc.getstatistics(
                np.abs(gabor_dst[scale_n, orientation_n]),
                BLOCK_SIZE,
                STRIDE,
            )
            # Combine statistics
            stats += weights[scale_n] * (
                np.abs(std_ref - std_dst)
                + 2 * np.abs(skw_ref - skw_dst)
                + np.abs(krt_ref - krt_dst)
            )

    # Kill the edges
    mp2 = stats[BLOCK_SIZE:-BLOCK_SIZE, BLOCK_SIZE:-BLOCK_SIZE]

    # Calculate low quality index by using the 2-norm
    d_appear = np.linalg.norm(mp2) / np.sqrt(np.prod(mp2.shape))
    return d_appear


def _pixel_to_lightness(
    img: np.ndarray,
    b: int = 0,
    k: float = 0.02874,
    gamma: float = 2.2,
    data_range: int = 255,
) -> np.ndarray:
    """Convert an image to perceived lightness."""
    # Authors
    # -------
    # Author: Eric Larson
    # Department of Electrical and Computer Engineering
    # Oklahoma State University, 2008
    # University Of Washington Seattle, 2009
    # Image Coding and Analysis Lab
    #
    # Translation: Lukas Behammer
    # Research Center Wels
    # University of Applied Sciences Upper Austria, 2023
    # CT Research Group
    #
    # Modifications
    # -------------
    # Original code, 2008, Eric Larson
    # Translated to Python, 2024, Lukas Behammer
    #
    # License
    # -------
    # No License attached to the original code.
    # Permission to use the code was granted by Eric Larson.

    if issubclass(img.dtype.type, np.integer):  # if image is integer
        # Create LUT
        lut = np.arange(0, data_range + 1, dtype=np.float64)
        lut = b + k * lut ** (gamma / 3)
        img_lum = lut[img]  # apply LUT
    else:  # if image is float
        img_lum = b + k * img ** (gamma / 3)
    return img_lum


def _contrast_sensitivity_function(m: int, n: int, nfreq: int, **kwargs) -> np.ndarray:
    """
    Calculate the contrast sensitivity function.

    Notes
    -----
    The code is adapted from the original MATLAB code available under [1]_.

    References
    ----------
    .. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad
        (version 2011_10_07)
    """
    # Authors
    # -------
    # Author: Eric Larson
    # Department of Electrical and Computer Engineering
    # Oklahoma State University, 2008
    # University Of Washington Seattle, 2009
    # Image Coding and Analysis Lab
    #
    # Translation: Lukas Behammer
    # Research Center Wels
    # University of Applied Sciences Upper Austria, 2023
    # CT Research Group
    #
    # Modifications
    # -------------
    # Original code, 2008, Eric Larson
    # Translated to Python, 2024, Lukas Behammer
    #
    # License
    # -------
    # No License attached to the original code.
    # Permission to use the code was granted by Eric Larson.

    # Create a meshgrid that represents the spatial domain of the image.
    x_plane, y_plane = np.meshgrid(
        np.arange(-n / 2 + 0.5, n / 2 + 0.5), np.arange(-m / 2 + 0.5, m / 2 + 0.5)
    )
    plane = (x_plane + 1j * y_plane) * 2 * nfreq / n  # convert to frequency domain
    rad_freq = np.abs(plane)  # radial frequency

    # w is a symmetry parameter that gives approx. 3dB down along the diagonals
    w = 0.7
    theta = np.angle(plane)
    # s is a function of theta that adjusts the radial frequency based on the direction
    # of each point in the frequency domain.
    s = ((1 - w) / 2) * np.cos(4 * theta) + ((1 + w) / 2)
    rad_freq /= s

    # Parameters for the contrast sensitivity function
    lambda_csf = kwargs.pop("lambda_csf", 0.114)
    f_peak = kwargs.pop("f_peak", 7.8909)
    # Calculate contrast sensitivity function
    cond = rad_freq < f_peak
    csf = (
        2.6
        * (0.0192 + lambda_csf * rad_freq)
        * np.exp(-((lambda_csf * rad_freq) ** 1.1))
    )
    csf[cond] = 0.9809
    return csf
