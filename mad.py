"""Module for the most apparent disorder (MAD) metric.

Notes
------
The code is adapted from the original MATLAB code available under [1]_. \n

References
----------
.. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad (version 2011_10_07)

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
from utils import _check_imgs, _to_float, extract_blocks, _ifft, _fft, gabor_convolve
import numpy as np
from warnings import warn
from scipy.ndimage import convolve
from scipy.stats import skew, kurtosis

# Global preinitialized variables
M = 0
N = 0
BLOCK_SIZE = 0
STRIDE = 0


class MAD(metrics.FullReferenceMetricsInterface):
    """Class to calculate the most apparent disorder (MAD) between two images.

    Parameters
    ----------
    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to `utils.load_data`.
        See below for details.

    Attributes
    ----------
    score_val : float
        MAD score value of the last calculation.

    Other Parameters
    ----------------
    data_range : {1, 255, 65535}, optional
        Data range of the returned data in data loading. Can be omitted if `normalize` is False.
    normalize : bool, default False
        If True, the input images are normalized to the `data_range` argument.
    batch : bool, default False
        If True, the input images are expected to be given as path to a folder containing the images.
        .. note:: Currently not supported. Added for later implementation.
    chromatic : bool, default False
        If True, the input images are expected to be RGB images. Currently not supported.


    Notes
    -----
    MAD [1]_ is a full-reference IQA metric. It is based on the human visual system and is designed to predict the
    perceived quality of an image.

    References
    ----------
    .. [1] Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion: full-reference image quality assessment
           and the role of strategy. Journal of Electronic Imaging, 19(1), 011006. https://doi.org/10.1117/1.3267105
    """

    def __init__(self, data_range=None, normalize=False, batch=False, **kwargs) -> None:
        """Constructor method"""
        super().__init__(data_range=data_range, normalize=normalize, batch=batch)
        self._parameters.update(**kwargs)

    def score(self, img_r, img_m, dim=None, im_slice=None, **kwargs):
        """Calculates the MAD between two images.

        The metric can be calculated for 2D and 3D images. If the images are 3D, the metric can be calculated for the
        full volume or for a given slice of the image by setting the parameter dim to the desired dimension and
        im_slice to the desired slice number.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of
        dim : {0, 1, 2}, optional
            If given, the MAD is calculated only for the given slice of the 3D image.
        im_slice : int, optional
            If given, the MAD is calculated only for the given slice of the 3D image.
        **kwargs : optional
            Additional parameters for MAD calculation. The keyword arguments are passed to
            `vIQA.mad.most_apparent_disorder()`.

        Returns
        -------
        score_val : float
            MAD score value.

        Raises
        ------
        ValueError
            If invalid dimension given in parameter dim.
        ValueError
            If images are neither 2D nor 3D.
        ValueError
            If images are 3D, but dim is not given.

        Warns
        -----
        RuntimeWarning
            If dim or im_slice is given for 2D images.

        Notes
        -----
        For 3D images if dim is given, but im_slice is not, the MAD is calculated for the full volume of the 3D image.
        This is implemented as mean of the MAD values of all slices of the given dimension. If dim is given and
        im_slice is given, the MAD is calculated for the given slice of the given dimension (represents a 2D metric of
        the given slice).
        """

        img_r, img_m = _check_imgs(img_r, img_m, data_range=self._parameters['data_range'],
                                   normalize=self._parameters['normalize'], batch=self._parameters['batch'])

        # Check if images are 2D or 3D
        if img_r.ndim == 3:
            if dim and im_slice:  # if dim and im_slice are given
                # Calculate MAD for given slice of given dimension
                match dim:
                    case 0:
                        score_val = most_apparent_disorder(img_r[im_slice, :, :], img_m[im_slice, :, :], **kwargs)
                    case 1:
                        score_val = most_apparent_disorder(img_r[:, im_slice, :], img_m[:, im_slice, :], **kwargs)
                    case 2:
                        score_val = most_apparent_disorder(img_r[:, :, im_slice], img_m[:, :, im_slice], **kwargs)
                    case _:
                        raise ValueError('Invalid dim value. Must be 0, 1 or 2.')
            elif dim and not im_slice:  # if dim is given, but im_slice is not, calculate MAD for full volume
                score_val = most_apparent_disorder_3d(img_r, img_m, dim=dim, **kwargs)
            else:
                raise ValueError('If images are 3D, dim and im_slice (optional) must be given.')
        elif img_r.ndim == 2:
            if dim or im_slice:
                warn('dim and im_slice are ignored for 2D images.', RuntimeWarning)
            # Calculate MAD for 2D images
            score_val = most_apparent_disorder(img_r, img_m, **kwargs)
        else:
            raise ValueError('Images must be 2D or 3D.')

        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Prints the MAD score value of the last calculation.

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
            print('MAD: {}'.format(round(self.score_val, decimals)))
        else:
            warn('No score value for MAD. Run score() first.', RuntimeWarning)


def most_apparent_disorder_3d(img_r, img_m, dim=2, **kwargs):
    """Calculates the MAD for a 3D image.

    Parameters
    ----------
    img_r : np.ndarray or Tensor or str or os.PathLike
        Reference image to calculate score against
    img_m : np.ndarray or Tensor or str or os.PathLike
        Distorted image to calculate score of
    dim : {0, 1, 2}, default=2
        Dimension to calculate MAD for.
    **kwargs : optional
            Additional parameters for MAD calculation. The keyword arguments are passed to
            `vIQA.mad.most_apparent_disorder()`.

    Returns
    -------
    score_val : float
        MAD score value as mean of MAD values of all slices of the given dimension.

    Raises
    ------
    ValueError
        If invalid dimension given in parameter dim.

    References
    ----------
    .. [1] Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion: full-reference image quality assessment
           and the role of strategy. Journal of Electronic Imaging, 19(1), 011006. https://doi.org/10.1117/1.3267105
    """

    x, y, z = img_r.shape  # get image dimensions
    scores = []
    # calculate MAD for all slices of the given dimension
    match dim:
        case 0:
            for slice_ in range(x):
                scores.append(most_apparent_disorder(img_r[slice_, :, :], img_m[slice_, :, :], **kwargs))
        case 1:
            for slice_ in range(y):
                scores.append(most_apparent_disorder(img_r[:, slice_, :], img_m[:, slice_, :], **kwargs))
        case 2:
            for slice_ in range(z):
                scores.append(most_apparent_disorder(img_r[:, :, slice_], img_m[:, :, slice_], **kwargs))
        case _:
            raise ValueError('Invalid dim value. Must be 0, 1 or 2.')
    return np.mean(np.array(scores))


def most_apparent_disorder(img_r, img_m, block_size=16, block_overlap=0.75, beta_1=0.467, beta_2=0.130, thresh_1=None,
                           thresh_2=None, **kwargs):
    """Calculates the most apparent disorder (MAD) between two images.

    Parameters
    ----------
    img_r : np.ndarray or Tensor or str or os.PathLike
        Reference image to calculate score against
    img_m : np.ndarray or Tensor or str or os.PathLike
        Distorted image to calculate score of
    block_size : int, default=16
        Size of the blocks in the MAD calculation. Must be positive.
    block_overlap : float, default=0.75
        Overlap of the blocks in the MAD calculation. Given as a fraction of the block size.
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

    Raises
    ------
    ValueError
        If `block_size` is not positive.
    ValueError
        If `weights` is not of length `scales_num`.

    Other Parameters
    ----------------
    account_monitor : bool, default False
        If True, the display function of the monitor is taken into account.
    display_function : dict, optional
        Parameters of the display function of the monitor. Must be given if `account_monitor` is True.
        disp_res : float
            Display resolution.
        view_dis : float
            Viewing distance. Same unit as `disp_res`.
    luminance_function : dict, optional
        Parameters of the luminance function. If not given, default values for sRGB displays are used.
        b : float, default=0.0
        k : float, default=0.02874
        gamma : float, default=2.2
    ms_scale : float, default=1
        Additional normalization parameter for the high quality index.
    orientations_num : int, default 4
        Number of orientations for the log-Gabor filters.
    scales_num : int, default 5
        Number of scales for the log-Gabor filters.
    weights : list, default [0.5, 0.75, 1, 5, 6]
        Weights for the different scales of the log-Gabor filters. Must be of `length scales_num`.

    Notes
    -----
    The metric is calculated as combination of two single metrics. One for high quality and one for low quality of the
    image. The parameters beta_1, beta_2, thresh_1 and thresh_2 determine the weighting of the two combined single
    metrics. If thresh_1 and thresh_2 are given, beta_1 and beta_2 are calculated from them, else beta_1 and beta_2 or
    their default values are used. For more information see [1]_. The code is adapted from the original MATLAB code
    available under [2]_.

    References
    ----------
    .. [1] Larson, E. C., & Chandler, D. M. (2010). Most apparent distortion: full-reference image quality assessment
           and the role of strategy. Journal of Electronic Imaging, 19(1), 011006. https://doi.org/10.1117/1.3267105
    .. [2] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad (version 2011_10_07)
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

    # TODO: add option for block_size = 0
    # Set global variables
    global BLOCK_SIZE, STRIDE
    if block_size > 0:
        BLOCK_SIZE = block_size
    else:
        raise ValueError('block_size must be positive.')
    STRIDE = BLOCK_SIZE - int(block_overlap * BLOCK_SIZE)
    global M, N
    M, N = img_r.shape

    # Parameters for single metrics combination
    if thresh_1 and thresh_2:
        beta_1 = np.exp(-thresh_1 / thresh_2)
        beta_2 = 1 / (np.log(10) * thresh_2)

    # Hiqh quality index
    d_detect = _high_quality(img_r, img_m, **kwargs)
    # Low quality index
    d_appear = _low_quality(img_r, img_m, **kwargs)

    # Combine single metrics with weighting
    alpha = 1 / (1 + beta_1 * d_detect ** beta_2)  # weighting factor
    mad_index = d_detect ** alpha * d_appear ** (1 - alpha)
    return mad_index


def _high_quality(img_r, img_m, **kwargs):
    """Calculates the high quality index of MAD.

    Notes
    ------
    The code is adapted from the original MATLAB code available under [1]_. \n

    References
    ----------
    .. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad (version 2011_10_07)
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

    # TODO: add comments
    account_monitor = kwargs.pop('account_monitor', False)
    if account_monitor:
        assert 'display_function' in kwargs, 'If account_monitor is True, display_function must be given.'
        display_function = kwargs.pop('display_function')
        cycles_per_degree = (display_function['disp_res']*display_function['view_dis']*np.tan(np.pi/180))/2
    else:
        cycles_per_degree = 32

    csf = _contrast_sensitivity_function(M, N, cycles_per_degree, lambda_=0.114, f_peak=7.8909)

    # Convert to perceived lightness
    luminance_function = kwargs.pop('luminance_function', {'k': 0.02874, 'gamma': 2.2})

    lum_r = _pixel_to_lightness(img_r, **luminance_function)
    lum_m = _pixel_to_lightness(img_m, **luminance_function)

    # Fourier transform
    lum_r_fft = _fft(lum_r)
    lum_m_fft = _fft(lum_m)

    i_org = np.real(_ifft(csf * lum_r_fft))
    i_dst = np.real(_ifft(csf * lum_m_fft))

    i_err = i_dst - i_org

    # Contrast masking
    i_org_blocks = extract_blocks(i_org, block_size=BLOCK_SIZE, stride=STRIDE)
    i_err_blocks = extract_blocks(i_err, block_size=BLOCK_SIZE, stride=STRIDE)

    # Calculate local statistics
    mu_org_p = np.mean(i_org_blocks, axis=(1, 2))
    std_err_p = np.std(i_err_blocks, axis=(1, 2), ddof=1)

    std_org = _min_std(i_org)

    mu_org = np.zeros(i_org.shape)
    std_err = np.zeros(i_err.shape)

    block_n = 0
    for x in range(0, i_org.shape[0] - STRIDE*3, STRIDE):
        for y in range(0, i_org.shape[1] - STRIDE*3, STRIDE):
            mu_org[x:x+STRIDE, y:y+STRIDE] = mu_org_p[block_n]
            std_err[x:x+STRIDE, y:y+STRIDE] = std_err_p[block_n]
            block_n += 1
    del mu_org_p, std_err_p  # free memory

    c_org = std_org / mu_org
    c_err = np.zeros(std_err.shape)
    _ = np.divide(std_err, mu_org, out=c_err, where=mu_org > 0.5)

    ci_org = np.log(c_org)
    ci_err = np.log(c_err)

    c_slope = 1
    ci_thrsh = -5
    cd_thrsh = -5
    tmp = c_slope * (ci_org - ci_thrsh) + cd_thrsh
    cond_1 = np.logical_and(ci_err > tmp, ci_org > ci_thrsh)
    cond_2 = np.logical_and(ci_err > cd_thrsh, ci_thrsh >= ci_org)

    ms_scale = kwargs.pop('ms_scale', 1)  # additional normalization parameter
    msk = np.zeros(c_err.shape)
    _ = np.subtract(ci_err, tmp, out=msk, where=cond_1)
    _ = np.subtract(ci_err, cd_thrsh, out=msk, where=cond_2)
    msk /= ms_scale

    win = np.ones((BLOCK_SIZE, BLOCK_SIZE)) / BLOCK_SIZE ** 2
    lmse = convolve((_to_float(img_r) - _to_float(img_m)) ** 2, win, mode='reflect')

    mp = msk * lmse
    mp2 = mp[BLOCK_SIZE:-BLOCK_SIZE, BLOCK_SIZE:-BLOCK_SIZE]

    d_detect = np.linalg.norm(mp2) / np.sqrt(np.prod(mp2.shape)) * 200
    return d_detect


def _low_quality(img_r, img_m, **kwargs):
    """Calculates the low quality index of MAD.

    Notes
    ------
    The code is adapted from the original MATLAB code available under [1]_. \n

    References
    ----------
    .. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad (version 2011_10_07)
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

    # TODO: add comments
    orientations_num = kwargs.pop('orientations_num', 4)
    scales_num = kwargs.pop('scales_num', 5)
    weights = kwargs.pop('weights', [0.5, 0.75, 1, 5, 6])
    weights /= np.sum(weights)
    if len(weights) != scales_num:
        raise ValueError('weights must be of length scales_num.')

    gabor_org = gabor_convolve(img_m, scales_num=scales_num, orientations_num=orientations_num, min_wavelength=3,
                               wavelength_scaling=3, bandwidth_param=0.55, d_theta_on_sigma=1.5)
    gabor_dst = gabor_convolve(img_r, scales_num=scales_num, orientations_num=orientations_num, min_wavelength=3,
                               wavelength_scaling=3, bandwidth_param=0.55, d_theta_on_sigma=1.5)

    stats = np.zeros((M, N))
    for scale_n in range(scales_num):
        for orientation_n in range(orientations_num):
            std_ref, skw_ref, krt_ref = _get_statistics(np.abs(gabor_org[scale_n, orientation_n]),
                                                        block_size=BLOCK_SIZE, stride=STRIDE)
            std_dst, skw_dst, krt_dst = _get_statistics(np.abs(gabor_dst[scale_n, orientation_n]),
                                                        block_size=BLOCK_SIZE, stride=STRIDE)

            stats += weights[scale_n] * (
                        np.abs(std_ref - std_dst) + 2 * np.abs(skw_ref - skw_dst) + np.abs(krt_ref - krt_dst))

    mp2 = stats[BLOCK_SIZE:-BLOCK_SIZE, BLOCK_SIZE:-BLOCK_SIZE]
    d_appear = np.linalg.norm(mp2) / np.sqrt(np.prod(mp2.shape))
    return d_appear


def _pixel_to_lightness(img, b=0, k=0.02874, gamma=2.2):
    """Converts an image to perceived lightness."""

    if issubclass(img.dtype.type, np.integer):  # if image is integer
        # Create LUT
        lut = np.arange(0, 256)
        lut = b + k * lut**(gamma/3)
        img_lum = lut[img]  # apply LUT
    else:  # if image is float
        img_lum = b + k * img**(gamma/3)
    return img_lum


def _contrast_sensitivity_function(m, n, nfreq, **kwargs):
    """
    Calculates the contrast sensitivity function.

    Notes
    ------
    The code is adapted from the original MATLAB code available under [1]_. \n

    References
    ----------
    .. [1] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad (version 2011_10_07)
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

    # TODO: add comments
    x_plane, y_plane = np.meshgrid(np.arange(-n/2 + 0.5, n/2 + 0.5), np.arange(-m/2 + 0.5, m/2 + 0.5))
    plane = (x_plane + 1j*y_plane)*2 * nfreq/n
    rad_freq = np.abs(plane)  # radial frequency

    # w is a symmetry parameter that gives approx. 3dB down along the diagonals
    w = 0.7
    theta = np.angle(plane)
    s = ((1-w)/2)*np.cos(4*theta) + ((1+w)/2)
    rad_freq /= s

    lambda_ = kwargs.pop('lambda_', 0.114)
    f_peak = kwargs.pop('f_peak', 7.8909)
    cond = rad_freq < f_peak
    csf = 2.6*(0.0192 + lambda_*rad_freq)*np.exp(-(lambda_*rad_freq)**1.1)
    csf[cond] = 0.9809

    return csf


def _min_std(image):
    """Calculates the minimum standard deviation of blocks of a given image."""

    # TODO: add comments
    tmp = np.empty(image.shape)
    stdout = np.empty(image.shape)
    for i in range(0, M - 15, 4):
        for j in range(0, N - 15, 4):
            mean = 0
            for u in range(i, i + 8):
                for v in range(j, j + 8):
                    mean += image[u, v]
            mean /= 64

            stdev = 0
            for u in range(i, i + 8):
                for v in range(j, j + 8):
                    stdev += (image[u, v] - mean) ** 2
            stdev = np.sqrt(stdev / 63)

            for u in range(i, i + 4):
                for v in range(j, j + 4):
                    tmp[u, v] = stdev
                    stdout[u, v] = stdev

    for i in range(0, M - 15, 4):
        for j in range(0, N - 15, 4):
            val = tmp[i, j]
            for u in range(i, i + 8, 5):
                for v in range(j, j + 8, 5):
                    if tmp[u, v] < val:
                        val = tmp[u, v]

            for u in range(i, i + 4):
                for v in range(j, j + 4):
                    stdout[u, v] = val

    return stdout


def _get_statistics(image, block_size, stride):
    """Calculates the statistics of blocks of a given image."""

    # TODO: add comments
    stdout = np.empty(image.shape)
    skwout = np.empty(image.shape)
    krtout = np.empty(image.shape)
    for i in range(0, M-(block_size-1), stride):
        for j in range(0, N-(block_size-1), stride):
            mean = 0
            for u in range(i, i + block_size):
                for v in range(j, j + block_size):
                    mean += image[u, v]
            mean /= block_size ** 2

            std = 0
            skw = 0
            krt = 0
            for u in range(i, i + block_size):
                for v in range(j, j + block_size):
                    tmp = (image[u, v] - mean)
                    std += tmp ** 2
                    skw += tmp ** 3
                    krt += tmp ** 4
            stmp = np.sqrt(std / (block_size**2))
            stdev = np.sqrt(std / (block_size**2 - 1))

            if stmp != 0:
                skw = skw / (block_size**2 * stmp**3)
                krt = krt / (block_size**2 * stmp**4)  # -3 in kurtosis definition
            else:
                skw = 0
                krt = 0

            for u in range(i, i + stride):
                for v in range(j, j + stride):
                    stdout[u, v] = stdev
                    skwout[u, v] = skw
                    krtout[u, v] = krt

    return stdout, skwout, krtout
