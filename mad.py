import metrics
from utils import _check_imgs, _to_float, extract_blocks, _ifft, _fft
import math
import numpy as np
import scipy.fft as fft


class MAD(metrics.Full_Reference_Metrics_Interface):
    """
    Calculates the most apparent disorder (MAD) between two images.
    """

    def __init__(self, data_range=255, **kwargs):
        """
        :param data_range: data range of the returned data in data loading
        :param kwargs:
        """
        super().__init__(data_range=data_range)
        self._parameters.update(**kwargs)

    def score(self, img_r, img_m):
        """
        Calculates the most apparent disorder (MAD) between two images.
        :param img_r: Reference image
        :param img_m: Modified image
        :return: Score value
        """
        img_r, img_m = _check_imgs(img_r, img_m, data_range=self._parameters['data_range'],
                                   normalize=self._parameters['normalize'], batch=self._parameters['batch'])
        score_val = most_apparent_disorder_3D(img_r, img_m, account_monitor=False)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print('MAD: {}'.format(round(self.score_val, decimals)))
        else:
            print('No score value for MAD. Run score() first.')


def most_apparent_disorder_3D(img_r, img_m, **kwargs):
    depth = img_r.shape[2]
    scores = []
    for d in range(depth):
        img_r_d = img_r[:, :, d]
        img_m_d = img_m[:, :, d]
        scores.append(most_apparent_disorder(img_r_d, img_m_d, **kwargs))
    return np.mean(np.array(scores))


def most_apparent_disorder(img_r, img_m, **kwargs):
    global M, N
    M, N = img_r.shape

    # Hiqh quality index
    d_detect = _high_quality(img_r, img_m, **kwargs)

    pass


def _high_quality(img_r, img_m, **kwargs):
    account_monitor = kwargs.pop('account_monitor', False)
    if account_monitor:
        assert 'display_function' in kwargs, 'If account_monitor is True, display_function must be given.'
        display_function = kwargs.pop('display_function')
        cycles_per_degree = (display_function['disp_res'] * display_function['view_dis'] * np.tan(np.pi / 180)) / 2
    else:
        cycles_per_degree = 32

    csf = _contrast_sensitivity_function(M, N, cycles_per_degree, lambda_=0.114, f_peak=7.8909)

    # Convert to perceived luminance
    luminance_function = kwargs.pop('luminance_function', {'b': 0, 'k': 0.02874, 'gamma': 2.2})

    lum_r = _pixel_to_luminance(img_r, **luminance_function)
    lum_m = _pixel_to_luminance(img_m, **luminance_function)

    # Fourier transform
    lum_r_fft = _fft(lum_r)
    lum_m_fft = _fft(lum_m)

    I_org = np.real(_ifft(csf * lum_r_fft))
    I_dst = np.real(_ifft(csf * lum_m_fft))

    I_err = I_dst - I_org

    # Contrast masking
    BLOCK_SIZE = kwargs.pop('block_size', 16)
    overlap = kwargs.pop('block_overlap', 0.75)
    STRIDE = BLOCK_SIZE - int(overlap * BLOCK_SIZE)
    I_org_blocks = extract_blocks(I_org, block_size=BLOCK_SIZE, stride=STRIDE)
    I_err_blocks = extract_blocks(I_err, block_size=BLOCK_SIZE, stride=STRIDE)

    # Calculate local statistics
    mu_org_p = np.mean(I_org_blocks, axis=(1, 2))
    std_org_p = np.array(
        [_min_std(block, block_size=int(BLOCK_SIZE / 2), stride=int(BLOCK_SIZE / 2)) for block in I_org_blocks])
    std_err_p = np.std(I_err_blocks, axis=(1, 2))

    mu_org = np.zeros(I_org.shape)
    std_org = np.zeros(I_org.shape)
    std_err = np.zeros(I_err.shape)

    block_n = 0
    for x in range(0, I_org.shape[0] - STRIDE*3, STRIDE):
        for y in range(0, I_org.shape[1] - STRIDE*3, STRIDE):
            mu_org[x:x+STRIDE, y:y+STRIDE] = mu_org_p[block_n]
            std_org[x:x+STRIDE, y:y+STRIDE] = std_org_p[block_n]
            std_err[x:x+STRIDE, y:y+STRIDE] = std_err_p[block_n]
            block_n += 1

    C_org = std_org / mu_org
    C_err = np.zeros(std_err.shape)
    np.divide(std_err, mu_org, out=C_err, where=mu_org > 0.5)

    delta = -5
    cond_1 = np.logical_and(np.log(C_err) > np.log(C_org), np.log(C_org) > delta)
    cond_2 = np.logical_and(np.log(C_err) > delta, delta >= np.log(C_org))

    xi = np.zeros(C_err.shape)
    np.subtract(np.log(C_err), np.log(C_org), out=xi, where=cond_1)
    np.subtract(np.log(C_err), delta, out=xi, where=cond_2)

    # Combination of contrast masking and contrast sensitivity
    D_p = np.sum(I_err**2) / (I_err.shape[0]*I_err.shape[1])
    d_detect = np.sqrt(np.sum((xi*D_p)**2) / len(xi))
    return d_detect


def _pixel_to_luminance(img, k=0.02874, gamma=2.2):
    """
    Converts an image to perceived luminance.
    :param img: Input image
    :param k: Constant
    :param gamma: Gamma correction factor
    :return: Luminance image
    """
    # Create LUT
    LUT = np.arange(0, 256)
    LUT = k * LUT**(gamma/3)
    img_lum = LUT[img]  # apply LUT
    return img_lum


def _contrast_sensitivity_function(m, n, nfreq, **kwargs):
    """
    Calculates the contrast sensitivity function.

    AUTHOR
    ------
    Author: Eric Larson \n
    Department of Electrical and Computer Engineering \n
    Oklahoma State University, 2008 \n
    University Of Washington Seattle, 2009 \n
    Image Coding and Analysis Lab \n

    Translation: Lukas Behammer \n
    Research Center Wels \n
    University of Applied Sciences Upper Austria \n
    CT Research Group \n

    MODIFICATIONS
    -------------
    Original code, 2008, Eric Larson \n
    Translated to Python, 2024, Lukas Behammer
    :param m: Size of image in y direction
    :param n: Size of image in x direction
    :param nfreq: Maximum spatial frequency
    :param kwargs: lambda_ and f_peak
    :return: Contrast sensitivity function
    """
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


def _min_std(image, block_size, stride):
    """Calculates the minimum standard deviation of blocks of a given image."""
    sub_blocks = extract_blocks(image, block_size=block_size, stride=stride)
    return np.min(np.std(sub_blocks, axis=(1, 2)))


def _gabor_convolve(im, scales_num: int, orientations_num: int, min_wavelength=3, wavelength_scaling=3,
                    bandwidth_param=0.55, d_theta_on_sigma=1.5):
    """
    Computes Gabor filter responses. \n
    bandwidth_param vs wavelength_scaling \n
    0.85 <--> 1.3 \n
    0.74 <--> 1.6 (1 octave bandwidth) \n
    0.65 <--> 2.1 \n
    0.55 <--> 3.0 (2 octave bandwidth) \n

    AUTHOR
    ------
    This code was originally written in Matlab by Peter Kovesi and adapted by Eric Larson. \n
    It is available from http://vision.eng.shizuoka.ac.jp/mad (version 2011_10_07). \n
    It was translated to Python by Lukas Behammer. \n

    Author: Peter Kovesi \n
    Department of Computer Science & Software Engineering \n
    The University of Western Australia \n
    pk@cs.uwa.edu.au  https://peterkovesi.com/projects/ \n

    Adaption: Eric Larson \n
    Department of Electrical and Computer Engineering \n
    Oklahoma State University, 2008 \n
    University Of Washington Seattle, 2009 \n
    Image Coding and Analysis lab

    Translation: Lukas Behammer \n
    Research Center Wels \n
    University of Applied Sciences Upper Austria \n
    CT Research Group \n

    MODIFICATIONS
    -------------
    Original code, May 2001, Peter Kovesi \n
    Altered, 2008, Eric Larson \n
    Altered precomputations, 2011, Eric Larson \n
    Translated to Python, 2024, Lukas Behammer

    Literature
    -------
    D. J. Field, "Relations Between the Statistics of Natural Images and the
    Response Properties of Cortical Cells", Journal of The Optical Society of
    America A, Vol 4, No. 12, December 1987. pp 2379-2394

    LICENSE
    -------
    Copyright (c) 2001-2010 Peter Kovesi
    www.peterkovesi.com

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    The Software is provided "as is", without warranty of any kind.
    :param im: Image to be filtered
    :param scales_num: Number of wavelet scales
    :param orientations_num: Number of filter orientations
    :param min_wavelength: Wavelength of smallest scale filter
    :param wavelength_scaling: Scaling factor between successive filters
    :param bandwidth_param: Ratio of standard deviation of the Gaussian describing log Gabor filter's transfer function
    in the frequency domain to the filter's center frequency (0.74 for 1 octave bandwidth, 0.55 for 2 octave bandwidth,
    0.41 for 3 octave bandwidth)
    :param d_theta_on_sigma: Ratio of angular interval between filter orientations and standard deviation of angular
    Gaussian function used to construct filters in the frequency plane
    :return:
    """
    # Precomputing and assigning variables
    scales = np.arange(0, scales_num)
    orientations = np.arange(0, orientations_num)
    rows, cols = im.shape  # image dimensions
    # center of image
    col_c = math.floor(cols/2)
    row_c = math.floor(rows/2)

    # set up filter wavelengths from scales
    wavelengths = [min_wavelength * wavelength_scaling**scale_n for scale_n in range(0, scales_num)]

    # convert image to frequency domain
    im_fft = fft.fftn(im)

    # compute matrices of same site as im with values ranging from -0.5 to 0.5 (-1.0 to 1.0) for horizontal and vertical directions each
    if cols % 2 == 0:
        x_range = np.linspace(-cols/2, (cols-2)/2, cols) / (cols/2)
    else:
        x_range = np.linspace(-cols/2, cols/2, cols) / (cols/2)
    if rows % 2 == 0:
        y_range = np.linspace(-rows/2, (rows-2)/2, rows) / (rows/2)
    else:
        y_range = np.linspace(-rows/2, rows/2, rows) / (rows/2)
    x, y = np.meshgrid(x_range, y_range)

    # filters have radial component (frequency band) and an angular component (orientation), those are multiplied to get the final filter

    # compute radial distance from center of matrix
    radius = np.sqrt(x**2 + y**2)
    radius[radius == 0] = 1  # avoid logarithm of zero

    # compute polar angle and its sine and cosine
    theta = np.arctan2(-y, x)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # compute standard deviation of angular Gaussian function
    theta_sigma = np.pi/orientations_num / d_theta_on_sigma

    # compute radial component
    radial_components = []
    for scale_n, scale in enumerate(scales):  # for each scale
        center_freq = 1.0/wavelengths[scale_n]  # center frequency of filter
        normalised_center_freq = center_freq/0.5
        # log Gabor response for each frequency band (scale)
        log_gabor = np.exp((np.log(radius)-np.log(normalised_center_freq))**2 / -(2 * np.log(bandwidth_param)**2))
        log_gabor[row_c, col_c] = 0
        radial_components.append(log_gabor)

    # angular component and final filtering
    res = np.empty((scales_num, orientations_num), dtype=object)  # precompute result array
    for orientation_n, orientation in enumerate(orientations):  # for each orientation
        # compute angular component
        # Pre-compute filter data specific to this orientation
        # For each point in the filter matrix calculate the angular distance from the specified filter orientation.  To overcome the angular wrap-around problem sine difference and cosine difference values are first computed and then the atan2 function is used to determine angular distance.
        angle = orientation_n*np.pi / orientations_num  # filter angle
        diff_sin = sin_theta*np.cos(angle) - cos_theta*np.sin(angle)  # difference of sin
        diff_cos = cos_theta*np.cos(angle) + sin_theta*np.sin(angle)  # difference of cos
        angular_distance = abs(np.arctan2(diff_sin, diff_cos))  # absolute angular distance
        spread = np.exp((-angular_distance**2)/(2 * theta_sigma**2))  # angular filter component

        # filtering
        for scale_n, scale in enumerate(scales):  # for each scale
            # compute final filter
            filter_ = fft.fftshift(radial_components[scale_n]*spread)
            filter_[0, 0] = 0

            # apply filter
            res[scale_n, orientation_n] = fft.ifftn(im_fft*filter_)

    return res
