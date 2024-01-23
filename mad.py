import metrics
from utils import _check_imgs, _to_float, extract_blocks, _ifft, _fft, _gabor_convolve
import numpy as np
from scipy.ndimage import convolve
from scipy.stats import skew, kurtosis

M = 0
N = 0
BLOCK_SIZE = 0
STRIDE = 0


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

    def score(self, img_r, img_m, **kwargs):
        """
        Calculates the most apparent disorder (MAD) between two images.
        :param img_r: Reference image
        :param img_m: Modified image
        :param kwargs: account_monitor, display_function, luminance_function, block_size, block_overlap
        :return: Score value
        """
        img_r, img_m = _check_imgs(img_r, img_m, data_range=self._parameters['data_range'],
                                   normalize=self._parameters['normalize'], batch=self._parameters['batch'])
        global M, N, BLOCK_SIZE, STRIDE
        M, N = img_r.shape
        BLOCK_SIZE = kwargs.pop('block_size', 16)
        overlap = kwargs.pop('block_overlap', 0.75)
        STRIDE = BLOCK_SIZE - int(overlap * BLOCK_SIZE)
        account_monitor = kwargs.pop('account_monitor')
        score_val = most_apparent_disorder_3d(img_r, img_m, account_monitor=account_monitor, **kwargs)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print('MAD: {}'.format(round(self.score_val, decimals)))
        else:
            print('No score value for MAD. Run score() first.')


def most_apparent_disorder_3d(img_r, img_m, **kwargs):
    depth = img_r.shape[2]
    scores = []
    for d in range(depth):
        img_r_d = img_r[:, :, d]
        img_m_d = img_m[:, :, d]
        scores.append(most_apparent_disorder(img_r_d, img_m_d, **kwargs))
    return np.mean(np.array(scores))


def most_apparent_disorder(img_r, img_m, **kwargs):
    beta_1 = kwargs.pop('beta1', 0.467)
    beta_2 = kwargs.pop('beta2', 0.130)
    if kwargs['thresh1'] and kwargs['thresh2']:
        thresh1 = kwargs.pop('thresh1', 2.55)
        thresh2 = kwargs.pop('thresh2', 3.35)
        beta_1 = np.exp(-thresh1 / thresh2)
        beta_2 = 1 / (np.log(10) * thresh2)

    # Hiqh quality index
    d_detect = _high_quality(img_r, img_m, **kwargs)
    # Low quality index
    d_appear = _low_quality(img_r, img_m, **kwargs)

    alpha = 1 / (1 + beta_1 * d_detect ** beta_2)
    mad_index = d_detect ** alpha * d_appear ** (1 - alpha)
    return mad_index


def _high_quality(img_r, img_m, **kwargs):
    account_monitor = kwargs.pop('account_monitor', False)
    if account_monitor:
        assert 'display_function' in kwargs, 'If account_monitor is True, display_function must be given.'
        display_function = kwargs.pop('display_function')
        cycles_per_degree = (display_function['disp_res'] * display_function['view_dis'] * np.tan(np.pi / 180)) / 2
    else:
        cycles_per_degree = 32

    csf = _contrast_sensitivity_function(M, N, cycles_per_degree, lambda_=0.114, f_peak=7.8909)

    # Convert to perceived lightness
    luminance_function = kwargs.pop('luminance_function', {'b': 0, 'k': 0.02874, 'gamma': 2.2})

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

    # in matlab: additional normalization parameter: ms_scale = 1
    # --> (... - ...) / ms_scale
    # not yet implemented
    msk = np.zeros(c_err.shape)
    _ = np.subtract(ci_err, tmp, out=msk, where=cond_1)
    _ = np.subtract(ci_err, cd_thrsh, out=msk, where=cond_2)

    win = np.ones((BLOCK_SIZE, BLOCK_SIZE)) / BLOCK_SIZE ** 2
    lmse = convolve((_to_float(img_r) - _to_float(img_m)) ** 2, win, mode='reflect')

    mp = msk * lmse
    mp2 = mp[BLOCK_SIZE:-BLOCK_SIZE, BLOCK_SIZE:-BLOCK_SIZE]

    d_detect = np.linalg.norm(mp2) / np.sqrt(np.prod(mp2.shape)) * 200
    return d_detect


def _low_quality(img_r, img_m, **kwargs):
    orientations_num = kwargs.pop('orientations_num', 4)
    scales_num = kwargs.pop('scales_num', 5)
    weights = kwargs.pop('weights', [0.5, 0.75, 1, 5, 6])
    weights /= np.sum(weights)

    gabor_org = _gabor_convolve(img_m, scales_num=scales_num, orientations_num=orientations_num, min_wavelength=3,
                                wavelength_scaling=3, bandwidth_param=0.55, d_theta_on_sigma=1.5)
    gabor_dst = _gabor_convolve(img_r, scales_num=scales_num, orientations_num=orientations_num, min_wavelength=3,
                                wavelength_scaling=3, bandwidth_param=0.55, d_theta_on_sigma=1.5)

    stats = np.zeros((M, N))
    for scale_n in range(scales_num):
        for orientation_n in range(orientations_num):
            std_ref_p, skw_ref_p, krt_ref_p = _get_statistics(np.abs(gabor_org[scale_n, orientation_n]),
                                                              block_size=BLOCK_SIZE, stride=STRIDE)
            std_dst_p, skw_dst_p, krt_dst_p = _get_statistics(np.abs(gabor_dst[scale_n, orientation_n]),
                                                              block_size=BLOCK_SIZE, stride=STRIDE)

            std_ref = np.zeros((M, N))
            std_dst = np.zeros((M, N))
            skw_ref = np.zeros((M, N))
            skw_dst = np.zeros((M, N))
            krt_ref = np.zeros((M, N))
            krt_dst = np.zeros((M, N))

            # --> as function?
            block_n = 0
            for x in range(0, M - STRIDE * 3, STRIDE):
                for y in range(0, N - STRIDE * 3, STRIDE):
                    std_ref[x:x + STRIDE, y:y + STRIDE] = std_ref_p[block_n]
                    std_dst[x:x + STRIDE, y:y + STRIDE] = std_dst_p[block_n]
                    skw_ref[x:x + STRIDE, y:y + STRIDE] = skw_ref_p[block_n]
                    skw_dst[x:x + STRIDE, y:y + STRIDE] = skw_dst_p[block_n]
                    krt_ref[x:x + STRIDE, y:y + STRIDE] = krt_ref_p[block_n]
                    krt_dst[x:x + STRIDE, y:y + STRIDE] = krt_dst_p[block_n]
                    block_n += 1

            stats += weights[scale_n] * (
                        np.abs(std_ref - std_dst) + 2 * np.abs(skw_ref - skw_dst) + np.abs(krt_ref - krt_dst))

    mp2 = stats[BLOCK_SIZE:-BLOCK_SIZE, BLOCK_SIZE:-BLOCK_SIZE]
    d_appear = np.linalg.norm(mp2) / np.sqrt(np.prod(mp2.shape))
    return d_appear


def _pixel_to_lightness(img, k=0.02874, gamma=2.2):
    """
    Converts an image to perceived lightness.
    :param img: Input image
    :param k: Constant
    :param gamma: Gamma correction factor
    :return: Luminance image
    """
    if issubclass(img.dtype.type, np.integer):
        # Create LUT
        lut = np.arange(0, 256)
        lut = k * lut**(gamma/3)
        img_lum = lut[img]  # apply LUT
    else:
        img_lum = k * img**(gamma/3)
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

    PARAMETERS
    ----------
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


def _min_std(image):
    """Calculates the minimum standard deviation of blocks of a given image."""
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
    # maybe implement in C
    sub_blocks = extract_blocks(image, block_size=block_size, stride=stride)
    std = np.std(sub_blocks, axis=(1, 2))
    skw = []
    krt = []
    for block in sub_blocks:
        skw.append(skew(np.abs(block.flatten())))
        krt.append(kurtosis(np.abs(block.flatten())) + 3)
    return std, skw, krt
