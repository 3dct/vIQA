import metrics
import numpy as np
from utils import _check_imgs, _to_float, correlate_convolve_abs
from kernels import *
import scipy.ndimage as ndi


class GSM(metrics.FullReferenceMetricsInterface):
    """
    Calculates the gradient similarity (GSM) between two images. This implementation is adapted for 3D images.
    Therefore, 12 kernels are used instead of the original 4. Also, the gradient is calculated by
    max{convolve(img, kernel)} instead of max{mean2(abs(x * kernel))}.
    """

    def __init__(self, data_range=255, **kwargs):
        """
        :param data_range: data range of the returned data in data loading
        :param kwargs:
        """
        super().__init__(data_range=data_range)
        self._parameters.update(**kwargs)

    def score(self, img_r, img_m, c=200, p=0.1):
        """
        Calculates the gradient similarity (GSM) between two images.
        :param img_r: Reference image
        :param img_m: Modified image
        :param c: Constant
        :param p: Constant for weighting between luminance and structure similarity. 0 <= p <= 1. Higher p means more
        accentuation of luminance.
        :return: Score value
        """
        img_r, img_m = _check_imgs(img_r, img_m, data_range=self._parameters['data_range'],
                                   normalize=self._parameters['normalize'], batch=self._parameters['batch'])

        kernels = [gsm_kernel_x(), gsm_kernel_y(), gsm_kernel_z(), gsm_kernel_xy1(), gsm_kernel_xy2(), gsm_kernel_yz1(),
                   gsm_kernel_yz2(), gsm_kernel_xz1(), gsm_kernel_xz2()]

        gradients_r = []
        gradients_m = []
        for kernel in kernels:
            gradients_r.append(ndi.correlate(img_r, kernel))
            gradients_m.append(ndi.correlate(img_m, kernel))

        def _mean_of_abs(input_array):
            return np.mean(np.abs(input_array))

        img_r_gradient = sorted(gradients_r, key=_mean_of_abs, reverse=True)[0]
        img_m_gradient = sorted(gradients_m, key=_mean_of_abs, reverse=True)[0]

        img_r_gradient = _to_float(img_r_gradient)
        img_m_gradient = _to_float(img_m_gradient)
        k = c / max(np.max(img_r_gradient), np.max(img_m_gradient))
        r = np.abs(img_r_gradient - img_m_gradient) / max(img_r_gradient, img_m_gradient, key=_mean_of_abs)
        con_struc_sim = ((2 * 1 - r) + k) / (1 + (1 - r)**2 + k)
        lum_sim = 1 - ((img_r - img_m) / self._parameters['data_range'])**2
        weight = p * con_struc_sim
        quality = (1 - weight) * con_struc_sim + weight * lum_sim

        score_val = np.nanmean(quality)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print('GSM: {}'.format(round(self.score_val, decimals)))
        else:
            print('No score value for GSM. Run score() first.')
