import metrics
from utils import _check_imgs, _to_float
import numpy as np
from IQA_pytorch import MAD as MAD_pytorch
import torch


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

        # check if chromatic
        if self._parameters['chromatic'] is False:
            # 3D images
            # img_r_tensor = torch.tensor(img_r).unsqueeze(0).permute(3, 0, 1, 2)
            # img_m_tensor = torch.tensor(img_m).unsqueeze(0).permute(3, 0, 1, 2)
            # 2D images
            img_r_tensor = torch.tensor(img_r).unsqueeze(0).unsqueeze(0)
            img_m_tensor = torch.tensor(img_m).unsqueeze(0).unsqueeze(0)
        else:
            img_r_tensor = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0)
            img_m_tensor = torch.tensor(img_m).permute(2, 0, 1).unsqueeze(0)
        metric = MAD_pytorch()
        score_val = metric(img_r, img_m, as_loss=False)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print('MAD: {}'.format(round(self.score_val, decimals)))
        else:
            print('No score value for MAD. Run score() first.')


def most_apparent_disorder(img_r, img_m, **kwargs):
    # luminance
    luminance_function = kwargs.pop('luminance_function', {'b': 0, 'k': 0.02874, 'gamma': 2.2})

    img_r_lum = _pixel_to_luminance(img_r, **luminance_function)
    img_m_lum = _pixel_to_luminance(img_m, **luminance_function)

    # perceived luminance
    lum_r = np.cbrt(img_r_lum)
    lum_m = np.cbrt(img_m_lum)

    lum_error = _to_float(lum_r) - _to_float(lum_m)

    # contrast sensitivity function

    pass


def _pixel_to_luminance(img, b=0, k=0.02874, gamma=2.2):
    """
    Converts an image to luminance.
    :param img: Input image
    :param b: Background luminance
    :param k: Constant
    :param gamma: Gamma correction factor
    :return: Luminance image
    """
    img_lum = (b + k * img) ** gamma
    return img_lum
