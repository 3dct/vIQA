import metrics
from utils import _check_imgs
from skimage.metrics import peak_signal_noise_ratio


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
        score_val = most_apparent_disorder(img_r, img_m)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print('MAD: {}'.format(round(self.score_val, decimals)))
        else:
            print('No score value for MAD. Run score() first.')


def most_apparent_disorder(img_r, img_m):
    pass
