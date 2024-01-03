import metrics
from utils import check_imgs
from skimage.metrics import structural_similarity


class SSIM(metrics.Full_Reference_Metrics_Interface):
    """
    Calculates the structural similarity index (SSIM) between two images.
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
        Calculates the structural similarity index (SSIM) between two images.
        :param img_r: Reference image
        :param img_m: Modified image
        :return: Score value
        """
        img_r, img_m = check_imgs(img_r, img_m, data_range=self._parameters['data_range'],
                                  normalize=self._parameters['normalize'], batch=self._parameters['batch'])
        score_val = structural_similarity(img_r, img_m, data_range=self._parameters['data_range'], **kwargs)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print('SSIM: {}'.format(round(self.score_val, decimals)))
        else:
            print('No score value for SSIM. Run score() first.')
