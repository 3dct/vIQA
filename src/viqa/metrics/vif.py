import torch
from piq import vif_p

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_imgs


class VIFp(FullReferenceMetricsInterface):
    """
    Calculates the visual information fidelity in pixel domain (VIFp) between two images.
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
        Calculates the visual information fidelity in pixel domain (VIFp) between two images.
        :param img_r: Reference image
        :param img_m: Modified image
        :return: Score value
        """
        img_r, img_m = _check_imgs(
            img_r,
            img_m,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            batch=self._parameters["batch"],
        )
        # check if chromatic
        if self._parameters["chromatic"] is False:
            # 3D images
            # img_r_tensor = torch.tensor(img_r).unsqueeze(0).permute(3, 0, 1, 2)
            # img_m_tensor = torch.tensor(img_m).unsqueeze(0).permute(3, 0, 1, 2)
            # 2D images
            img_r_tensor = torch.tensor(img_r).unsqueeze(0).unsqueeze(0)
            img_m_tensor = torch.tensor(img_m).unsqueeze(0).unsqueeze(0)
        else:
            img_r_tensor = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0)
            img_m_tensor = torch.tensor(img_m).permute(2, 0, 1).unsqueeze(0)

        score_val = vif_p(
            img_r_tensor,
            img_m_tensor,
            data_range=self._parameters["data_range"],
            **kwargs,
        )
        # score_val = vifp(img_r, img_m, **self.parameters)
        self.score_val = float(score_val)
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print("VIFp: {}".format(round(self.score_val, decimals)))
        else:
            print("No score value for VIFp. Run score() first.")
