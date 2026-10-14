import numpy as np
import pytest
import torch
from piq import vif_p
from scipy.ndimage import uniform_filter

import viqa


@pytest.fixture(scope="module")
def blurred_pair_2d():
    rng = np.random.default_rng(0)
    img_r = uniform_filter(rng.random((128, 128)), size=5)
    img_m = uniform_filter(img_r, size=9)
    return img_r, img_m


class TestArgumentOrder:
    def test_vif_with_identical_images_2d(self, blurred_pair_2d):
        img_r, _ = blurred_pair_2d
        vif = viqa.VIFp(data_range=1, normalize=False)
        assert pytest.approx(vif(img_r, img_r), abs=1e-4) == 1.0, 'VIFp of identical images should be 1.0'

    def test_vif_matches_piq_prediction_target_order(self, blurred_pair_2d):
        img_r, img_m = blurred_pair_2d
        vif = viqa.VIFp(data_range=1, normalize=False)
        expected = vif_p(
            torch.tensor(img_m)[None, None], torch.tensor(img_r)[None, None], data_range=1
        )
        assert pytest.approx(vif(img_m, img_r), abs=1e-6) == float(expected), \
            'VIFp should pass the modified image as x and the reference image as y to piq'

    def test_vif_is_not_symmetric(self, blurred_pair_2d):
        img_r, img_m = blurred_pair_2d
        vif = viqa.VIFp(data_range=1, normalize=False)
        score = vif(img_m, img_r)
        score_swapped = vif(img_r, img_m)
        assert score < 1.0, 'VIFp of a blurred image should be smaller than 1'
        assert score != pytest.approx(score_swapped), 'VIFp should depend on the argument order'
