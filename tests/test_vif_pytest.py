import re
import warnings

import numpy as np
import pytest
import torch
from piq import vif_p
from scipy.ndimage import uniform_filter

import viqa


def _smooth_image(rng, shape):
    """Random image with structure (values between 0.25 and 0.75)."""
    return uniform_filter(rng.random(shape), size=5) * 0.5 + 0.25


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def blurred_pair_2d():
    rng = np.random.default_rng(0)
    img_r = uniform_filter(rng.random((128, 128)), size=5)
    img_m = uniform_filter(img_r, size=9)
    return img_r, img_m


class TestInit:
    def test_init_with_default_parameters(self):
        vif = viqa.VIFp()
        assert vif.score_val is None, 'Score value should be None'
        assert vif.parameters['data_range'] == 255, 'Data range should be 255'
        assert vif.parameters['normalize'] is False, 'Normalize should be False'
        assert vif.parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        vif = viqa.VIFp(data_range=1, normalize=True, chromatic=True)
        assert vif.score_val is None, 'Score value should be None'
        assert vif.parameters['data_range'] == 1, 'Data range should be 1'
        assert vif.parameters['normalize'] is True, 'Normalize should be True'
        assert vif.parameters['chromatic'] is True, 'Chromatic should be True'

    def test_init_without_data_range(self):
        with pytest.raises(ValueError, match=re.escape('Parameter data_range must be set.')):
            viqa.VIFp(data_range=None, normalize=True, chromatic=True)


class TestScoring2D:
    def test_vif_with_identical_images_2d(self, rng):
        img_r = _smooth_image(rng, (256, 256))
        vif = viqa.VIFp(data_range=1, normalize=False)
        assert pytest.approx(vif(img_r, img_r), abs=1e-6) == 1.0, 'VIF of identical images should be 1.0'

    def test_vif_with_blurred_image_2d(self, rng):
        img_r = _smooth_image(rng, (256, 256))
        img_m = uniform_filter(img_r, size=9)
        vif = viqa.VIFp(data_range=1, normalize=False)
        score = vif(img_m, img_r)
        assert 0.0 <= score < 1.0, 'VIF of a blurred image should be between 0 and 1'

    def test_vif_with_noisy_image_2d(self, rng):
        img_r = _smooth_image(rng, (256, 256))
        img_m = np.clip(img_r + rng.normal(0, 0.05, img_r.shape), 0, 1)
        vif = viqa.VIFp(data_range=1, normalize=False)
        score = vif(img_m, img_r)
        assert 0.0 <= score < 1.0, 'VIF of a noisy image should be between 0 and 1'

    def test_vif_with_contrast_enhanced_image_2d(self, rng):
        img_r = _smooth_image(rng, (256, 256))
        img_m = (img_r - img_r.mean()) * 1.5 + img_r.mean()
        vif = viqa.VIFp(data_range=1, normalize=False)
        assert vif(img_m, img_r) > 1.0, 'VIF of a contrast enhanced image should be greater than 1'

    def test_vif_with_random_images_2d(self, rng):
        img_r = rng.random((256, 256))
        img_m = rng.random((256, 256))
        vif = viqa.VIFp(data_range=1, normalize=False)
        score = vif(img_m, img_r)
        assert 0.0 <= score < 0.1, 'VIF of unrelated images should be close to 0'

    def test_vif_with_different_data_ranges_2d(self, rng):
        img_r = rng.random((256, 256))
        img_m = rng.random((256, 256))
        vif = viqa.VIFp(data_range=1, normalize=False)
        score1 = vif(img_m, img_r)
        vif = viqa.VIFp(data_range=255, normalize=True)
        score2 = vif(img_m, img_r)
        assert score1 != score2, 'VIF should be different for different data ranges'


class TestScoring3D:
    def test_vif_with_identical_images_3d_dim0_slice32(self, rng):
        img_r = _smooth_image(rng, (64, 64, 64))
        vif = viqa.VIFp(data_range=1, normalize=False)
        score = vif(img_r, img_r, dim=0, im_slice=32)
        assert pytest.approx(score, abs=1e-6) == 1.0, 'VIF of identical images should be 1.0'

    def test_vif_with_blurred_image_3d_dim2_slice32(self, rng):
        img_r = _smooth_image(rng, (64, 64, 64))
        img_m = uniform_filter(img_r, size=9)
        vif = viqa.VIFp(data_range=1, normalize=False)
        score = vif(img_m, img_r, dim=2, im_slice=32)
        assert 0.0 <= score < 1.0, 'VIF of a blurred image should be between 0 and 1'

    def test_vif_with_blurred_image_3d_full_volume(self, rng):
        img_r = _smooth_image(rng, (64, 64, 64))
        img_m = uniform_filter(img_r, size=9)
        vif = viqa.VIFp(data_range=1, normalize=False)
        with pytest.warns(RuntimeWarning, match=re.escape('im_slice is not given. Calculating VIFp for full volume.')):
            score = vif(img_m, img_r, dim=0)
        assert 0.0 <= score < 1.0, 'VIF of a blurred image should be between 0 and 1'

    def test_vif_with_different_data_ranges_3d_dim1_slice32(self, rng):
        img_r = rng.random((64, 64, 64))
        img_m = rng.random((64, 64, 64))
        vif = viqa.VIFp(data_range=1, normalize=False)
        score1 = vif(img_m, img_r, dim=1, im_slice=32)
        vif = viqa.VIFp(data_range=255, normalize=True)
        score2 = vif(img_m, img_r, dim=1, im_slice=32)
        assert score1 != score2, 'VIF should be different for different data ranges'

    def test_vif_3d_without_dim(self, rng):
        img_r = rng.random((64, 64, 64))
        vif = viqa.VIFp(data_range=1, normalize=False)
        with pytest.raises(ValueError, match=re.escape('If images are 3D, dim and im_slice (optional) must be given.')):
            vif(img_r, img_r)

    def test_vif_3d_invalid_dim(self, rng):
        img_r = rng.random((64, 64, 64))
        vif = viqa.VIFp(data_range=1, normalize=False)
        with pytest.raises(ValueError, match=re.escape('Invalid dim value. Must be integer of 0, 1 or 2.')):
            vif(img_r, img_r, dim=3, im_slice=32)


class TestArgumentOrder:
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


class TestPrinting:
    def test_vif_print_score_without_calculating_score(self):
        vif = viqa.VIFp()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for VIFp. Calculate the score first.')):
            vif.print_score()

    def test_vif_print_score_with_different_decimals(self, capsys, blurred_pair_2d):
        img_r, img_m = blurred_pair_2d
        vif = viqa.VIFp(data_range=1, normalize=False)
        vif(img_m, img_r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vif.print_score(decimals=3)
            captured = capsys.readouterr()
            assert re.fullmatch(r'VIFp: -?\d+\.\d{1,3}\n', captured.out), 'Printed score should have at most 3 decimals'
