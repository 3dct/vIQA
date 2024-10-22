import re
import warnings

import pytest
import numpy as np

import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        ssim = viqa.SSIM()
        assert ssim.score_val is None, 'Score value should be None'
        assert ssim.parameters['data_range'] == 255, 'Data range should be 255'
        assert ssim.parameters['normalize'] is False, 'Normalize should be False'
        assert ssim.parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        ssim = viqa.SSIM(data_range=1, normalize=True, chromatic=True)
        assert ssim.score_val is None, 'Score value should be None'
        assert ssim.parameters['data_range'] == 1, 'Data range should be 1'
        assert ssim.parameters['normalize'] is True, 'Normalize should be True'
        assert ssim.parameters['chromatic'] is True, 'Chromatic should be True'

    def test_init_without_data_range(self):
        with pytest.raises(ValueError, match=re.escape('Parameter data_range must be set.')):
            viqa.SSIM(data_range=None, normalize=True, chromatic=True)


class TestScoring2D:
    def test_ssim_with_identical_images_2d(self):
        img_r = np.random.rand(256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score = ssim.score(img_r, img_r)
        assert score == 1.0, 'SSIM of identical images should be 1.0'

    def test_ssim_with_completely_different_images_no_gaussian_2d(self):
        img_r = np.zeros((256, 256))
        img_m = np.ones((256, 256))
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score = ssim.score(img_r, img_m, gaussian_weights=False)
        assert pytest.approx(score, abs=1e-4) == 0.0, 'SSIM of completely different images should be 0'

    def test_ssim_with_random_images_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score = ssim.score(img_r, img_m)
        assert 0 <= score <= 1.0, 'SSIM should be between 0 and 1'

    def test_ssim_with_different_data_ranges_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score1 = ssim.score(img_r, img_m)
        ssim = viqa.SSIM(data_range=255, normalize=True)
        score2 = ssim.score(img_r, img_m)
        assert score1 != score2, 'SSIM should be different for different data ranges'

    def test_ssim_with_random_images_float_alpha_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        with pytest.warns(RuntimeWarning, match=re.escape('alpha is not an integer. Cast to int.')):
            ssim.score(img_r, img_m, alpha=8.5)

    def test_ssim_with_random_images_float_beta_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        with pytest.warns(RuntimeWarning, match=re.escape('beta is not an integer. Cast to int.')):
            ssim.score(img_r, img_m, beta=8.5)

    def test_ssim_with_random_images_float_gamma_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        with pytest.warns(RuntimeWarning, match=re.escape('gamma is not an integer. Cast to int.')):
            ssim.score(img_r, img_m, gamma=8.5)


class TestScoring3D:
    def test_ssim_with_identical_images_3d(self):
        img_r = np.random.rand(256, 256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score = ssim.score(img_r, img_r)
        assert score == 1.0, 'SSIM of identical images should be 1.0'

    def test_ssim_with_completely_different_images_3d(self):
        img_r = np.zeros((256, 256, 256))
        img_m = np.ones((256, 256, 256))
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score = ssim.score(img_r, img_m)
        assert pytest.approx(score, abs=1e-4) == 0.0, 'SSIM of completely different images should be 0'

    def test_ssim_with_random_images_3d(self):
        img_r = np.random.rand(256, 256, 256)
        img_m = np.random.rand(256, 256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score = ssim.score(img_r, img_m)
        assert 0 <= score <= 1.0, 'SSIM should be between 0 and 1'

    def test_ssim_with_different_data_ranges_3d(self):
        img_r = np.random.rand(256, 256, 256)
        img_m = np.random.rand(256, 256, 256)
        ssim = viqa.SSIM(data_range=1, normalize=False)
        score1 = ssim.score(img_r, img_m)
        ssim = viqa.SSIM(data_range=255, normalize=True)
        score2 = ssim.score(img_r, img_m)
        assert score1 != score2, 'SSIM should be different for different data ranges'


class TestPrinting:
    def test_ssim_print_score_without_calculating_score(self):
        ssim = viqa.SSIM()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for SSIM. Run score() first.')):
            ssim.print_score()

    def test_ssim_print_score_with_calculating_score(self, capsys):
        img_r = np.zeros((256, 256))
        img_m = np.zeros((256, 256))
        ssim = viqa.SSIM()
        ssim.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'SSIM: 1.0\n', 'Printed score should be inf'

    def test_ssim_print_score_with_different_decimals(self, capsys):
        img_r = np.zeros((256, 256))
        img_m = np.ones((256, 256))
        ssim = viqa.SSIM()
        ssim.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssim.print_score(decimals=2)
            captured = capsys.readouterr()
            assert len(captured.out) == 11, 'Printed score should have 11 characters'


def test_ssim_invalid_k1():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('K1 must be positive')):
        viqa.SSIM().score(img_r, img_m, K1=-1)


def test_ssim_invalid_k2():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('K2 must be positive')):
        viqa.SSIM().score(img_r, img_m, K2=-1)


def test_ssim_invalid_sigma():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('sigma must be positive')):
        viqa.SSIM().score(img_r, img_m, sigma=-1)


def test_ssim_even_win_size():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('win_size must be odd.')):
        viqa.SSIM().score(img_r, img_m, win_size=2)
