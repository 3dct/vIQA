import re
import warnings

import pytest
import numpy as np

import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        psnr = viqa.PSNR()
        assert psnr.score_val is None, 'Score value should be None'
        assert psnr.parameters['data_range'] == 255, 'Data range should be 255'
        assert psnr.parameters['normalize'] is False, 'Normalize should be False'
        assert psnr.parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        psnr = viqa.PSNR(data_range=1, normalize=True, chromatic=True)
        assert psnr.score_val is None, 'Score value should be None'
        assert psnr.parameters['data_range'] == 1, 'Data range should be 1'
        assert psnr.parameters['normalize'] is True, 'Normalize should be True'
        assert psnr.parameters['chromatic'] is True, 'Chromatic should be True'

    def test_init_without_data_range(self):
        with pytest.raises(ValueError, match=re.escape('Parameter data_range must be set.')):
            viqa.PSNR(data_range=None, normalize=True, chromatic=True)


class TestScoring2D:
    def test_psnr_with_identical_images_2d(self):
        img_r = np.random.rand(256, 256)
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score = psnr.score(img_r, img_r)
        assert score == float('inf'), 'PSNR of identical images should be infinity'

    def test_psnr_with_completely_different_images_2d(self):
        img_r = np.zeros((256, 256))
        img_m = np.ones((256, 256))
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score = psnr.score(img_r, img_m)
        assert score == 0.0, 'PSNR of completely different images should be 0'

    def test_psnr_with_random_images_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score = psnr.score(img_r, img_m)
        assert 0 <= score <= float('inf'), 'PSNR should be between 0 and infinity'

    def test_psnr_with_different_data_ranges_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score1 = psnr.score(img_r, img_m)
        psnr = viqa.PSNR(data_range=255, normalize=True)
        score2 = psnr.score(img_r, img_m)
        assert score1 != score2, 'PSNR should be different for different data ranges'


class TestScoring3D:
    def test_psnr_with_identical_images_3d(self):
        img_r = np.random.rand(256, 256, 256)
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score = psnr.score(img_r, img_r)
        assert score == float('inf'), 'PSNR of identical images should be infinity'

    def test_psnr_with_completely_different_images_3d(self):
        img_r = np.zeros((256, 256, 256))
        img_m = np.ones((256, 256, 256))
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score = psnr.score(img_r, img_m)
        assert score == 0.0, 'PSNR of completely different images should be 0'

    def test_psnr_with_random_images_3d(self):
        img_r = np.random.rand(256, 256, 256)
        img_m = np.random.rand(256, 256, 256)
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score = psnr.score(img_r, img_m)
        assert 0 <= score <= float('inf'), 'PSNR should be between 0 and infinity'

    def test_psnr_with_different_data_ranges_3d(self):
        img_r = np.random.rand(256, 256, 256)
        img_m = np.random.rand(256, 256, 256)
        psnr = viqa.PSNR(data_range=1, normalize=False)
        score1 = psnr.score(img_r, img_m)
        psnr = viqa.PSNR(data_range=255, normalize=True)
        score2 = psnr.score(img_r, img_m)
        assert score1 != score2, 'PSNR should be different for different data ranges'


class TestPrinting:
    def test_psnr_print_score_without_calculating_score(self):
        psnr = viqa.PSNR()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for PSNR. Run score() first.')):
            psnr.print_score()

    def test_psnr_print_score_with_calculating_score(self, capsys):
        img_r = np.zeros((256, 256))
        img_m = np.zeros((256, 256))
        psnr = viqa.PSNR()
        psnr.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psnr.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'PSNR: inf\n', 'Printed score should be inf'

    def test_psnr_print_score_with_different_decimals(self, capsys):
        img_r = np.zeros((256, 256))
        img_m = np.ones((256, 256))
        psnr = viqa.PSNR()
        psnr.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            psnr.print_score(decimals=2)
            captured = capsys.readouterr()
            assert len(captured.out) == 12, 'Printed score should have 11 characters'
