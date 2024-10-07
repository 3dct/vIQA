import re
import warnings

import pytest
import numpy as np

import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        rmse = viqa.RMSE()
        assert rmse.score_val is None, 'Score value should be None'
        assert rmse.parameters['data_range'] is None, 'Data range should be None'
        assert rmse.parameters['normalize'] is False, 'Normalize should be False'
        assert rmse.parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        rmse = viqa.RMSE(data_range=1, normalize=True, chromatic=True)
        assert rmse.score_val is None, 'Score value should be None'
        assert rmse.parameters['data_range'] == 1, 'Data range should be 1'
        assert rmse.parameters['normalize'] is True, 'Normalize should be True'
        assert rmse.parameters['chromatic'] is True, 'Chromatic should be True'


class TestScoring2D:
    def test_rmse_with_identical_images_2d(self):
        img_r = np.zeros((256, 256))
        img_m = np.zeros((256, 256))
        rmse = viqa.RMSE()
        assert rmse.score(img_r, img_m) == 0.0, 'RMSE of identical images should be 0'

    def test_rmse_with_completely_different_images_2d(self):
        img_r = np.zeros((256, 256))
        img_m = np.ones((256, 256))
        rmse = viqa.RMSE()
        assert rmse.score(img_r, img_m) == 1.0, 'RMSE of completely different images should be 1'

    def test_rmse_with_random_images_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        rmse = viqa.RMSE()
        assert 0 <= rmse.score(img_r, img_m) <= 1.0, 'RMSE should be between 0 and 1'

    def test_psnr_with_different_data_ranges_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        rmse = viqa.RMSE()
        score1 = rmse.score(img_r, img_m)
        rmse = viqa.RMSE(data_range=255, normalize=True)
        score2 = rmse.score(img_r, img_m)
        assert score1 != score2, 'RMSE should be different for different data ranges'


class TestScoring3D:
    def test_rmse_with_identical_images_3d(self):
        img_r = np.zeros((256, 256, 256))
        img_m = np.zeros((256, 256, 256))
        rmse = viqa.RMSE()
        assert rmse.score(img_r, img_m) == 0.0, 'RMSE of identical images should be 0'

    def test_rmse_with_completely_different_images_3d(self):
        img_r = np.zeros((256, 256, 256))
        img_m = np.ones((256, 256, 256))
        rmse = viqa.RMSE()
        assert rmse.score(img_r, img_m) == 1.0, 'RMSE of completely different images should be 1'

    def test_rmse_with_random_images_3d(self):
        img_r = np.random.rand(256, 256, 256)
        img_m = np.random.rand(256, 256, 256)
        rmse = viqa.RMSE()
        assert 0 <= rmse.score(img_r, img_m) <= 1.0, 'RMSE should be between 0 and 1'

    def test_psnr_with_different_data_ranges_3d(self):
        img_r = np.random.rand(256, 256, 256)
        img_m = np.random.rand(256, 256, 256)
        rmse = viqa.RMSE()
        score1 = rmse.score(img_r, img_m)
        rmse = viqa.RMSE(data_range=255, normalize=True)
        score2 = rmse.score(img_r, img_m)
        assert score1 != score2, 'RMSE should be different for different data ranges'


class TestPrinting:
    def test_print_score_without_calculating_score_first(self):
        rmse = viqa.RMSE()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for RMSE. Run score() first.')):
            rmse.print_score()

    def test_print_score_with_calculating_score_first(self, capsys):
        img_r = np.zeros((256, 256))
        img_m = np.zeros((256, 256))
        rmse = viqa.RMSE()
        rmse.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rmse.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'RMSE: 0.0\n', 'Printed score should be 0.0'

    def test_print_score_with_different_decimals(self, capsys):
        img_r = np.zeros((256, 256))
        img_m = np.random.rand(256, 256)
        rmse = viqa.RMSE()
        rmse.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rmse.print_score(decimals=2)
            captured = capsys.readouterr()
            assert len(captured.out) == 11, 'Printed score should have 11 characters'
