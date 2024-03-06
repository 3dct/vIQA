import re
import warnings

import pytest
import numpy as np

from .context import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        snr = viqa.SNR()
        assert snr.score_val is None, 'Score value should be None'
        assert snr._parameters['data_range'] == 255, 'Data range should be 255'
        assert snr._parameters['normalize'] is False, 'Normalize should be False'
        assert snr._parameters['batch'] is False, 'Batch should be False'
        assert snr._parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        snr = viqa.SNR(data_range=1, normalize=True, batch=True, chromatic=True)
        assert snr.score_val is None, 'Score value should be None'
        assert snr._parameters['data_range'] == 1, 'Data range should be 1'
        assert snr._parameters['normalize'] is True, 'Normalize should be True'
        assert snr._parameters['batch'] is True, 'Batch should be True'
        assert snr._parameters['chromatic'] is True, 'Chromatic should be True'


class TestScoring2D:
    def test_snr_with_modified_image_2d(self, modified_image_2d_255):
        img = modified_image_2d_255
        snr = viqa.SNR()
        score = snr.score(img, signal_center=(300, 300), radius=20)
        assert score != 0, 'SNR of identical images should not be 0'

    def test_snr_with_image_consisting_of_zeros_2d(self):
        img = np.zeros((256, 256))
        snr = viqa.SNR()
        score = snr.score(img, signal_center=(128, 128), radius=8)
        assert score == 0.0, 'SNR of image consisting of only zeros should be 0'

    def test_snr_with_different_images_2d(self, reference_image_2d_255, modified_image_2d_255):
        img_r = reference_image_2d_255
        img_m = modified_image_2d_255
        snr = viqa.SNR()
        score1 = snr.score(img_r, signal_center=(300, 300), radius=20)
        snr = viqa.SNR()
        score2 = snr.score(img_m, signal_center=(300, 300), radius=20)
        assert score1 != score2, 'SNR should be different for different images'

    def test_snr_with_different_regions_2d(self, reference_image_2d_255):
        img = reference_image_2d_255
        snr = viqa.SNR()
        score1 = snr.score(img, signal_center=(300, 300), radius=20)
        snr = viqa.SNR()
        score2 = snr.score(img, signal_center=(600, 600), radius=20)
        assert score1 != score2, 'SNR should be different for different images'


class TestScoring3D:
    def test_snr_with_modified_image_3d(self, modified_image_3d_255):
        img = modified_image_3d_255
        snr = viqa.SNR()
        score = snr.score(img, signal_center=(300, 300, 290), radius=20)
        assert score != 0, 'SNR of identical images should not be 0'

    def test_snr_with_image_consisting_of_zeros_3d(self):
        img = np.zeros((256, 256, 256))
        snr = viqa.SNR()
        score = snr.score(img, signal_center=(128, 128, 128), radius=8)
        assert score == 0.0, 'SNR of image consisting of only zeros should be 0'

    def test_snr_with_different_images_3d(self, reference_image_3d_255, modified_image_3d_255):
        img_r = reference_image_3d_255
        img_m = modified_image_3d_255
        snr = viqa.SNR()
        score1 = snr.score(img_r, signal_center=(300, 300, 290), radius=20)
        snr = viqa.SNR()
        score2 = snr.score(img_m, signal_center=(300, 300, 290), radius=20)
        assert score1 != score2, 'SNR should be different for different images'

    def test_snr_with_different_regions_3d(self, reference_image_3d_255):
        img = reference_image_3d_255
        snr = viqa.SNR()
        score1 = snr.score(img, signal_center=(300, 300, 290), radius=20)
        snr = viqa.SNR()
        score2 = snr.score(img, signal_center=(600, 600, 500), radius=20)
        assert score1 != score2, 'SNR should be different for different images'


class TestPrinting:
    def test_snr_print_score_without_calculating_score(self):
        snr = viqa.SNR()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for SNR. Run score() first.')):
            snr.print_score()

    def test_snr_print_score_with_calculating_score(self, capsys):
        img = np.zeros((256, 256))
        snr = viqa.SNR()
        snr.score(img, signal_center=(128, 128), radius=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'SNR: 0\n', 'Printed score should be inf'

    def test_snr_print_score_with_different_decimals(self, capsys, modified_image_2d_255):
        img = modified_image_2d_255
        snr = viqa.SNR()
        snr.score(img, signal_center=(300, 300), radius=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr.print_score(decimals=2)
            captured = capsys.readouterr()
            assert len(captured.out) == 11, 'Printed score should have 11 characters'
