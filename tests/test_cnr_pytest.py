import re
import warnings

import pytest
import numpy as np

from .context import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        cnr = viqa.CNR()
        assert cnr.score_val is None, 'Score value should be None'
        assert cnr._parameters['data_range'] == 255, 'Data range should be 255'
        assert cnr._parameters['normalize'] is False, 'Normalize should be False'
        assert cnr._parameters['batch'] is False, 'Batch should be False'
        assert cnr._parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        cnr = viqa.CNR(data_range=1, normalize=True, batch=True, chromatic=True)
        assert cnr.score_val is None, 'Score value should be None'
        assert cnr._parameters['data_range'] == 1, 'Data range should be 1'
        assert cnr._parameters['normalize'] is True, 'Normalize should be True'
        assert cnr._parameters['batch'] is True, 'Batch should be True'
        assert cnr._parameters['chromatic'] is True, 'Chromatic should be True'


class TestScoring2D:
    def test_cnr_with_modified_image_2d(self, modified_image_2d_255):
        img = modified_image_2d_255
        cnr = viqa.CNR()
        score = cnr.score(img, background_center=(16, 16), signal_center=(128, 128), radius=8)
        assert score != 0, 'CNR of identical images should not be 0'

    def test_cnr_with_image_consisting_of_zeros_2d(self):
        img = np.zeros((256, 256))
        cnr = viqa.CNR()
        score = cnr.score(img, background_center=(16, 16), signal_center=(128, 128), radius=8)
        assert score == 0.0, 'CNR of image consisting of only zeros should be 0'

    def test_cnr_with_different_images_2d(self, reference_image_2d_255, modified_image_2d_255):
        img_r = reference_image_2d_255
        img_m = modified_image_2d_255
        cnr = viqa.CNR()
        score1 = cnr.score(img_r, background_center=(16, 16), signal_center=(400, 400), radius=8)
        cnr = viqa.CNR()
        score2 = cnr.score(img_m, background_center=(16, 16), signal_center=(400, 400), radius=8)
        assert score1 != score2, 'CNR should be different for different images'

    def test_cnr_with_different_regions_2d(self, reference_image_2d_255):
        img = reference_image_2d_255
        cnr = viqa.CNR()
        score1 = cnr.score(img, background_center=(16, 16), signal_center=(400, 400), radius=8)
        cnr = viqa.CNR()
        score2 = cnr.score(img, background_center=(300, 300), signal_center=(400, 400), radius=8)
        assert score1 != score2, 'CNR should be different for different images'


class TestScoring3D:
    def test_cnr_with_modified_image_3d(self, modified_image_3d_255):
        img = modified_image_3d_255
        cnr = viqa.CNR()
        score = cnr.score(img, background_center=(16, 16, 16), signal_center=(500, 500, 400), radius=8)
        assert score != 0, 'CNR of identical images should not be 0'

    def test_cnr_with_image_consisting_of_zeros_3d(self):
        img = np.zeros((256, 256, 256))
        cnr = viqa.CNR()
        score = cnr.score(img, background_center=(16, 16, 16), signal_center=(128, 128, 128), radius=8)
        assert score == 0.0, 'CNR of image consisting of only zeros should be 0'

    def test_cnr_with_different_images_3d(self, reference_image_3d_255, modified_image_3d_255):
        img_r = reference_image_3d_255
        img_m = modified_image_3d_255
        cnr = viqa.CNR()
        score1 = cnr.score(img_r, background_center=(16, 16, 16), signal_center=(500, 500, 400), radius=8)
        cnr = viqa.CNR()
        score2 = cnr.score(img_m, background_center=(16, 16, 16), signal_center=(500, 500, 400), radius=8)
        assert score1 != score2, 'CNR should be different for different images'

    def test_cnr_with_different_regions_3d(self, reference_image_3d_255):
        img = reference_image_3d_255
        cnr = viqa.CNR()
        score1 = cnr.score(img, background_center=(16, 16, 16), signal_center=(500, 500, 400), radius=8)
        cnr = viqa.CNR()
        score2 = cnr.score(img, background_center=(300, 300, 300), signal_center=(500, 500, 400), radius=8)
        assert score1 != score2, 'CNR should be different for different images'


class TestPrinting:
    def test_cnr_print_score_without_calculating_score(self):
        cnr = viqa.CNR()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for CNR. Run score() first.')):
            cnr.print_score()

    def test_cnr_print_score_with_calculating_score(self, capsys):
        img = np.zeros((256, 256))
        cnr = viqa.CNR()
        cnr.score(img, background_center=(16, 16), signal_center=(128, 128), radius=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cnr.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'CNR: 0\n', 'Printed score should be 0'

    def test_cnr_print_score_with_different_decimals(self, capsys, modified_image_2d_255):
        img = modified_image_2d_255
        cnr = viqa.CNR()
        cnr.score(img, background_center=(16, 16), signal_center=(400, 400), radius=8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cnr.print_score(decimals=2)
            captured = capsys.readouterr()
            assert len(captured.out) == 12, 'Printed score should have 11 characters'
