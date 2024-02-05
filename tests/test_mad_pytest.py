import re
import warnings

import pytest
import numpy as np

from .context import vIQA


class TestInit:
    def test_init_with_default_parameters(self):
        mad = vIQA.MAD()
        assert mad.score_val is None, 'Score value should be None'
        assert mad._parameters['data_range'] is None, 'Data range should be None'
        assert mad._parameters['normalize'] is False, 'Normalize should be False'
        assert mad._parameters['batch'] is False, 'Batch should be False'
        assert mad._parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        mad = vIQA.MAD(data_range=255, normalize=True, batch=True, chromatic=True)
        assert mad.score_val is None, 'Score value should be None'
        assert mad._parameters['data_range'] == 255, 'Data range should be 255'
        assert mad._parameters['normalize'] is True, 'Normalize should be True'
        assert mad._parameters['batch'] is True, 'Batch should be True'
        assert mad._parameters['chromatic'] is True, 'Chromatic should be True'


class TestScoring2D:
    # TODO: Add tests for different combinations of dim and im_slice
    def test_mad_score_with_identical_images_2d(self):
        img_r = np.zeros((128, 128))
        img_m = np.zeros((128, 128))
        mad = vIQA.MAD()
        assert mad.score(img_r, img_m) == 0.0, 'MAD of identical images should be 0'

    def test_mad_score_with_completely_different_images_2d(self):
        img_r = np.zeros((128, 128))
        img_m = np.ones((128, 128))
        mad = vIQA.MAD()
        assert mad.score(img_r, img_m) != 0.0, 'MAD of completely different images should not be 0'

    def test_mad_score_with_random_images_2d(self):
        img_r = np.random.rand(128, 128)
        img_m = np.random.rand(128, 128)
        mad = vIQA.MAD()
        assert 0 <= mad.score(img_r, img_m) <= 1.0, 'MAD should be between 0 and 1'

    def test_mad_score_with_different_data_ranges_2d(self):
        img_r = np.random.rand(128, 128)
        img_m = np.random.rand(128, 128)
        mad = vIQA.MAD()
        score1 = mad.score(img_r, img_m)
        mad = vIQA.MAD(data_range=255, normalize=True)
        score2 = mad.score(img_r, img_m)
        assert score1 != score2, 'MAD should be different for different data ranges'


class TestScoring3D:
    # TODO: Add tests for different combinations of dim and im_slice
    def test_mad_score_with_identical_images_3d(self):
        img_r = np.zeros((128, 128, 128))
        img_m = np.zeros((128, 128, 128))
        mad = vIQA.MAD()
        assert mad.score(img_r, img_m) == 0.0, 'MAD of identical images should be 0'
        
    def test_mad_score_with_completely_different_images_3d(self):
        img_r = np.zeros((128, 128, 128))
        img_m = np.ones((128, 128, 128))
        mad = vIQA.MAD()
        assert mad.score(img_r, img_m) != 0.0, 'MAD of completely different images should not be 0'
        
    def test_mad_score_with_random_images_3d(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        mad = vIQA.MAD()
        assert 0 <= mad.score(img_r, img_m) <= 1.0, 'MAD should be between 0 and 1'
        
    def test_mad_score_with_different_data_ranges_3d(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        mad = vIQA.MAD()
        score1 = mad.score(img_r, img_m)
        mad = vIQA.MAD(data_range=255, normalize=True)
        score2 = mad.score(img_r, img_m)
        assert score1 != score2, 'MAD should be different for different data ranges'


class TestPrintScore:
    def test_print_score_without_calculating_score_firstself(self):
        mad = vIQA.MAD()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for MAD. Run score() first.')):
            mad.print_score()

    def test_print_score_with_calculating_score_first(self, capsys):
        img_r = np.random.rand(128, 128)
        img_m = np.random.rand(128, 128)
        mad = vIQA.MAD()
        mad.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mad.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'MAD: 0.0\n', 'Printed score is incorrect'

    # TODO: Add tests for different values of decimals
