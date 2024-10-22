import re
import warnings

import pytest
import numpy as np

import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        mad = viqa.MAD()
        assert mad.score_val is None, 'Score value should be None'
        assert mad.parameters['data_range'] == 255, 'Data range should be None'
        assert mad.parameters['normalize'] is False, 'Normalize should be False'
        assert mad.parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        mad = viqa.MAD(data_range=1, normalize=True, chromatic=True)
        assert mad.score_val is None, 'Score value should be None'
        assert mad.parameters['data_range'] == 1, 'Data range should be 255'
        assert mad.parameters['normalize'] is True, 'Normalize should be True'
        assert mad.parameters['chromatic'] is True, 'Chromatic should be True'

    def test_init_without_data_range(self):
        with pytest.raises(ValueError, match=re.escape('Parameter data_range must be set.')):
            viqa.MAD(data_range=None, normalize=True, chromatic=True)


class TestScoring2D:
    def test_mad_score_with_identical_images_2d(self):
        img_r = np.zeros((128, 128))
        img_m = np.zeros((128, 128))
        mad = viqa.MAD(data_range=1)
        assert mad.score(img_r, img_m) == 0.0, 'MAD of identical images should be 0'

    def test_mad_score_with_different_images_2d(self, data_2d_255_600x400):
        img_r, img_m = data_2d_255_600x400
        mad = viqa.MAD(data_range=255, normalize=False)
        assert mad.score(img_r, img_m) != 0.0, 'MAD of completely different images should not be 0'

    def test_mad_score_with_random_images_2d(self):
        img_r = np.random.rand(128, 128)
        img_m = np.random.rand(128, 128)
        mad = viqa.MAD(data_range=1)
        assert 0 <= mad.score(img_r, img_m) <= 1.0, 'MAD should be between 0 and 1'

    def test_mad_score_with_different_data_ranges_2d(self):
        img_r = np.random.rand(128, 128)
        img_m = np.random.rand(128, 128)
        mad = viqa.MAD(data_range=1)
        score1 = mad.score(img_r, img_m)
        mad = viqa.MAD(data_range=255, normalize=True)
        score2 = mad.score(img_r, img_m)
        assert score1 != score2, 'MAD should be different for different data ranges'

    def test_mad_score_with_identical_images_2d_dim0(self):
        img_r = np.zeros((128, 128))
        img_m = np.zeros((128, 128))
        mad = viqa.MAD(data_range=1)
        with pytest.warns(RuntimeWarning, match=re.escape('dim and im_slice are ignored for 2D images.')):
            mad.score(img_r, img_m, dim=0, im_slice=32)


class TestScoring3D:
    def test_mad_score_with_identical_images_3d_dim1(self):
        img_r = np.zeros((128, 128, 128))
        img_m = np.zeros((128, 128, 128))
        mad = viqa.MAD(data_range=1)
        assert mad.score(img_r, img_m, dim=1) == 0.0, 'MAD of identical images should be 0'

    def test_mad_score_with_different_images_3d_dim2_slice64(self, data_3d_255_400x400x200):
        img_r, img_m = data_3d_255_400x400x200
        mad = viqa.MAD()
        assert mad.score(img_r, img_m, dim=2, im_slice=64) != 0.0, 'MAD of completely different images should not be 0'

    # takes 7hr 41min
    def test_mad_score_with_different_images_3d_dim0(self, data_3d_255_400x400x200):
        img_r, img_m = data_3d_255_400x400x200
        mad = viqa.MAD()
        assert 0 <= mad.score(img_r, img_m, dim=0) <= np.inf, 'MAD should be between 0 and inf'

    def test_mad_score_3d_dim3_slice64(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        mad = viqa.MAD(data_range=1)
        with pytest.raises(ValueError, match=re.escape('Invalid dim value. Must be integer of 0, 1 or 2.')):
            mad.score(img_r, img_m, dim=3, im_slice=64)

    def test_mad_score_3d_dim1_slice64(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        mad = viqa.MAD(data_range=1)
        mad.score(img_r, img_m, dim=1, im_slice=64)

    def test_mad_score_3d_dim2(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        mad = viqa.MAD(data_range=1)
        mad.score(img_r, img_m, dim=2)

    def test_mad_score_3d(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        mad = viqa.MAD(data_range=1)
        with pytest.raises(ValueError, match=re.escape('If images are 3D, dim and im_slice (optional) must be given.')):
            mad.score(img_r, img_m)

    def test_mad_score_with_different_data_ranges_3d_dim0_slice64(self, data_3d_255_400x400x200,
                                                                  data_3d_native_400x400x200):
        img_r_255, img_m_255 = data_3d_255_400x400x200
        img_r, img_m = data_3d_native_400x400x200
        mad = viqa.MAD(data_range=255)
        score1 = mad.score(img_r_255, img_m_255, dim=0, im_slice=64)
        mad_2 = viqa.MAD(data_range=65535)
        score2 = mad_2.score(img_r, img_m, dim=0, im_slice=64)
        assert score1 != score2, 'MAD should be different for different data ranges'


class TestPrintScore:
    def test_print_score_without_calculating_score_first(self):
        mad = viqa.MAD()
        with pytest.warns(RuntimeWarning, match=re.escape('No score value for MAD. Run score() first.')):
            mad.print_score()

    def test_print_score_with_calculating_score_first(self, capsys):
        img_r = np.random.rand(128, 128)
        img_m = np.random.rand(128, 128)
        mad = viqa.MAD(data_range=1)
        mad.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mad.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'MAD: 0.0\n', 'Printed score is incorrect'

    def test_print_score_with_different_decimals(self, data_2d_255_600x400, capsys):
        img_r, img_m = data_2d_255_600x400
        mad = viqa.MAD()
        mad.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mad.print_score(decimals=2)
            captured = capsys.readouterr()
            assert len(captured.out) == 10, 'Printed score should have 10 characters'


def test_mad_score_with_random_data_4d():
    img_r = np.random.rand(128, 128, 128, 128)
    img_m = np.random.rand(128, 128, 128, 128)
    mad = viqa.MAD(data_range=1)
    with pytest.raises(ValueError, match=re.escape('Images must be 2D or 3D.')):
        mad.score(img_r, img_m)
