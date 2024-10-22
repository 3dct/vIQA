import re
import warnings

import pytest
import numpy as np

import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        psnr = viqa.FSIM()
        assert psnr.score_val is None, 'Score value should be None'
        assert psnr.parameters['data_range'] == 255, 'Data range should be 255'
        assert psnr.parameters['normalize'] is False, 'Normalize should be False'
        assert psnr.parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        psnr = viqa.FSIM(data_range=1, normalize=True,chromatic=True)
        assert psnr.score_val is None, 'Score value should be None'
        assert psnr.parameters['data_range'] == 1, 'Data range should be 1'
        assert psnr.parameters['normalize'] is True, 'Normalize should be True'
        assert psnr.parameters['chromatic'] is True, 'Chromatic should be True'

    def test_init_without_data_range(self):
        with pytest.raises(ValueError, match=re.escape('Parameter data_range must be set.')):
            viqa.FSIM(data_range=None, normalize=True, chromatic=True)


class TestScoring2D:
    def test_fsim_with_identical_images_2d(self):
        img_r = np.random.rand(256, 256)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        assert fsim.score(img_r, img_r) == 1.0, 'FSIM of identical images should be 1.0'

    def test_fsim_with_completely_different_images_2d(self, data_2d_255_600x400):
        img_r, img_m = data_2d_255_600x400
        fsim = viqa.FSIM(data_range=255, normalize=False)
        assert pytest.approx( fsim.score(img_r, img_m), abs=1e-4) == 0.9229, 'FSIM of completely different images should be 0'

    def test_fsim_with_random_images_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        assert 0 <= fsim.score(img_r, img_m) <= 1.0, 'FSIM should be between 0 and 1'

    def test_fsim_with_different_data_ranges_2d(self):
        img_r = np.random.rand(256, 256)
        img_m = np.random.rand(256, 256)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        score1 = fsim.score(img_r, img_m)
        fsim = viqa.FSIM(data_range=255, normalize=True)
        score2 = fsim.score(img_r, img_m)
        assert score1 != score2, 'FSIM should be different for different data ranges'

    def test_fsim_score_with_identical_images_2d_dim0(self):
        img_r = np.zeros((128, 128))
        img_m = np.zeros((128, 128))
        fsim = viqa.FSIM(data_range=1)
        with pytest.warns(RuntimeWarning, match=re.escape('dim and im_slice are ignored for 2D images.')):
            fsim.score(img_r, img_m, dim=0, im_slice=32)


class TestScoring3D:
    def test_fsim_with_identical_images_3d_dim1(self):
        img_r = np.random.rand(128, 128, 128)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        assert fsim.score(img_r, img_r, dim=1) == 1.0, 'FSIM of identical images should be 1.0'

    def test_fsim_with_different_images_3d_dim2_slice64(self, data_3d_255_400x400x200):
        img_r, img_m = data_3d_255_400x400x200
        fsim = viqa.FSIM(data_range=255, normalize=False)
        assert pytest.approx(fsim.score(img_r, img_m, dim=2, im_slice=64), abs=1e-4) == 0.9144, 'FSIM of completely different images should be 0'

    def test_fsim_with_different_images_3d_dim0(self, data_3d_255_400x400x200):
        img_r, img_m = data_3d_255_400x400x200
        fsim = viqa.FSIM(data_range=255, normalize=False)
        assert pytest.approx(fsim.score(img_r, img_m, dim=0), abs=1e-4) == 0.9225, 'FSIM of different images should be 0'

    def test_fsim_3d_dim3_slice64(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        with pytest.raises(ValueError, match=re.escape('Invalid dim value. Must be integer of 0, 1 or 2.')):
            fsim.score(img_r, img_m, dim=3, im_slice=64)

    def test_fsim_3d_dim1_slice64(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        assert pytest.approx(fsim.score(img_r, img_m, dim=1, im_slice=64), abs=1e-4) == 0.7708, 'FSIM of completely different images should be 0'

    def test_fsim_3d_dim2(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        fsim.score(img_r, img_m, dim=2)

    def test_fsim_3d(self):
        img_r = np.random.rand(128, 128, 128)
        img_m = np.random.rand(128, 128, 128)
        fsim = viqa.FSIM(data_range=1, normalize=False)
        with pytest.raises(ValueError, match=re.escape('If images are 3D, dim and im_slice (optional) must be given.')):
            fsim.score(img_r, img_m)

    def test_fsim_with_different_data_ranges_3d_dim0_slice64(self, data_3d_255_400x400x200, data_3d_native_400x400x200):
        img_r_255, img_m_255 = data_3d_255_400x400x200
        img_r, img_m = data_3d_native_400x400x200
        fsim = viqa.FSIM(data_range=255, normalize=False)
        score1 = fsim.score(img_r_255, img_m_255, dim=0, im_slice=64)
        fsim = viqa.FSIM(data_range=65535, normalize=True)
        score2 = fsim.score(img_r, img_m, dim=0, im_slice=64)
        assert score1 != score2, 'FSIM should be different for different data ranges'

class TestPrintScore:
    def test_print_score_without_calculating_score_first(self):
        fsim = viqa.FSIM()
        with pytest.warns(RuntimeWarning, match=re.escape(
                'No score value for FSIM. Run score() first.')):
            fsim.print_score()

    def test_print_score_with_calculating_score_first(self, capsys):
        img_r = np.random.rand(128, 128)
        img_m = np.random.rand(128, 128)
        fsim = viqa.FSIM(data_range=1)
        fsim.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fsim.print_score()
            captured = capsys.readouterr()
            assert captured.out == 'FSIM: 0.78\n', 'Printed score should be 0.0'

    def test_print_score_with_different_decimals(self, data_2d_255_600x400, capsys):
        img_r, img_m = data_2d_255_600x400
        fsim = viqa.FSIM()
        fsim.score(img_r, img_m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fsim.print_score(decimals=3)
            captured = capsys.readouterr()
            assert len(captured.out) == 12, 'Printed score should have 12 characters'


def test_fsim_with_random_data_4d():
    img_r = np.random.rand(128, 128, 128, 128)
    img_m = np.random.rand(128, 128, 128, 128)
    fsim = viqa.FSIM(data_range=1, normalize=False)
    with pytest.raises(ValueError, match=re.escape('Images must be 2D or 3D.')):
        fsim.score(img_r, img_m)
