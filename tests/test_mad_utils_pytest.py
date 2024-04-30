import re

import pytest
import numpy as np

import viqa


def test_most_apparent_distortion_negative_block_size():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('block_size must be positive.')):
        viqa.fr_metrics.mad.most_apparent_distortion(img_r, img_m, block_size=-1)


def test_most_apparent_distortion_manual_thresh():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    viqa.fr_metrics.mad.most_apparent_distortion(img_r, img_m, thresh_1=2.55, thresh_2=3.35)


def test_most_apparent_distortion_3d_invalid_dim_value():
    img_r = np.random.rand(128, 128, 128)
    img_m = np.random.rand(128, 128, 128)
    with pytest.raises(ValueError, match=re.escape('Invalid dim value. Must be integer of 0, 1 or 2.')):
        viqa.fr_metrics.mad.most_apparent_distortion_3d(img_r, img_m, dim=3)


def test_low_quality_invalid_weights_length():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('weights must be of length scales_num (4).')):
        viqa.fr_metrics.mad._low_quality(img_r, img_m, scales_num=4, weights=[1, 2, 3, 4, 5])


def test_high_quality_missing_display_function():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('If account_monitor is True, display_function must be given.')):
        viqa.fr_metrics.mad._high_quality(img_r, img_m, account_monitor=True)


def test_high_quality_account_monitor():
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    viqa.fr_metrics.mad._high_quality(img_r, img_m, account_monitor=True, display_function={'disp_res': 300, 'view_dis': 30})
