import re

import pytest
import numpy as np

import viqa


def test_pixel_to_lightness():
    img = np.array([[0, 127], [255, 63]], dtype=np.uint8)
    luminance_function = {'b': 127, 'k': 2,  'gamma': 0.5}
    expected = np.array([[0.0, 0.5], [1.0, 0.25]])
    assert np.allclose(viqa.metrics.mad._pixel_to_lightness(img, luminance_function), expected)

def test_contrast_sensitivity_function():
    m, n, nfreq = 256, 256, 32
    csf = viqa.metrics.mad._contrast_sensitivity_function(m, n, nfreq)
    assert csf.shape == (m, n)

def test_min_std():
    img = np.ones((256, 256))
    block_size, stride = 16, 8
    assert np.all(viqa.metrics.mad._min_std(img, block_size, stride) == 0.0)

def test_get_statistics():
    img = np.ones((256, 256))
    block_size, stride = 16, 8
    std, skw, krt = viqa.metrics.mad._get_statistics(img, block_size, stride)
    assert np.all(std == 0.0)
    assert np.all(skw == 0.0)
    assert np.all(krt == 0.0)

def test_high_quality():
    img_r = np.ones((256, 256))
    img_m = np.ones((256, 256))
    assert viqa.metrics.mad._high_quality(img_r, img_m) == 0.0

def test_low_quality():
    img_r = np.ones((256, 256))
    img_m = np.ones((256, 256))
    assert viqa.metrics.mad._low_quality(img_r, img_m) == 0.0

def test_most_apparent_distortion():
    img_r = np.ones((256, 256))
    img_m = np.ones((256, 256))
    assert viqa.metrics.mad.most_apparent_distortion(img_r, img_m) == 0.0

def test_most_apparent_distortion_3d():
    img_r = np.ones((256, 256, 256))
    img_m = np.ones((256, 256, 256))
    assert viqa.metrics.mad.most_apparent_distortion_3d(img_r, img_m) == 0.0
