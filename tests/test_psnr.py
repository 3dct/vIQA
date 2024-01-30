import re

import pytest
import numpy as np

from .context import vIQA


def test_psnr_with_identical_images():
    img1 = np.random.rand(100, 100)
    psnr = vIQA.PSNR(data_range=1, normalize=False)
    score = psnr.score(img1, img1)
    assert score == float('inf'), 'PSNR of identical images should be infinity'


def test_psnr_with_completely_different_images():
    img1 = np.zeros((100, 100))
    img2 = np.ones((100, 100))
    psnr = vIQA.PSNR(data_range=1, normalize=False)
    score = psnr.score(img1, img2)
    assert score == 0.0, 'PSNR of completely different images should be 0'


def test_psnr_with_random_images():
    img1 = np.random.rand(100, 100)
    img2 = np.random.rand(100, 100)
    psnr = vIQA.PSNR(data_range=1, normalize=False)
    score = psnr.score(img1, img2)
    assert 0 <= score <= float('inf'), 'PSNR should be between 0 and infinity'


def test_psnr_print_score_without_calculating_score():
    psnr = vIQA.PSNR()
    with pytest.warns(RuntimeWarning, match=re.escape('No score value for PSNR. Run score() first.')):
        psnr.print_score()


def test_psnr_with_different_data_ranges():
    img1 = np.random.rand(100, 100)
    img2 = np.random.rand(100, 100)
    psnr = vIQA.PSNR(data_range=1, normalize=False)
    score1 = psnr.score(img1, img2)
    psnr = vIQA.PSNR(data_range=255, normalize=True)
    score2 = psnr.score(img1, img2)
    assert score1 != score2, 'PSNR should be different for different data ranges'


def test_psnr_with_different_image_shapes():
    img1 = np.random.rand(100, 100)
    img2 = np.random.rand(100, 200)
    psnr = vIQA.PSNR(data_range=1, normalize=False)
    with pytest.raises(ValueError, match='Image shapes do not match'):
        psnr.score(img1, img2)
