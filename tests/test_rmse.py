import re

import pytest
import numpy as np

from .context import vIQA


def test_rmse_with_identical_images():
    img_r = np.zeros((256, 256))
    img_m = np.zeros((256, 256))
    rmse = vIQA.RMSE()
    assert rmse.score(img_r, img_m) == 0.0, 'RMSE of identical images should be 0'


def test_rmse_with_completely_different_images():
    img_r = np.zeros((256, 256))
    img_m = np.ones((256, 256))
    rmse = vIQA.RMSE()
    assert rmse.score(img_r, img_m) == 1.0, 'RMSE of completely different images should be 1'


def test_rmse_with_random_images():
    img_r = np.random.rand(256, 256)
    img_m = np.random.rand(256, 256)
    rmse = vIQA.RMSE()
    assert 0 <= rmse.score(img_r, img_m) <= 1.0, 'RMSE should be between 0 and 1'


def test_print_score_without_calculating_score_first():
    rmse = vIQA.RMSE()
    with pytest.warns(RuntimeWarning, match=re.escape('No score value for RMSE. Run score() first.')):
        rmse.print_score()


def test_psnr_with_different_data_ranges():
    img_r = np.random.rand(256, 256)
    img_m = np.random.rand(256, 256)
    rmse = vIQA.RMSE()
    score1 = rmse.score(img_r, img_m)
    rmse = vIQA.RMSE(data_range=255, normalize=True)
    score2 = rmse.score(img_r, img_m)
    assert score1 != score2, 'RMSE should be different for different data ranges'


def test_rmse_with_different_sized_images():
    img_r = np.random.rand(256, 256)
    img_m = np.random.rand(128, 128)
    rmse = vIQA.RMSE()
    with pytest.raises(ValueError, match='Image shapes do not match'):
        rmse.score(img_r, img_m)
