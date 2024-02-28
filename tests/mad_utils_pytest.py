import re

import pytest
import numpy as np

from .context import viqa


def test_most_apparent_distortion_negative_block_size(self):
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    with pytest.raises(ValueError, match=re.escape('Block size must be a positive integer.')):
        viqa.metrics.mad.most_apparent_distortion(img_r, img_m, block_size=-1)


def test_most_apparent_distortion_manual_thresh(self):
    img_r = np.random.rand(128, 128)
    img_m = np.random.rand(128, 128)
    viqa.metrics.mad.most_apparent_distortion(img_r, img_m, thresh_1=2.55, thresh_2=3.35)
