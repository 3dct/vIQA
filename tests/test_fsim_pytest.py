import re
import warnings

import pytest
import numpy as np

# from .context import viqa
import viqa


class TestInit:
    def test_init_with_default_parameters(self):
        psnr = viqa.FSIM()
        assert psnr.score_val is None, 'Score value should be None'
        assert psnr._parameters['data_range'] == 255, 'Data range should be 255'
        assert psnr._parameters['normalize'] is False, 'Normalize should be False'
        assert psnr._parameters['batch'] is False, 'Batch should be False'
        assert psnr._parameters['chromatic'] is False, 'Chromatic should be False'

    def test_init_with_custom_parameters(self):
        psnr = viqa.FSIM(data_range=1, normalize=True, batch=True, chromatic=True)
        assert psnr.score_val is None, 'Score value should be None'
        assert psnr._parameters['data_range'] == 1, 'Data range should be 1'
        assert psnr._parameters['normalize'] is True, 'Normalize should be True'
        assert psnr._parameters['batch'] is True, 'Batch should be True'
        assert psnr._parameters['chromatic'] is True, 'Chromatic should be True'

    def test_init_without_data_range(self):
        with pytest.raises(ValueError, match=re.escape('Parameter data_range must be set.')):
            viqa.FSIM(data_range=None, normalize=True, batch=True, chromatic=True)

