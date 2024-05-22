"""Module for the Q-Measure [1]_ metric.

References
----------
.. [1] Reiter, M., Weiß, D., Gusenbauer, C., Erler, M., Kuhn, C., Kasperl, S., &
    Kastner, J. (2014). Evaluation of a Histogram-based Image Quality Measure for X-ray
    computed Tomography. 5th Conference on Industrial Computed Tomography (iCT) 2014,
    25-28 February 2014, Wels, Austria. e-Journal of Nondestructive Testing Vol. 19(6).
    https://www.ndt.net/?id=15715

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import QMeasure
        >>> img = np.random.rand(256, 256)
        >>> qm = QMeasure()
        >>> qm
        QMeasure(score_val=None)
        >>> score = qm.score(img, hist_bins=128, num_peaks=2)
"""

# Authors
# -------
# Author: Lukas Behammer
# Research Center Wels
# University of Applied Sciences Upper Austria, 2023
# CT Research Group
#
# Modifications
# -------------
# Original code, 2024, Lukas Behammer

#
# License
# -------
# BSD-3-Clause License


from warnings import warn

import numpy as np

from viqa._metrics import NoReferenceMetricsInterface
from viqa.load_utils import load_data
from viqa.nr_metrics.qmeasure_utils import qmeasurecalc
from viqa.utils import _to_float


class QMeasure(NoReferenceMetricsInterface):
    """Class to calculate the Q-Measure [1]_ for an image.

    Attributes
    ----------
    score_val : float
        Q-Measure value of the last calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True. Passed to :py:func:`viqa.utils.load_data`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.utils.load_data`.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. note::
            Currently not supported.

    Notes
    -----
    This metric will always be calculated with float32 precision.

    References
    ----------
    .. [1] Reiter, M., Weiß, D., Gusenbauer, C., Erler, M., Kuhn, C., Kasperl, S., &
        Kastner, J. (2014). Evaluation of a Histogram-based Image Quality Measure for
        X-ray computed Tomography. 5th Conference on Industrial Computed Tomography
        (iCT) 2014, 25-28 February 2014, Wels, Austria. e-Journal of Nondestructive
        Testing Vol. 19(6). https://www.ndt.net/?id=15715
    """

    def __init__(self, data_range=255, normalize=False, **kwargs) -> None:
        """Constructor method."""
        super().__init__(
            data_range=data_range, normalize=normalize, **kwargs
        )
        self._name = "Q-Measure"

    def score(self, img, **kwargs):
        """Calculate the Q-Measure between two images.

        Parameters
        ----------
        img : np.ndarray or Tensor or str or os.PathLike
            Image to calculate score of.
        **kwargs : optional
            Additional parameters for Q-Measure calculation. The keyword arguments are
            passed to :py:func:`viqa.nr_metrics.qmeasure_utils.qmeasurecalc.qmeasure`.

        Other Parameters
        ----------------
        hist_bins : int, default=128
            Number of bins for the histogram calculation.
        num_peaks : int, default=2
            Number of peaks to consider in the histogram.

        Returns
        -------
        score_val : float
            Q-Measure value.
        """
        # Load image
        img = load_data(
            img,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
        )

        # Convert to float and get min and max values
        img = _to_float(img, np.float32)
        img_min = int(img.min())
        img_max = int(img.max())

        # Get additional parameters
        hist_bins = kwargs.pop("hist_bins", 128)
        num_peaks = kwargs.pop("num_peaks", 2)

        # Calculate score
        score_val = qmeasurecalc.qmeasure(img, img_min, img_max, hist_bins, num_peaks)
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the Q-Measure value of the last calculation.

        Parameters
        ----------
        decimals : int, default=2
            Number of decimal places to print the score value.

        Warns
        -----
        RuntimeWarning
            If no score value is available. Run score() first.
        """
        if self.score_val is not None:
            print("Q-Measure: {}".format(round(self.score_val, decimals)))
        else:
            warn("No score value for Q-Measure. Run score() first.", RuntimeWarning)
