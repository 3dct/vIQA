"""Module for the Universal Quality Index (UQI) metric.

Notes
-----
The Universal Quality Index [1]_ is a special case of the Structural Similarity Index
(SSIM) [2]_. Therefore, SSIM is used for calculating it.

References
----------
.. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: From error visibility to structural similarity. IEEE
    Transactions on Image Processing, 13(4), 600–612.
    https://doi.org/10.1109/TIP.2003.819861
.. [2] Wang, Z., & Bovik, A. C. (2002). A Universal Image Quality Index. IEEE SIGNAL
    PROCESSING LETTERS, 9(3). https://doi.org/10.1109/97.995823

Examples
--------
    .. doctest-requires:: numpy

        >>> import numpy as np
        >>> from viqa import UQI
        >>> img_r = np.zeros((256, 256))
        >>> img_m = np.ones((256, 256))
        >>> uqi = UQI(data_range=1, normalize=False)
        >>> uqi
        UQI(score_val=None)
        >>> score = uqi.score(img_r, img_m)
        >>> score
        0.0
        >>> uqi.print_score()
        UQI: 1.0
        >>> img_r = np.zeros((256, 256))
        >>> img_m = np.zeros((256, 256))
        >>> uqi.score(img_r, img_m)
        1.0
        >>> img_r = np.random.rand(256, 256)
        >>> img_m = np.random.rand(128, 128)
        >>> uqi.score(img_r, img_m)
        Traceback (most recent call last):
            ...
        ValueError: Image shapes do not match
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

from viqa._metrics import FullReferenceMetricsInterface
from viqa.fr_metrics.ssim import structural_similarity


class UQI(FullReferenceMetricsInterface):
    """Calculate the universal quality index (UQI) between two images.

    Attributes
    ----------
    score_val : float or None
        Score value of the UQI metric.
    parameters : dict
        Dictionary containing the parameters for UQI calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, optional
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the UQI calculation. Passed to
        :py:func:`viqa.utils.load_data` and
        :py:func:`viqa.fr_metrics.ssim.structural_similarity`.
    normalize : bool, default False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`viqa.utils.load_data`.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.
        If False, the input images are converted to grayscale images if necessary.

    Raises
    ------
    ValueError
        If ``data_range`` is not set.

    Notes
    -----
    ``data_range`` for image loading is also used for the UQI calculation if the image
    type is integer and therefore must be set. The parameter is set through the
    constructor of the class and is passed to :py:meth:`score`. UQI [1]_ is a
    full-reference IQA metric. It is based on the human visual system and is designed to
    predict the perceived quality of an image.

    See Also
    --------
    viqa.fr_metrics.ssim.SSIM : Structural similarity index.
    viqa.fr_metrics.msssim.MSSSIM : Multi-scale structural similarity index.

    References
    ----------
    .. [1] Wang, Z., & Bovik, A. C. (2002). A Universal Image Quality Index. IEEE SIGNAL
        PROCESSING LETTERS, 9(3). https://doi.org/10.1109/97.995823
    """

    def __init__(self, data_range=255, normalize=False, **kwargs):
        """Construct method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        self._name = "UQI"

    def score(self, img_r, img_m, **kwargs):
        """Calculate the universal quality index (UQI) between two images.

        Parameters
        ----------
        img_r : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
            Modified image to calculate score of.
        **kwargs : optional
            Additional parameters for the UQI calculation. The keyword arguments are
            passed to :py:func:`viqa.fr_metrics.ssim.structural_similarity`.

        Returns
        -------
        score_val : float
            UQI score value.

        Notes
        -----
        In the original implementation `win_size` is set to 8, here it is set to 7 by
        default, but can be changed to other odd values.
        """
        img_r, img_m = self.load_images(img_r, img_m)

        score_val = structural_similarity(
            img_r,
            img_m,
            data_range=self.parameters["data_range"],
            k_1=0,
            k_2=0,
            alpha=1,
            beta=1,
            gamma=1,
            gaussian_weights=False,
            **kwargs,
        )
        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the UQI score value of the last calculation.

        Parameters
        ----------
        decimals : int, default=2
            Number of decimal places to print the score value.

        Warns
        -----
        RuntimeWarning
            If :py:attr:`score_val` is not available.
        """
        if self.score_val is not None:
            print("UQI: {}".format(np.round(self.score_val, decimals)))
        else:
            warn("No score value for UQI. Run score() first.", RuntimeWarning)
