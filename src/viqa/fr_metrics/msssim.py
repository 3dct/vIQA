"""Module for calculating the multiscale structural similarity index (MS-SSIM) between
two images.

Examples
--------
    .. doctest-skip::

        >>> import numpy as np
        >>> from viqa import MSSSIM
        >>> img_r = np.random.rand(256, 256)
        >>> img_m = np.random.rand(256, 256)
        >>> msssim = MSSSIM()
        >>> msssim.score(img_r, img_m, data_range=1)

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
from piq import multi_scale_ssim
from torch import tensor

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_chromatic


class MSSSIM(FullReferenceMetricsInterface):
    """Calculate the multiscale structural similarity index (MS-SSIM) between two
    images.

    Attributes
    ----------
    score_val : float
        MS-SSIM score value of the last calculation.
    parameters : dict
        Dictionary containing the parameters for MS-SSIM calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the MS-SSIM calculation. Passed to
        :py:func:`viqa.utils.load_data` and :py:meth:`score`.
    normalize : bool, default=False
        If True, the input images are normalized to the ``data_range`` argument.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`.viqa.utils.load_data`.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. note::
            Color images can be used, but it is unclear how the called implementation
            :py:func:`piq.multi_scale_ssim` handles the color channels.

    Raises
    ------
    ValueError
        If ``data_range`` is not set.

    Warnings
    --------
    This metric is not yet tested. The metric should be only used for experimental
    purposes.

    .. todo:: test

    Notes
    -----
    For more information on the MS-SSIM metric, see [1]_.

    See Also
    --------
    viqa.fr_metrics.uqi.UQI : Universal quality index (UQI) between two images.
    viqa.fr_metrics.ssim.SSIM : Structural similarity index (SSIM) between two images.

    References
    ----------
    .. [1] Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multi-scale structural
        similarity for image quality assessment. The Thirty-Seventh Asilomar Conference
        on Signals, Systems & Computers, 1298–1402.
        https://doi.org/10.1109/ACSSC.2003.1292216
    """

    def __init__(self, data_range=255, normalize=False, **kwargs):
        """Construct method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        self._name = "MS-SSIM"

    def score(self, img_r, img_m, dim=None, im_slice=None, **kwargs):
        """
        Calculate the multiscale structural similarity index (MS-SSIM) between two
        images.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.
        dim : {0, 1, 2}, optional
            MS-SSIM for 3D images is calculated as mean over all slices of the given
            dimension.
        im_slice : int, optional
            If given, MS-SSIM is calculated only for the given slice of the 3D image.
        **kwargs : optional
            Additional parameters for MS-SSIM calculation. The keyword arguments are
            passed to :py:func:`piq.multi_scale_ssim`. See the documentation under [2]_.

        Other Parameters
        ----------------
            kernel_size : int, default=11
                The side-length of the sliding window used in comparison. Must be an odd
                value.
            kernel_sigma : float, default=1.5
                Sigma of normal distribution.
            reduction : str, default='mean'
                Specifies the reduction type: 'none', 'mean' or 'sum'.
            scale_weights : list, default=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
                Weights for different scales.
            k1 : float, default=0.01
                Algorithm parameter, K1 (small constant, see [3]_).
            k2 : float, default=0.03
                Algorithm parameter, K2 (small constant, see [3]_).
                Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN
                results.

        .. seealso::
            See :py:func:`.viqa.fr_metrics.ssim.structural_similarity` for more
            information

        Returns
        -------
        score_val : float
            MS-SSIM score value.

        Raises
        ------
        ValueError
            If invalid dimension given in ``dim``. \n
            If images are neither 2D nor 3D. \n
            If images are 3D, but dim is not given. \n
            If ``im_slice`` is given, but not an integer.

        Warns
        -----
        RuntimeWarning
            If ``dim`` or ``im_slice`` is given for 2D images. \n
            If ``im_slice`` is not given, but ``dim`` is given for 3D images, MS-SSIM is
            calculated for the full volume.

        Notes
        -----
        For 3D images if ``dim`` is given, but ``im_slice`` is not, the MS-SSIM is
        calculated for the full volume of the 3D image. This is implemented as `mean` of
        the MS-SSIM values of all slices of the given dimension. If ``dim`` is given and
        ``im_slice`` is given,  the MS-SSIM is calculated for the given slice of the
        given dimension (represents a 2D metric of the given slice).

        References
        ----------
        .. [2] https://piq.readthedocs.io/en/latest/functions.html#piq.multi_scale_ssim
        .. [3] Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003). Multi-scale
            structural similarity for image quality assessment. The Thirty-Seventh
            Asilomar Conference on Signals, Systems & Computers, 1298–1402.
            https://doi.org/10.1109/ACSSC.2003.1292216
        """
        img_r, img_m = self.load_images(img_r, img_m)

        if "scale_weights" in kwargs and type(kwargs["scale_weights"]) is list:
            kwargs["scale_weights"] = tensor(kwargs["scale_weights"])

        if img_r.ndim == 3 and img_r.shape[-1] != 3:
            if (
                dim is not None and type(im_slice) is int
            ):  # if dim and im_slice are given
                # Calculate MS-SSIM for given slice of given dimension
                match dim:
                    case 0:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[im_slice, :, :],
                            img_m[im_slice, :, :],
                            self.parameters["chromatic"],
                        )
                        score_val = multi_scale_ssim(
                            img_r_tensor,
                            img_m_tensor,
                            data_range=self.parameters["data_range"],
                            **kwargs,
                        )
                    case 1:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[:, im_slice, :],
                            img_m[:, im_slice, :],
                            self.parameters["chromatic"],
                        )
                        score_val = multi_scale_ssim(
                            img_r_tensor,
                            img_m_tensor,
                            data_range=self.parameters["data_range"],
                            **kwargs,
                        )
                    case 2:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[:, :, im_slice],
                            img_m[:, :, im_slice],
                            self.parameters["chromatic"],
                        )
                        score_val = multi_scale_ssim(
                            img_r_tensor,
                            img_m_tensor,
                            data_range=self.parameters["data_range"],
                            **kwargs,
                        )
                    case _:
                        raise ValueError(
                            "Invalid dim value. Must be integer of 0, 1 or 2."
                        )
            elif (
                dim is not None and im_slice is None
            ):  # if dim is given, but im_slice is not, calculate MS-SSIM for full
                # volume
                warn(
                    "im_slice is not given. Calculating MS-SSIM for full volume.",
                    RuntimeWarning,
                )
                img_r_tensor, img_m_tensor = _check_chromatic(
                    img_r,
                    img_m,
                    self.parameters["chromatic"],
                )
                score_val = multi_scale_ssim(
                    img_r_tensor,
                    img_m_tensor,
                    data_range=self.parameters["data_range"],
                    **kwargs,
                )
            else:
                if type(im_slice) is not int or None:
                    raise ValueError("im_slice must be an integer.")
                raise ValueError(
                    "If images are 3D, dim and im_slice (optional) must be given."
                )
        elif img_r.ndim == 2 or (img_r.ndim == 3 and img_r.shape[-1] == 3):
            if dim or im_slice:
                warn("dim and im_slice are ignored for 2D images.", RuntimeWarning)
            # Calculate MS-SSIM for 2D images
            img_r_tensor, img_m_tensor = _check_chromatic(
                img_r,
                img_m,
                self.parameters["chromatic"],
            )
            score_val = multi_scale_ssim(
                img_r_tensor,
                img_m_tensor,
                data_range=self.parameters["data_range"],
                **kwargs,
            )
        else:
            raise ValueError("Images must be 2D or 3D.")

        self.score_val = float(score_val)
        return score_val

    def print_score(self, decimals=2):
        """Print the MSSSIM score value of the last calculation.

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
            print("MS-SSIM: {}".format(np.round(self.score_val, decimals)))
        else:
            print("No score value for MS-SSIM. Run score() first.")
