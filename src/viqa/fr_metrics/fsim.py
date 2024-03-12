"""Module for calculating the feature similarity (FSIM) between two images.

Examples
--------
    .. todo:: Add examples

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

import torch
from piq import fsim

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_imgs, _check_chromatic


class FSIM(FullReferenceMetricsInterface):
    """Calculate the feature similarity (FSIM) between two images.

    Attributes
    ----------
    score_val : float
        FSIM score value of the last calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the FSIM calculation.
    normalize : bool, default=False
        If True, the input images are normalized to the ``data_range`` argument.
    batch : bool, default=False
        If True, the input images are expected to be given as path to a folder
        containing the images.

        .. note::
            Currently not supported. Added for later implementation.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to
        :py:func:`.viqa.utils.load_data`.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

    Raises
    ------
    ValueError
        If ``data_range`` is not set.

    Notes
    -----
    For more information on the FSIM metric, see [1].

    References
    ----------
    [1]: Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011). FSIM: A feature similarity
    index for image quality assessment. IEEE Transactions on Image Processing, 20(8).
    https://doi.org/10.1109/TIP.2011.2109730
    """

    def __init__(self, data_range=255, normalize=False, batch=False, **kwargs):
        """Constructor method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(
            data_range=data_range, normalize=normalize, batch=batch, **kwargs
        )

    def score(self, img_r, img_m, dim=None, im_slice=None, **kwargs):
        img_r, img_m = _check_imgs(
            img_r,
            img_m,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            batch=self._parameters["batch"],
        )

        if img_r.ndim == 3:
            if (
                dim is not None and type(im_slice) is int
            ):  # if dim and im_slice are given
                # Calculate FSIM for given slice of given dimension
                match dim:
                    case 0:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[im_slice, :, :],
                            img_m[im_slice, :, :],
                            self._parameters["chromatic"],
                        )
                        score_val = fsim(
                            img_r_tensor,
                            img_m_tensor,
                            data_range=self._parameters["data_range"],
                            chromatic=self._parameters["chromatic"],
                            **kwargs,
                        )
                    case 1:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[:, im_slice, :],
                            img_m[:, im_slice, :],
                            self._parameters["chromatic"],
                        )
                        score_val = fsim(
                            img_r_tensor,
                            img_m_tensor,
                            data_range=self._parameters["data_range"],
                            chromatic=self._parameters["chromatic"],
                            **kwargs,
                        )
                    case 2:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[:, :, im_slice],
                            img_m[:, :, im_slice],
                            self._parameters["chromatic"],
                        )
                        score_val = fsim(
                            img_r_tensor,
                            img_m_tensor,
                            data_range=self._parameters["data_range"],
                            chromatic=self._parameters["chromatic"],
                            **kwargs,
                        )
                    case _:
                        raise ValueError(
                            "Invalid dim value. Must be integer of 0, 1 or 2."
                        )
            elif (
                dim is not None and im_slice is None
            ):  # if dim is given, but im_slice is not, calculate FSIM for full volume
                warn(
                    "im_slice is not given. Calculating FSIM for full volume.",
                    RuntimeWarning,
                )
                img_r_tensor, img_m_tensor = _check_chromatic(
                    img_r,
                    img_m,
                    self._parameters["chromatic"],
                )
                score_val = fsim(
                    img_r_tensor,
                    img_m_tensor,
                    data_range=self._parameters["data_range"],
                    chromatic=self._parameters["chromatic"],
                    **kwargs,
                )
            else:
                if type(im_slice) is not int or None:
                    raise ValueError("im_slice must be an integer.")
                raise ValueError(
                    "If images are 3D, dim and im_slice (optional) must be given."
                )
        elif img_r.ndim == 2:
            if dim or im_slice:
                warn("dim and im_slice are ignored for 2D images.", RuntimeWarning)
            # Calculate FSIM for 2D images
            img_r_tensor, img_m_tensor = _check_chromatic(
                img_r,
                img_m,
                self._parameters["chromatic"],
            )
            score_val = fsim(
                img_r_tensor,
                img_m_tensor,
                data_range=self._parameters["data_range"],
                chromatic=self._parameters["chromatic"],
                **kwargs,
            )
        else:
            raise ValueError("Images must be 2D or 3D.")

        self.score_val = float(score_val)
        return score_val

    def print_score(self, decimals=2):
        if self.score_val is not None:
            print("FSIM: {}".format(round(self.score_val, decimals)))
        else:
            print("No score value for FSIM. Run score() first.")
