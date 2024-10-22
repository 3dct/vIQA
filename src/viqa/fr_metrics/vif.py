"""Module for calculating the visual information fidelity in pixel domain (VIFp) between
two images.

Examples
--------
    .. doctest-skip::

        >>> import numpy as np
        >>> from viqa import VIFp
        >>> img_r = np.random.rand(256, 256)
        >>> img_m = np.random.rand(256, 256)
        >>> vifp = VIFp()
        >>> vifp.score(img_r, img_m, data_range=1)

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
from piq import vif_p

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_chromatic


class VIFp(FullReferenceMetricsInterface):
    """Calculate the visual information fidelity in pixel domain (VIFp) between two
    images.

    Attributes
    ----------
    score_val : float
        VIFp score value of the last calculation.
    parameters : dict
        Dictionary containing the parameters for VIFp calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the VIFp calculation. Passed to
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
            :py:func:`piq.vif_p` handles the color channels.

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
    For more information on the VIFp metric, see [1]_.

    References
    ----------
    .. [1] Sheikh, H. R., & Bovik, A. C. (2006). Image information and visual quality.
        IEEE Transactions on Image Processing, 15(2), 430–444.
        https://doi.org/10.1109/TIP.2005.859378
    """

    def __init__(self, data_range=255, normalize=False, **kwargs):
        """Construct method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        self._name = "VIFp"

    def score(self, img_r, img_m, dim=None, im_slice=None, **kwargs):
        """
        Calculate the visual information fidelity in pixel domain (VIFp) between two
        images.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.
        dim : {0, 1, 2}, optional
            VIFp for 3D images is calculated as mean over all slices of the given
            dimension.
        im_slice : int, optional
            If given, VIFp is calculated only for the given slice of the 3D image.
        **kwargs : optional
            Additional parameters for VIFp calculation. The keyword arguments are passed
            to :py:func:`piq.vif_p`. See the documentation under [2]_.

        Other Parameters
        ----------------
        sigma_n_sq : float, default=2.0
            HVS model parameter (variance of the visual noise). See [3]_.
        reduction : str, default='mean'
            Specifies the reduction type: 'none', 'mean' or 'sum'.

        Returns
        -------
        score_val : float
            VIFp score value.

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
            If ``im_slice`` is not given, but ``dim`` is given for 3D images, VIFp is
            calculated for the full volume.

        Notes
        -----
        For 3D images if ``dim`` is given, but ``im_slice`` is not, the VIFp is
        calculated for the full volume of the 3D image. This is implemented as `mean` of
        the VIFp values of all slices of the given dimension. If ``dim`` is given and
        ``im_slice`` is given, the VIFp is calculated for the given slice of the given
        dimension (represents a 2D metric of the given slice).

        References
        ----------
        .. [2] https://piq.readthedocs.io/en/latest/functions.html#piq.vif_p
        .. [3] Sheikh, H. R., & Bovik, A. C. (2006). Image information and visual
            quality. IEEE Transactions on Image Processing, 15(2), 430–444.
            https://doi.org/10.1109/TIP.2005.859378
        """
        img_r, img_m = self.load_images(img_r, img_m)

        if img_r.ndim == 3 and img_r.shape[-1] != 3:
            if (
                dim is not None and type(im_slice) is int
            ):  # if dim and im_slice are given
                # Calculate VIFp for given slice of given dimension
                match dim:
                    case 0:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[im_slice, :, :],
                            img_m[im_slice, :, :],
                            self.parameters["chromatic"],
                        )
                        score_val = vif_p(
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
                        score_val = vif_p(
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
                        score_val = vif_p(
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
            ):  # if dim is given, but im_slice is not, calculate VIFp for full volume
                warn(
                    "im_slice is not given. Calculating VIFp for full volume.",
                    RuntimeWarning,
                )
                img_r_tensor, img_m_tensor = _check_chromatic(
                    img_r,
                    img_m,
                    self.parameters["chromatic"],
                )
                score_val = vif_p(
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
            # Calculate VIFp for 2D images
            img_r_tensor, img_m_tensor = _check_chromatic(
                img_r,
                img_m,
                self.parameters["chromatic"],
            )
            score_val = vif_p(
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
        """Print the VIFp score value of the last calculation.

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
            print("VIFp: {}".format(np.round(self.score_val, decimals)))
        else:
            print("No score value for VIFp. Run score() first.")
