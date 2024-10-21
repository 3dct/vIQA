"""Module for calculating the visual saliency index (VSI) between two images.

Examples
--------
    .. doctest-skip::

        >>> import numpy as np
        >>> from viqa import VSI
        >>> img_r = np.random.rand(256, 256)
        >>> img_m = np.random.rand(256, 256)
        >>> vsi = VSI()
        >>> vsi.score(img_r, img_m, data_range=1)

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
from piq import vsi

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_chromatic


class VSI(FullReferenceMetricsInterface):
    """Calculate the visual saliency index (VSI) between two images.

    Attributes
    ----------
    score_val : float
        VSI score value of the last calculation.
    parameters : dict
        Dictionary containing the parameters for VSI calculation.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when
        ``normalize`` is True and for the VSI calculation. Passed to
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
    For more information on the VSI metric, see [1]_.

    .. note::
        The original metric supports RGB images only. This implementation can
        work with grayscale images by copying the luminance channel 3 times.


    References
    ----------
    .. [1] Zhang, L., Shen, Y., & Li, H. (2014). VSI: A visual saliency-induced
        index for perceptual image quality assessment. IEEE Transactions on Image
        Processing, 23(10), 4270–4281. https://doi.org/10.1109/TIP.2014.2346028
    """

    def __init__(self, data_range=255, normalize=False, **kwargs):
        """Construct method."""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, **kwargs)
        self._name = "VSI"

    def score(self, img_r, img_m, dim=None, im_slice=None, **kwargs):
        """Calculate the visual saliency index (VSI) between two images.

        The metric can be calculated for 2D and 3D images. If the images are 3D, the
        metric can be calculated for the full volume or for a given slice of the image
        by setting ``dim`` to the desired dimension and ``im_slice`` to the desired
        slice number.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.
        dim : {0, 1, 2}, optional
            VSI for 3D images is calculated as mean over all slices of the given
            dimension.
        im_slice : int, optional
            If given, VSI is calculated only for the given slice of the 3D image.
        **kwargs : optional
            Additional parameters for VSI calculation. The keyword arguments are passed
            to :py:func:`piq.vsi`. See the documentation under [2]_.

        Other Parameters
        ----------------
        reduction : str, default='mean'
            Specifies the reduction type: 'none', 'mean' or 'sum'.
        c1 : float, default=1.27
            Coefficient to calculate saliency component. See [3]_.
        c2 : float, default=386.0
            Coefficient to calculate gradient component. See [3]_.
        c3 : float, default=130.0
            Coefficient to calculate color component. See [3]_.
        alpha : float, default=0.4
            Power for gradient component.
        beta : float, default=0.02
            Power for color component.
        omega_0 : float, default=0.021
            Coefficient to get log Gabor filter with SDSP. See [4]_.
        sigma_f : float, default=1.34
            Coefficient to get log Gabor filter with SDSP. See [4]_.
        sigma_d : float, default=145.0
            Coefficient to get SDSP. See [4]_.
        sigma_c : float, default=0.001
            Coefficient to get SDSP. See [4]_.

        Returns
        -------
        score_val : float
            VSI score value.

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
            If ``im_slice`` is not given, but ``dim`` is given for 3D images, VSI is
            calculated for the full volume.

        Notes
        -----
        For 3D images if ``dim`` is given, but ``im_slice`` is not, the VSI is
        calculated for the full volume of the 3D image. This is implemented as `mean` of
        the VSI values of all slices of the given dimension. If ``dim`` is given and
        ``im_slice`` is given, the VSI is calculated for the given slice of the given
        dimension (represents a 2D metric of the given slice).

        References
        ----------
        .. [2] https://piq.readthedocs.io/en/latest/functions.html#piq.vsi
        .. [3] Zhang, L., Shen, Y., & Li, H. (2014). VSI: A visual saliency-induced
            index for perceptual image quality assessment. IEEE Transactions on Image
            Processing, 23(10), 4270–4281. https://doi.org/10.1109/TIP.2014.2346028
        .. [4] Zhang, L., Gu, Z., & Li, H. (2013). SDSP: A novel saliency detection
            method by combining simple priors. 2013 IEEE International Conference on
            Image Processing, 171–175. https://api.semanticscholar.org/CorpusID:6028723
        """
        img_r, img_m = self.load_images(img_r, img_m)

        if img_r.ndim == 3:
            if (
                dim is not None and type(im_slice) is int
            ):  # if dim and im_slice are given
                # Calculate VSI for given slice of given dimension
                match dim:
                    case 0:
                        img_r_tensor, img_m_tensor = _check_chromatic(
                            img_r[im_slice, :, :],
                            img_m[im_slice, :, :],
                            self.parameters["chromatic"],
                        )
                        score_val = vsi(
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
                        score_val = vsi(
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
                        score_val = vsi(
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
            ):  # if dim is given, but im_slice is not, calculate VSI for full volume
                warn(
                    "im_slice is not given. Calculating VSI for full volume.",
                    RuntimeWarning,
                )
                img_r_tensor, img_m_tensor = _check_chromatic(
                    img_r,
                    img_m,
                    self.parameters["chromatic"],
                )
                score_val = vsi(
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
        elif img_r.ndim == 2:
            if dim or im_slice:
                warn("dim and im_slice are ignored for 2D images.", RuntimeWarning)
            # Calculate VSI for 2D images
            img_r_tensor, img_m_tensor = _check_chromatic(
                img_r,
                img_m,
                self.parameters["chromatic"],
            )
            score_val = vsi(
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
        """Print the VSI score value of the last calculation.

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
            print("VSI: {}".format(np.round(self.score_val, decimals)))
        else:
            print("No score value for VSI. Run score() first.")
