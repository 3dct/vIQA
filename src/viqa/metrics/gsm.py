"""Module for calculating the Gradient Similarity Metric (GSM) metric.

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

import scipy.ndimage as ndi

from viqa._metrics import FullReferenceMetricsInterface
from viqa.utils import _check_imgs, _to_float
from viqa._kernels import *

# Load the kernels as constants
KERNELS_3D = [
            gsm_kernel_x(),
            gsm_kernel_y(),
            gsm_kernel_z(),
            gsm_kernel_xy1(),
            gsm_kernel_xy2(),
            gsm_kernel_yz1(),
            gsm_kernel_yz2(),
            gsm_kernel_xz1(),
            gsm_kernel_xz2(),
            gsm_kernel_xyz1(),
            gsm_kernel_xyz2(),
            gsm_kernel_xyz3(),
            gsm_kernel_xyz4(),
        ]

KERNELS_2D = [
            gsm_kernel_2d_x(),
            gsm_kernel_2d_y(),
            gsm_kernel_2d_xy(),
            gsm_kernel_2d_yx(),
]


class GSM(FullReferenceMetricsInterface):
    """Calculate the gradient similarity (GSM) between two images.

    Parameters
    ----------
    data_range : {1, 255, 65535}, default=255
        Data range of the returned data in data loading. Is used for image loading when `normalize` is True and for the
        GSM calculation.
    normalize : bool, default=False
        If True, the input images are normalized to the `data_range` argument.
    batch : bool, default=False
        If True, the input images are expected to be given as path to a folder containing the images.

        .. note::
            Currently not supported. Added for later implementation.

    **kwargs : optional
        Additional parameters for data loading. The keyword arguments are passed to `viqa.utils.load_data`.
        See below for details.

    Attributes
    ----------
    score_val : float
        GSM score value of the last calculation.

    Raises
    ------
    ValueError
        If the parameter `data_range` is not set.

    Other Parameters
    ----------------
    chromatic : bool, default False
        If True, the input images are expected to be RGB images.

        .. note::
            Currently not supported.

    Notes
    -----
    GSM is a full reference IQA metric based on gradient similarity. It combines luminosity information and
    contrast-structural information. For further details, see [1].
    The parameter `data_range` for image loading is also used for the GSM calculation and therefore must be set.
    The parameter is set through the constructor of the class and is passed to the `score` method.

    References
    ----------
    .. [1] Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on gradient similarity. IEEE
           Transactions on Image Processing, 21(4), 1500–1512. https://doi.org/10.1109/TIP.2011.2175935
    """

    def __init__(self, data_range=255, normalize=False, batch=False, **kwargs):
        """Constructor method"""
        if data_range is None:
            raise ValueError("Parameter data_range must be set.")
        super().__init__(data_range=data_range, normalize=normalize, batch=batch, **kwargs)
        self._parameters.update(**kwargs)

    def score(self, img_r, img_m, dim=None, im_slice=None, **kwargs):
        """Calculate the gradient similarity (GSM) between two images.

        Parameters
        ----------
        img_r : np.ndarray or Tensor or str or os.PathLike
            Reference image to calculate score against.
        img_m : np.ndarray or Tensor or str or os.PathLike
            Distorted image to calculate score of.
        dim : {0, 1, 2}, optional
            GSM for 3D images is calculated as mean over all slices of the given dimension.
        im_slice : int, optional
            If given, GSM is calculated only for the given slice of the 3D image.
        **kwargs : optional
            Additional parameters for GSM calculation. The keyword arguments are passed to
            `viqa.gsm.gradient_similarity_3d()` or `viqa.gsm.gradient_similarity()`.

        Returns
        -------
        score_val : float
            GSM score value.

        Raises
        ------
        ValueError
            If invalid dimension given in parameter dim.
        ValueError
            If images are neither 2D nor 3D.
        ValueError
            If images are 3D, but dim is not given.
        ValueError
            If im_slice is given, but not an integer.

        Warns
        -----
        RuntimeWarning
            If dim or im_slice is given for 2D images.
        RuntimeWarning
            If im_slice is not given, but dim is given for 3D images, MAD is calculated for the full volume.

        Notes
        -----
        This implementation is adapted for 3D images. Therefore, 12 kernels are used instead of the original 4. Also,
        the gradient is calculated by max{convolve(img, kernel)} instead of max{mean2(abs(x * kernel))}.
        """
        img_r, img_m = _check_imgs(
            img_r,
            img_m,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            batch=self._parameters["batch"],
        )

        if img_r.ndim == 3:
            if dim is not None and type(im_slice) is int:  # if dim and im_slice are given
                # Calculate GSM for given slice of given dimension
                match dim:
                    case 0:
                        score_val = gradient_similarity(
                            img_r[im_slice, :, :], img_m[im_slice, :, :],
                            data_range=self._parameters["data_range"], **kwargs
                        )
                    case 1:
                        score_val = gradient_similarity(
                            img_r[:, im_slice, :], img_m[:, im_slice, :],
                            data_range=self._parameters["data_range"], **kwargs
                        )
                    case 2:
                        score_val = gradient_similarity(
                            img_r[:, :, im_slice], img_m[:, :, im_slice],
                            data_range=self._parameters["data_range"], **kwargs
                        )
                    case _:
                        raise ValueError("Invalid dim value. Must be integer of 0, 1 or 2.")
            elif (
                    dim is not None and im_slice is None
            ):  # if dim is given, but im_slice is not, calculate MAD for full volume
                warn("im_slice is not given. Calculating GSM for full volume.", RuntimeWarning)
                score_val = gradient_similarity_3d(img_r, img_m, data_range=self._parameters["data_range"],
                                                   dim=dim, **kwargs)
            else:
                if type(im_slice) is not int or None:
                    raise ValueError("im_slice must be an integer.")
                raise ValueError("If images are 3D, dim and im_slice (optional) must be given.")
        elif img_r.ndim == 2:
            if dim or im_slice:
                warn("dim and im_slice are ignored for 2D images.", RuntimeWarning)
            # Calculate MAD for 2D images
            score_val = gradient_similarity(img_r, img_m, data_range=self._parameters["data_range"], **kwargs)
        else:
            raise ValueError("Images must be 2D or 3D.")

        self.score_val = score_val
        return score_val

    def print_score(self, decimals=2):
        """Print the GSM score value of the last calculation.

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
            print("GSM: {}".format(round(self.score_val, decimals)))
        else:
            warn("No score value for GSM. Run score() first.", RuntimeWarning)


def gradient_similarity_3d(img_r, img_m, dim=0, experimental=False, **kwargs):
    """Calculate the gradient similarity (GSM) between two 3D images.

    Parameters
    ----------
    img_r : np.ndarray
        Reference image to calculate score against
    img_m : np.ndarray
        Distorted image to calculate score of
    dim : {0, 1, 2}, default=2
        Dimension on which the slices are iterated.
    experimental : bool, default=False
        If True, calculate GSM for the full volume with experimental 3D kernels. If False, calculate GSM for all slices
        of the given dimension and calculate mean over all single slice values.

        .. warning::
            This is experimental and the resulting values are not validated.

    **kwargs : optional
            Additional parameters for GSM calculation. The keyword arguments are passed to
            `viqa.gsm.gradient_similarity()`.

    Returns
    -------
    gsm_score : float
        GSM score value.

    Raises
    ------
    ValueError
        If the parameter `dim` is not an integer of 0, 1 or 2.

    References
    ----------
    .. [1] Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on gradient similarity. IEEE
           Transactions on Image Processing, 21(4), 1500–1512. https://doi.org/10.1109/TIP.2011.2175935
    """
    if not experimental:
        x, y, z = img_r.shape  # get image dimensions
        scores = []
        # Calculate MAD for all slices of the given dimension
        match dim:
            case 0:
                for slice_ in range(x):
                    scores.append(
                        gradient_similarity(
                            img_r[slice_, :, :], img_m[slice_, :, :], **kwargs
                        )
                    )
            case 1:
                for slice_ in range(y):
                    scores.append(
                        gradient_similarity(
                            img_r[:, slice_, :], img_m[:, slice_, :], **kwargs
                        )
                    )
            case 2:
                for slice_ in range(z):
                    scores.append(
                        gradient_similarity(
                            img_r[:, :, slice_], img_m[:, :, slice_], **kwargs
                        )
                    )
            case _:
                raise ValueError("Invalid dim value. Must be integer of 0, 1 or 2.")
        return np.mean(np.array(scores))
    else:
        return gradient_similarity(img_r, img_m, **kwargs)


def gradient_similarity(img_r, img_m, data_range=255, c=200, p=0.1):
    """Calculate the gradient similarity between two images.

    Parameters
    ----------
    img_r : np.ndarray
        Reference image to calculate score against
    img_m : np.ndarray
        Distorted image to calculate score of
    data_range : {1, 255, 65535}
        Data range of the input images
    c : int, default=200
        Constant as masking parameter. Typically, 200 <= c <= 1000. See [1] for details.
    p : float, default=0.1
        Constant for weighting between luminance and structure similarity. 0 <= p <= 1. Higher p means more accentuation
        of luminance. Should be significantly smaller than 0.5. See [1] for details.

    Returns
    -------
    gsm_score : float
        GSM score value.

    References
    ----------
    .. [1] Liu, A., Lin, W., & Narwaria, M. (2012). Image quality assessment based on gradient similarity. IEEE
           Transactions on Image Processing, 21(4), 1500–1512. https://doi.org/10.1109/TIP.2011.2175935
    """

    gradients_r = []
    gradients_m = []

    if img_r.ndim == 3:
        kernels = KERNELS_3D
    elif img_r.ndim == 2:
        kernels = KERNELS_2D
    else:
        raise ValueError("Images must be 2D or 3D.")

    for kernel in kernels:
        gradients_r.append(ndi.correlate(img_r, kernel))
        gradients_m.append(ndi.correlate(img_m, kernel))

    # key for sorting the gradients by their mean of absolute values
    def _mean_of_abs(input_array):
        return np.mean(np.abs(input_array))

    img_r_gradient = sorted(gradients_r, key=_mean_of_abs, reverse=True)[0]
    img_m_gradient = sorted(gradients_m, key=_mean_of_abs, reverse=True)[0]

    img_r_gradient = _to_float(img_r_gradient)
    img_m_gradient = _to_float(img_m_gradient)
    k = c / max(np.max(img_r_gradient), np.max(img_m_gradient))
    r = np.abs(img_r_gradient - img_m_gradient) / max(
        img_r_gradient, img_m_gradient, key=_mean_of_abs
    )
    con_struc_sim = ((2 * (1 - r)) + k) / (1 + (1 - r) ** 2 + k)
    lum_sim = 1 - ((img_r - img_m) / data_range) ** 2
    weight = p * con_struc_sim
    quality = (1 - weight) * con_struc_sim + weight * lum_sim

    gsm_score = np.nanmean(quality)
    return gsm_score
