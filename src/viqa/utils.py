"""Module for utility functions.

Examples
--------
    .. doctest-skip::

        >>> from viqa import load_data
        >>> img_path = "path/to/image.mhd"
        >>> img = load_data(img_path)

        >>> from viqa import export_results, PSNR, RMSE
        >>> metrics = [PSNR, RMSE]
        >>> output_path = "path/to/output"
        >>> filename = "metrics.csv"
        >>> export_results(metrics, output_path, filename)

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

import csv
import math
import os
from datetime import datetime
from importlib.metadata import version
from typing import Tuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
import skimage as ski
import torch
from skimage.transform import resize
from torch import Tensor

from viqa.load_utils import load_data


def _check_imgs(
    img_r: np.ndarray | Tensor | str | os.PathLike,
    img_m: np.ndarray | Tensor | str | os.PathLike,
    **kwargs,
) -> Tuple[list | np.ndarray, list | np.ndarray]:
    """Check if two images are of the same type and shape."""
    chromatic = kwargs.pop("chromatic", False)
    # load images
    img_r_loaded = load_data(img_r, **kwargs)
    img_m_loaded = load_data(img_m, **kwargs)

    if isinstance(img_r_loaded, np.ndarray) and isinstance(
        img_m_loaded, np.ndarray
    ):  # If both images are numpy arrays
        # Check if images are of the same type and shape
        if img_r_loaded.dtype != img_m_loaded.dtype:  # If image types do not match
            raise ValueError("Image types do not match")
        if img_r_loaded.shape != img_m_loaded.shape:  # If image shapes do not match
            raise ValueError("Image shapes do not match")
    elif type(img_r_loaded) is not type(img_m_loaded):  # If image types do not match
        raise ValueError(
            "Image types do not match. img_r is of type {type(img_r_loaded)} and img_m "
            "is of type {type("
            "img_m_loaded)}"
        )
    elif isinstance(img_r, list) and isinstance(
        img_m, list
    ):  # If both images are lists or else
        if len(img_r_loaded) != len(img_m_loaded):  # If number of images do not match
            raise ValueError(
                "Number of images do not match. img_r has {len(img_r_loaded)} images "
                "and img_m has {len(img_m_loaded)} images"
            )
        for img_a, img_b in zip(
            img_r_loaded, img_m_loaded, strict=False
        ):  # For each image in the list
            if img_a.dtype != img_b.dtype:  # If image types do not match
                raise ValueError("Image types do not match")
            if img_a.dtype != img_b.shape:  # If image shapes do not match
                raise ValueError("Image shapes do not match")
    else:
        raise ValueError("Image format not supported.")

    if not isinstance(img_r_loaded, list):
        # Check if images are chromatic
        if chromatic is False and img_r_loaded.shape[-1] == 3:
            # Convert to grayscale as backup if falsely claimed to be non-chromatic
            warn("Images are chromatic. Converting to grayscale.")
            img_r_loaded = _to_grayscale(img_r_loaded)
            img_m_loaded = _to_grayscale(img_m_loaded)
        elif chromatic is True and img_r_loaded.shape[-1] != 3:
            raise ValueError("Images are not chromatic.")

    return img_r_loaded, img_m_loaded


def _to_grayscale(img):
    """Convert an image to grayscale."""
    img = _to_float(img)
    img_gray = ski.color.rgb2gray(img)
    img_gray[img_gray > 255] = 255
    return img_gray


def _rgb_to_yuv(img):
    """Convert an RGB image to YUV."""
    weights = np.array(
        [
            [0.2126, 0.7152, 0.0722],
            [-0.09991, -0.33609, 0.436],
            [0.615, -0.55861, -0.05639],
        ]
    )
    img_yuv = img @ weights
    return img_yuv


def _to_float(img, dtype=np.float64):
    """Convert a numpy array to float."""
    match img.dtype:
        case np.float32 | np.float64:
            return img
        case _:
            return img.astype(dtype)


def correlate_convolve_abs(
    img, kernel, mode="correlate", border_mode="constant", value=0
):
    """Correlates or convolves a numpy array with a kernel.

    Parameters
    ----------
    img : np.ndarray
        Input image
    kernel : np.ndarray
        Kernel
    mode : str, default='correlate'
        'correlate' or 'convolve'
    border_mode : str, default='constant'
        'constant', 'reflect', 'nearest', 'mirror' or 'wrap'

        .. seealso::
            See NumPy documentation for :py:func:`numpy.pad`.

    value : int, optional, default=0
        Value for constant border mode

    Returns
    -------
    res : np.ndarray
        Convolved result as numpy array

    Raises
    ------
    ValueError
        If ``border_mode`` is not supported \n
        If number of dimensions is not supported

    See Also
    --------
    :py:func:`scipy.signal.correlate`
    :py:func:`scipy.signal.convolve`

    Notes
    -----
    Correlates or convolves a numpy array with a kernel in the form
    :math:`\\operatorname{mean}(\\lvert \\pmb{I} \\cdot \\mathcal{K} \\rvert)` with
    :math:`\\pmb{I}` denoting the image and :math:`\\mathcal{K}` denoting the Kernel.
    Works in 2D and 3D.

    Examples
    --------
    >>> import numpy as np
    >>> from viqa import kernels
    >>> img = np.random.rand(128, 128)
    >>> kernel = kernels.sobel_kernel_2d_x()
    >>> res = correlate_convolve_abs(
    ...         img,
    ...         kernel,
    ...         mode="correlate",
    ...         border_mode="constant",
    ...         value=0
    ... )
    """
    if mode == "convolve":  # If mode is convolve
        kernel = np.flip(kernel)  # Flip kernel

    kernel_size = kernel.shape[0]  # Get kernel size
    ndim = len(img.shape)  # Get number of dimensions

    # Pad image
    match border_mode:
        case "constant":
            origin = np.pad(img, kernel_size, mode="constant", constant_values=value)
        case "reflect":
            origin = np.pad(img, kernel_size, mode="reflect")
        case "nearest":
            origin = np.pad(img, kernel_size, mode="edge")
        case "mirror":
            origin = np.pad(img, kernel_size, mode="symmetric")
        case "wrap":
            origin = np.pad(img, kernel_size, mode="wrap")
        case _:
            raise ValueError("Border mode not supported")

    # Correlate or convolve
    res = np.zeros(img.shape)  # Initialize result array
    for k in range(0, img.shape[0]):
        for m in range(0, img.shape[1]):
            # Check if 2D or 3D
            if ndim == 3:
                for n in range(0, img.shape[2]):
                    res[k, m, n] = np.mean(
                        abs(
                            kernel
                            * origin[
                                k : k + kernel_size,
                                m : m + kernel_size,
                                n : n + kernel_size,
                            ]
                        )
                    )
            elif ndim == 2:
                res[k, m] = np.mean(
                    abs(kernel * origin[k : k + kernel_size, m : m + kernel_size])
                )
            else:
                raise ValueError("Number of dimensions not supported")

    return res


def _extract_blocks(img, block_size, stride):
    """Extract blocks from an image.

    Parameters
    ----------
    img : np.ndarray
        Input image
    block_size : int
        Size of the block
    stride : int
        Stride

    Returns
    -------
    np.ndarray
        Numpy array of blocks
    """
    boxes = []
    m, n = img.shape
    for i in range(0, m - (block_size - 1), stride):
        for j in range(0, n - (block_size - 1), stride):
            boxes.append(img[i : i + block_size, j : j + block_size])
            # yield(img[i:i+block_size, j:j+block_size])  # TODO: change to generator
    return np.array(boxes)


def _fft(img):
    """Wrap scipy fft."""
    return fft.fftshift(fft.fftn(img))


def _ifft(fourier_img):
    """Wrap scipy ifft."""
    return fft.ifftn(fft.ifftshift(fourier_img))


def _is_even(num):
    """Check if a number is even."""
    return num % 2 == 0


def gabor_convolve(
    img,
    scales_num: int,
    orientations_num: int,
    min_wavelength=3,
    wavelength_scaling=3,
    bandwidth_param=0.55,
    d_theta_on_sigma=1.5,
):
    """Compute Log Gabor filter responses.

    Parameters
    ----------
    img : np.ndarray
        Image to be filtered
    scales_num : int
        Number of wavelet scales
    orientations_num : int
        Number of filter orientations
    min_wavelength : int, default=3
        Wavelength of smallest scale filter, maximum frequency is set by this value,
        should be >= 3
    wavelength_scaling : int, default=3
        Scaling factor between successive filters
    bandwidth_param : float, default=0.55
        Ratio of standard deviation of the Gaussian describing log Gabor filter's
        transfer function in the frequency domain to the filter's center frequency
        (0.74 for 1 octave bandwidth, 0.55 for 2 octave bandwidth, 0.41 for 3 octave
        bandwidth)
    d_theta_on_sigma : float, default=1.5
        Ratio of angular interval between filter orientations and standard deviation of
        angular Gaussian spreading function, a value of 1.5 results in approximately
        the minimum overlap needed to get even spectral coverage

    Returns
    -------
    np.ndarray
        Log Gabor filtered image

    Notes
    -----
    Even spectral coverage and independence of filter output are dependent on
    bandwidth_param vs wavelength_scaling.
    Some experimental values: \n
    0.85 <--> 1.3 \n
    0.74 <--> 1.6 (1 octave bandwidth) \n
    0.65 <--> 2.1 \n
    0.55 <--> 3.0 (2 octave bandwidth) \n
    Additionally d_theta_on_sigma should be set to 1.5 for approximately the minimum
    overlap needed to get even spectral coverage.

    For more information see [1]_. This code was originally written in Matlab by Peter
    Kovesi and adapted by Eric Larson. The adaption by Eric Larson is available under
    [2]_.

    References
    ----------
    .. [1] Kovesi, Peter.
        https://www.peterkovesi.com/matlabfns/PhaseCongruency/Docs/convexpl.html
    .. [2] Larson, E. C. (2008). http://vision.eng.shizuoka.ac.jp/mad
        (version 2011_10_07)
    .. [3] Field, D. J. (1987). Relations between the statistics of natural images and
        the response properties of cortical cells. Journal of The Optical Society of
        America A, 4(12), 2379–2394. https://doi.org/10.1364/JOSAA.4.002379

    Examples
    --------
    >>> import numpy as np
    >>> from viqa.utils import gabor_convolve
    >>> img = np.random.rand(128, 128)
    >>> res = gabor_convolve(img, scales_num=3, orientations_num=4)
    """
    # Authors
    # -------
    # Author: Peter Kovesi
    # Department of Computer Science & Software Engineering
    # The University of Western Australia
    # pk@cs.uwa.edu.au  https://peterkovesi.com/projects/
    #
    # Adaption: Eric Larson
    # Department of Electrical and Computer Engineering
    # Oklahoma State University, 2008
    # University Of Washington Seattle, 2009
    # Image Coding and Analysis lab
    #
    # Translation: Lukas Behammer
    # Research Center Wels
    # University of Applied Sciences Upper Austria
    # CT Research Group
    #
    # MODIFICATIONS
    # -------------
    # Original code, May 2001, Peter Kovesi
    # Altered, 2008, Eric Larson
    # Altered precomputations, 2011, Eric Larson
    # Translated to Python, 2024, Lukas Behammer
    #
    # LICENSE
    # -------
    # Copyright (c) 2001-2010 Peter Kovesi
    # www.peterkovesi.com
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in
    # all copies or substantial portions of the Software.
    #
    # The Software is provided "as is", without warranty of any kind.

    # Precomputing and assigning variables
    scales = np.arange(0, scales_num)
    orientations = np.arange(0, orientations_num)
    rows, cols = img.shape  # image dimensions
    # center of image
    col_c = math.floor(cols / 2)
    row_c = math.floor(rows / 2)

    # set up filter wavelengths from scales
    wavelengths = [
        min_wavelength * wavelength_scaling**scale_n for scale_n in range(0, scales_num)
    ]

    # convert image to frequency domain
    im_fft = fft.fftn(img)

    # compute matrices of same size as im with values ranging from -0.5 to 0.5 (-1.0 to
    # 1.0) for horizontal and vertical
    #   directions each
    if _is_even(cols):
        x_range = np.linspace(-cols / 2, (cols - 2) / 2, cols) / (cols / 2)
    else:
        x_range = np.linspace(-cols / 2, cols / 2, cols) / (cols / 2)
    if _is_even(rows):
        y_range = np.linspace(-rows / 2, (rows - 2) / 2, rows) / (rows / 2)
    else:
        y_range = np.linspace(-rows / 2, rows / 2, rows) / (rows / 2)
    x, y = np.meshgrid(x_range, y_range)

    # filters have radial component (frequency band) and an angular component
    # (orientation), those are multiplied to get the final filter

    # compute radial distance from center of matrix
    radius = np.sqrt(x**2 + y**2)
    radius[radius == 0] = 1  # avoid logarithm of zero

    # compute polar angle and its sine and cosine
    theta = np.arctan2(-y, x)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # compute standard deviation of angular Gaussian function
    theta_sigma = np.pi / orientations_num / d_theta_on_sigma

    # compute radial component
    radial_components = []
    for scale_n, _scale in enumerate(scales):  # for each scale
        center_freq = 1.0 / wavelengths[scale_n]  # center frequency of filter
        normalised_center_freq = center_freq / 0.5
        # log Gabor response for each frequency band (scale)
        log_gabor = np.exp(
            (np.log(radius) - np.log(normalised_center_freq)) ** 2
            / -(2 * np.log(bandwidth_param) ** 2)
        )
        log_gabor[row_c, col_c] = 0
        radial_components.append(log_gabor)

    # angular component and final filtering
    res = np.empty(
        (scales_num, orientations_num), dtype=object
    )  # precompute result array
    for orientation_n, _orientation in enumerate(orientations):  # for each orientation
        # compute angular component
        # Pre-compute filter data specific to this orientation
        # For each point in the filter matrix calculate the angular distance from the
        # specified filter orientation. To overcome the angular wrap-around problem
        # sine difference and cosine difference values are first computed and then
        # the atan2 function is used to determine angular distance.
        angle = orientation_n * np.pi / orientations_num  # filter angle
        diff_sin = sin_theta * np.cos(angle) - cos_theta * np.sin(
            angle
        )  # difference of sin
        diff_cos = cos_theta * np.cos(angle) + sin_theta * np.sin(
            angle
        )  # difference of cos
        angular_distance = abs(
            np.arctan2(diff_sin, diff_cos)
        )  # absolute angular distance
        spread = np.exp(
            (-(angular_distance**2)) / (2 * theta_sigma**2)
        )  # angular filter component

        # filtering
        for scale_n, _scale in enumerate(scales):  # for each scale
            # compute final filter
            filter_ = fft.fftshift(radial_components[scale_n] * spread)
            filter_[0, 0] = 0

            # apply filter
            res[scale_n, orientation_n] = fft.ifftn(im_fft * filter_)

    return res


def _check_chromatic(img_r, img_m, chromatic):
    """Permute image based on dimensions and chromaticity."""
    img_r = _to_float(img_r, np.float32)
    img_m = _to_float(img_m, np.float32)
    # check if chromatic
    if chromatic is False:
        if img_r.ndim == 3:
            # 3D images
            img_r_tensor = torch.tensor(img_r).unsqueeze(0).permute(3, 0, 1, 2)
            img_m_tensor = torch.tensor(img_m).unsqueeze(0).permute(3, 0, 1, 2)
        elif img_r.ndim == 2:
            # 2D images
            img_r_tensor = torch.tensor(img_r).unsqueeze(0).unsqueeze(0)
            img_m_tensor = torch.tensor(img_m).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Image format not supported.")
    else:
        img_r_tensor = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0)
        img_m_tensor = torch.tensor(img_m).permute(2, 0, 1).unsqueeze(0)
    return img_r_tensor, img_m_tensor


def export_results(metrics, output_path, filename):
    """Export data to a csv file.

    Parameters
    ----------
    metrics : list
        List of metrics
    output_path : str or os.PathLike
        Output path
    filename : str or os.PathLike
        Name of the file

    Notes
    -----
    This function just writes the ``score_val`` attribute of instanced metrics
    to a csv file. Therefore, the metrics must have been calculated before exporting and
    no-reference metrics cannot be distinguished between reference and modified image.

    .. attention::

        The csv file will be overwritten if it already exists.

    Examples
    --------
        .. doctest-skip::

            >>> from viqa import export_results, FSIM, PSNR
            >>> metric1 = FSIM()
            >>> metric2 = PSNR()
            >>> metrics = [metric1, metric2]
            >>> export_results(metrics, "path/to/output", "filename.csv")
    """
    # Check if filename has the correct extension
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    # Create file path
    file_path = os.path.join(output_path, filename)
    with open(file_path, mode="w", newline="") as f:  # Open file
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric in metrics:
            if metric.score_val is None:
                metric.score_val = "n/a"
            else:
                writer.writerow([metric._name, metric.score_val])


def _resize_image(img_r, img_m, scaling_order=1):
    # Resize image if shapes unequal
    if img_r.shape != img_m.shape:
        img_m = resize(img_m, img_r.shape, preserve_range=True, order=scaling_order)
        img_m = img_m.astype(img_r.dtype)
    return img_m


def export_metadata(metrics, metrics_parameters, file_path, file_name="metadata.txt"):
    """Export the metadata (custom parameters and package version) to a txt file.

    Parameters
    ----------
    metrics : list
        List of metric instances.
    metrics_parameters : list
        List of dictionaries containing the parameters for the metrics.
    file_path : str
        Path to the directory where the txt file should be saved.
    file_name : str, default='metadata.txt'
        Name of the txt file. Default is 'metadata.txt'.

    Notes
    -----
        .. attention::

            The txt file will be overwritten if it already exists.
    """
    if os.path.splitext(file_name)[1] != ".txt":
        raise ValueError(
            f"The file name {file_name} must have the " f"extension '.txt'."
        )
    path = os.path.join(file_path, file_name)
    with open(path, mode="w") as txtfile:
        txtfile.write("vIQA_version: " + version("viqa") + "\n")
        txtfile.write("Time: " + datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        txtfile.write("\n")
        txtfile.write("\n")
        txtfile.write("custom metric parameters: \n")
        txtfile.write("========================= \n")
        for metric_num, metric in enumerate(metrics):
            txtfile.write(metric.__str__().split("(")[0])
            txtfile.write("\n")
            [txtfile.write("-") for char in metric.__str__().split("(")[0]]
            txtfile.write("\n")
            txtfile.write("data_range: " + str(metric._parameters["data_range"]) + "\n")
            for key, value in metrics_parameters[metric_num].items():
                txtfile.write(key + ": " + str(value))
                txtfile.write("\n")
            txtfile.write("\n")


def export_image(
    results,
    img_r,
    img_m,
    x=None,
    y=None,
    z=None,
    file_path=None,
    file_name="image_comparison.png",
    show_image=True,
    **kwargs,
):
    """Print the reference and modified image side by side with the metric values.

    Parameters
    ----------
    results : dict
        Dictionary containing the metric values.
    img_r : str or np.ndarray
        Path to the reference image or the image itself.
    img_m : str or np.ndarray
        Path to the modified image or the image itself.
    x, y, z : int, optional
        The index of the slice to be plotted. Only one axis can be specified.
    file_path : str, optional
        Path to the directory where the image should be saved. If None, the image
        will be displayed only.
    file_name : str, optional
        Name of the image file. Default is 'image_comparison.png'.
    show_image : bool, optional
        If True, the image will be displayed. Default is True.
    kwargs : dict
        Additional parameters. Passed to :py:func:`matplotlib.pyplot.subplots`.

    Other Parameters
    ----------------
    dpi : int, default=300
        Dots per inch of the figure.
    scaling_order : int, default=1
        Order of the spline interpolation used for image resizing. Default is 1.
        Passed to :py:func:`skimage.transform.resize`

    Raises
    ------
    Exception
        If the area to be plotted was not correctly specified.
        If the image is not 2D or 3D.
        If no axis or more than one axis was specified.
    """
    if len(results) == 0:
        warn("No results to plot. Only the images are plotted.", UserWarning)

    dpi = kwargs.pop("dpi", 300)

    img_r = load_data(img_r)
    img_m = load_data(img_m)
    scaling_order = kwargs.pop("scaling_order", 1)
    img_m = _resize_image(img_r, img_m, scaling_order=scaling_order)
    img_r, img_m = _check_imgs(img_r, img_m)

    if img_r.ndim == 2 or (img_r.ndim == 3 and img_r.shape[-1] == 3):
        # For 2D (color) images, flip the image to match the imshow orientation
        img_r_plot = np.flip(img_r, 1)
        img_m_plot = np.flip(img_m, 1)
    elif img_r.ndim == 3:
        if {x, y, z} == {None}:
            raise Exception("One axis must be specified")
        if len({x, y, z} - {None}) != 1:
            raise Exception("Only one axis can be specified")

        # For 3D images, plot the specified area
        x_1, x_2 = 0, img_r.shape[0]
        y_1, y_2 = 0, img_r.shape[1]
        z_1, z_2 = 0, img_r.shape[2]

        if x is not None:
            img_r_plot = np.rot90(np.flip(img_r[x, y_1:y_2, z_1:z_2], 1))
            img_m_plot = np.rot90(np.flip(img_m[x, y_1:y_2, z_1:z_2], 1))
        elif y is not None:
            img_r_plot = np.rot90(np.flip(img_r[x_1:x_2, y, z_1:z_2], 1))
            img_m_plot = np.rot90(np.flip(img_m[x_1:x_2, y, z_1:z_2], 1))
        elif z is not None:
            img_r_plot = np.rot90(np.flip(img_r[x_1:x_2, y_1:y_2, z], 0), -1)
            img_m_plot = np.rot90(np.flip(img_m[x_1:x_2, y_1:y_2, z], 0), -1)
        else:
            raise Exception("Area to be plotted was not correctly specified")
    else:
        raise Exception("Image must be 2D or 3D")

    fig, axs = plt.subplots(1, 2, dpi=dpi, **kwargs)
    axs[0].imshow(img_r_plot, cmap="gray")
    axs[0].invert_yaxis()
    axs[1].imshow(img_m_plot, cmap="gray")
    axs[1].invert_yaxis()

    fig.suptitle("Image Comparison and IQA metric values", y=0.92)
    axs[0].set_title("Reference image")
    axs[1].set_title("Modified image")

    num_full_reference = 0
    num_no_reference = 0
    for metric in results.keys():
        if not (metric.endswith("_r") or metric.endswith("_m")):
            num_full_reference += 1
        else:
            num_no_reference += 1

    cols = ((num_full_reference - 1) // 4) + 1  # 4 metrics per column
    if num_no_reference != 0:
        cols += 1

    # Split the results into full-reference and no-reference metrics
    results_fr = {
        k: v for k, v in results.items() if not (k.endswith("_r") or k.endswith("_m"))
    }
    results_nr = {
        k: v for k, v in results.items() if k.endswith("_r") or k.endswith("_m")
    }

    # Plot full-reference metrics
    counter = 0
    for i in range(cols - 1):
        x_pos = 1.0 / (cols + 1)
        lines = 4  # 4 metrics per column
        x_pos = x_pos * (i + 1)
        for j in range(lines):
            if counter < num_full_reference:
                y_pos = 0.09 - 0.03 * j  # 0.09 is the top of the plot
                metric, result = list(results_fr.items())[counter]
                fig.text(
                    x_pos, y_pos, f"{metric}: {result:.2f}", ha="center", fontsize=8
                )
                counter += 1
    # Plot no-reference metrics
    x_pos = 1.0 / (cols + 1) * cols  # last column
    for j in range(num_no_reference):
        y_pos = 0.09 - 0.03 * j
        metric, result = list(results_nr.items())[j]
        fig.text(x_pos, y_pos, f"{metric}: {result:.2f}", ha="center", fontsize=8)

    axs[0].axis("off")
    axs[1].axis("off")

    # TODO: add check for file_name ending
    if file_path:
        file_path = os.path.join(file_path, file_name)
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0.5)
        if show_image:
            plt.show()
    else:
        plt.show()
