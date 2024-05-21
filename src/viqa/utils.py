"""Module for utility functions.

Examples
--------
    .. doctest-skip::

        >>> from viqa import load_data
        >>> img_path = "path/to/image.mhd"
        >>> img = load_data(img_path)

        >>> import numpy as np
        >>> from viqa import normalize_data
        >>> img = np.random.rand(128, 128)
        >>> img.dtype
        dtype('float64')
        >>> img = normalize_data(img, data_range_output=(0, 255))
        >>> img.dtype
        dtype('uint8')

        >>> from viqa import export_csv, PSNR, RMSE
        >>> metrics = [PSNR, RMSE]
        >>> output_path = "path/to/output"
        >>> filename = "metrics.csv"
        >>> export_csv(metrics, output_path, filename)

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
import glob
import math
import os
import re
from typing import Tuple
from warnings import warn

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.fft as fft
import skimage as ski
import torch
from torch import Tensor


def _load_data_from_disk(
    file_dir: str | os.PathLike, file_name: str | os.PathLike
) -> np.ndarray:
    """
    Load data from a file.

    Parameters
    ----------
    file_dir : str or os.PathLike
        Directory of the file
    file_name : str or os.PathLike
        Name of the file with extension

    Returns
    -------
    img_arr : np.ndarray
        Numpy array containing the data

    Raises
    ------
    ValueError
        If the file extension is not supported \n
        If the bit depth is not supported \n
        If no bit depth was found \n
        If no dimension was found
    """
    # Create file path components
    file_name_split = os.path.splitext(file_name)  # Split file name and extension
    file_ext = file_name_split[-1]
    file_path = os.path.join(file_dir, file_name)  # Complete file path

    # Check file extension
    if file_ext == ".mhd":  # If file is a .mhd file
        img_arr = load_mhd(file_dir, file_name)
        return img_arr
    elif file_ext == ".raw":  # If file is a .raw file
        img_arr = load_raw(file_dir, file_name)
        return img_arr
    elif file_ext == ".nii":
        img_arr = load_nifti(file_path)
        return img_arr
    elif file_ext == ".gz":
        if re.search('.nii', file_name):
            img_arr = load_nifti(file_path)
            return img_arr
        else:
            raise ValueError(
                "File extension not supported"
            )
    elif file_ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        img_arr = ski.io.imread(file_path)
        return img_arr
    else:
        raise ValueError(
            "File extension not supported"
        )  # Raise exception if file extension is not supported


def load_mhd(file_dir: str | os.PathLike, file_name: str | os.PathLike) -> np.ndarray:
    """
    Load data from a ``.mhd`` file.

    Parameters
    ----------
    file_dir : str or os.PathLike
        Directory of the file
    file_name : str or os.PathLike
        Name of the file with extension

    Returns
    -------
    img_arr : np.ndarray
        Numpy array containing the data

    Raises
    ------
    ValueError
        If the bit depth is not supported

    Examples
    --------
    >>> from viqa.utils import load_mhd  # doctest: +SKIP
    >>> img = load_raw("path/to/image.mhd")  # doctest: +SKIP
    """
    file_path = os.path.join(file_dir, file_name)  # Complete file path

    f = open(file=file_path, mode="rt")  # Open header file

    file_header_txt = f.read().split("\n")  # Extract header lines
    # Create dictionary from lines
    file_header = {
        key: value
        for line in file_header_txt[0:-1]
        for key, value in [line.split(" = ")]
    }

    data_file_path = os.path.join(
        file_dir, file_header["ElementDataFile"]
    )  # Get data file path from header

    # Extract dimension
    # Change DimSize to type int
    file_header.update(
        {"DimSize": [int(val) for val in file_header["DimSize"].split()]}
    )
    dim_size = file_header["DimSize"]  # Get DimSize from header

    # Check bit depth
    bit_depth = file_header["ElementType"]  # Get ElementType from header

    # Set data type according to bit depth
    if bit_depth == "MET_USHORT":
        data_type = np.ushort  # Set data type to unsigned short
    elif bit_depth == "MET_UCHAR":
        data_type = np.ubyte  # Set data type to unsigned byte
    elif bit_depth == "MET_FLOAT":
        data_type = np.float32  # Set data type to float32
    else:
        raise ValueError(
            "Bit depth not supported"
        )  # Raise exception if the bit depth is not supported

    img_arr = _load_binary(data_file_path, data_type, dim_size)
    return img_arr


def load_raw(file_dir: str | os.PathLike, file_name: str | os.PathLike) -> np.ndarray:
    """
    Load data from a ``.raw`` file.

    Parameters
    ----------
    file_dir : str or os.PathLike
        Directory of the file
    file_name : str or os.PathLike
        Name of the file with extension

    Returns
    -------
    img_arr : np.ndarray
        Numpy array containing the data

    Raises
    ------
    ValueError
        If the bit depth is not supported \n
        If no bit depth was found \n
        If no dimension was found

    Examples
    --------
    >>> from viqa.utils import load_raw  # doctest: +SKIP
    >>> img = load_raw("path/to/image.raw")  # doctest: +SKIP
    """
    # Create file path components
    file_name_split = os.path.splitext(file_name)  # Split file name and extension
    file_name_head = file_name_split[0]  # File name without extension

    # Check dimension
    dim_search_result = re.search(
        r"(\d+(x)\d+(x)\d+)", file_name_head
    )  # Search for dimension in file name
    if dim_search_result is not None:  # If dimension was found
        dim = dim_search_result.group(1)  # Get dimension from file name
    else:
        raise ValueError(
            "No dimension found"
        )  # Raise exception if no dimension was found

    # Extract dimension
    dim_size = re.split("x", dim)  # Split dimension string into list
    dim_size = [int(val) for val in dim_size]  # Change DimSize to type int

    # Check bit depth
    bit_depth_search_result = re.search(
        r"(\d{1,2}bit)", file_name_head
    )  # Search for the bit depth in file name
    if bit_depth_search_result is not None:  # If the bit depth was found
        bit_depth = bit_depth_search_result.group(
            1
        )  # Get the bit depth from file name
    else:
        raise ValueError(
            "No bit depth found"
        )  # Raise exception if no bit depth was found

    # Set data type according to bit depth
    if bit_depth == "16bit":
        data_type = np.ushort  # Set data type to unsigned short
    elif bit_depth == "8bit":
        data_type = np.ubyte  # Set data type to unsigned byte
    else:
        raise ValueError(
            "Bit depth not supported"
        )  # Raise exception if the bit depth is not supported

    data_file_path = os.path.join(file_dir, file_name)  # Get data file path

    img_arr = _load_binary(data_file_path, data_type, dim_size)
    return img_arr


def load_nifti(file_path: str | os.PathLike) -> np.ndarray:
    """
    Load data from a ``.nii`` file.

    Parameters
    ----------
    file_path : str or os.PathLike
        File path

    Returns
    -------
    img_arr : np.ndarray
        Numpy array containing the data

    Examples
    --------
    >>> from viqa.utils import load_nifti  # doctest: +SKIP
    >>> img = load_nifti("path/to/image.nii.gz")  # doctest: +SKIP

    Notes
    -----
    This function wraps the nibabel function ``nib.load``.
    """
    img = nib.load(file_path)
    img_arr = img.get_fdata()
    return img_arr


def _load_binary(data_file_path, data_type, dim_size):
    # Load data
    with open(file=data_file_path, mode="rb") as f:  # Open data file
        img_arr_orig = np.fromfile(
            file=f, dtype=data_type
        )  # Read data file into numpy array according to data type

    # Reshape numpy array according to DimSize
    img_arr = img_arr_orig.reshape(*dim_size[::-1])
    return img_arr


def load_data(
    img: np.ndarray | Tensor | str | os.PathLike,
    data_range: int | None = None,
    normalize: bool = False,
    batch: bool = False,
) -> list | np.ndarray:
    """
    Load data from a numpy array, a pytorch tensor or a file path.

    Parameters
    ----------
    img : np.ndarray, torch.Tensor, str or os.PathLike
        Numpy array, tensor or file path
    data_range : int, optional, default=None
        Maximum value of the returned data. Passed to
        :py:func:`.viqa.utils.normalize_data`.
    normalize : bool, default False
        If True, data is normalized to (0, ``data_range``) based on min and max of img.
    batch : bool, default False
        If True, img is a file path and all files in the directory are loaded.

        .. caution::
            Currently not tested.

    Returns
    -------
    img_arr : np.ndarray
        Numpy array containing the data

    Raises
    ------
    ValueError
        If input type is not supported \n
        If ``data_range=None`` and ``normalize=True``

    Warns
    -----
    RuntimeWarning
        If ``data_range`` is set but ``normalize=False``. ``data_range`` will be
        ignored.

    Warnings
    --------
    ``batch`` is currently not tested.

    Examples
    --------
        .. doctest-skip::

            >>> from viqa import load_data
            >>> path_r = "path/to/reference/image.mhd"
            >>> path_m = "path/to/modified/image.mhd"
            >>> img_r = load_data(path_r)
            >>> img_m = load_data(path_m)

    >>> from viqa import load_data
    >>> img_r = np.random.rand(128, 128)
    >>> img_r = load_data(img_r, data_range=255, normalize=True)
    """
    # exceptions and warning for data_range and normalize
    if normalize and data_range is None:
        raise ValueError("Parameter data_range must be set if normalize is True.")
    if not normalize and data_range is not None:
        warn(
            "Parameter data_range is set but normalize is False. Parameter "
            "data_range will be ignored.",
            RuntimeWarning,
        )

    img_arr: list[np.ndarray] | np.ndarray
    # Check input type
    match img:
        case str() | os.PathLike():  # If input is a file path
            # Check if batch
            if batch:
                # Get all files in directory
                files = glob.glob(img)  # type: ignore[type-var]
                img_arr = []  # Initialize list for numpy arrays
                # Load data from disk for each file
                for file in files:
                    img_arr.append(
                        _load_data_from_disk(
                            file_dir=os.path.dirname(file),
                            file_name=os.path.basename(file),
                        )
                    )
            else:
                file_dir = os.path.dirname(img)
                file_name = os.path.basename(img)
                img_arr = _load_data_from_disk(
                    file_dir, file_name
                )  # Load data from disk
        case np.ndarray():  # If input is a numpy array
            if not normalize:
                return img
            else:
                img_arr = img  # Use input as numpy array
        case Tensor():  # If input is a pytorch tensor
            img_arr = img.cpu().numpy()  # Convert tensor to numpy array
        case [np.ndarray()]:  # If input is a list
            img_arr = img  # Use input as list of numpy arrays
            batch = True  # Set batch to True to normalize list of numpy arrays
        case _:
            raise ValueError(
                "Input type not supported"
            )  # Raise exception if input type is not supported

    # Normalize data
    if normalize:
        if batch:
            img_arr = [
                normalize_data(img=img, data_range_output=(0, data_range))  # type: ignore[arg-type]
                for img in img_arr
            ]
        else:
            img_arr = normalize_data(img=img_arr, data_range_output=(0, data_range))  # type: ignore[arg-type]

    return img_arr


def _check_imgs(
    img_r: np.ndarray | Tensor | str | os.PathLike,
    img_m: np.ndarray | Tensor | str | os.PathLike,
    **kwargs,
) -> Tuple[list | np.ndarray, list | np.ndarray]:
    """Check if two images are of the same type and shape."""
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
    return img_r_loaded, img_m_loaded


def normalize_data(
        img: np.ndarray,
        data_range_output: Tuple[int, int],
        data_range_input: Tuple[int, int] = None,
        automatic_data_range: bool = True,
) -> np.ndarray:
    """Normalize a numpy array to a given data range.

    Parameters
    ----------
    img : np.ndarray
        Input image
    data_range_output : Tuple[int]
        Data range of the returned data
    data_range_input : Tuple[int], default=None
        Data range of the input data
    automatic_data_range : bool, default=True
        Automatically determine the input data range

    Returns
    -------
    img_arr : np.ndarray
        Input image normalized to data_range

    Raises
    ------
    ValueError
        If data type is not supported. \n
        If ``data_range`` is not supported.

    Warns
    -----
    RuntimeWarning
        If data is already normalized.

    Notes
    -----
    Currently only 8 bit int (0-255), 16 bit int (0-65535) and 32 bit float (0-1)
    data ranges are supported.

    Examples
    --------
    >>> import numpy as np
    >>> from viqa import normalize_data
    >>> img = np.random.rand(128, 128)
    >>> img_norm = normalize_data(
    >>>             img,
    >>>             data_range_output=(0, 255),
    >>>             automatic_data_range=True,
    >>> )
    >>> np.max(img_norm)
    255
    """
    # Check data type
    if np.issubdtype(img.dtype, np.integer):  # If data type is integer
        info = np.iinfo(img.dtype)  # type: ignore[assignment]
    elif np.issubdtype(img.dtype, np.floating):  # If data type is float
        info = np.finfo(img.dtype)  # type: ignore[assignment]
    else:
        raise ValueError("Data type not supported")

    # Check if data is already normalized
    if info.max is not data_range_output[1] or info.min is not data_range_output[0]:
        # Normalize data
        if automatic_data_range:
            img_min = np.min(img)  # Get minimum value of numpy array
            img_max = np.max(img)  # Get maximum value of numpy array
        else:
            img_min = data_range_input[0]
            img_max = data_range_input[1]
        # Normalize numpy array
        img = ((img - img_min) * (data_range_output[1] - data_range_output[0])
               / (img_max - img_min)) + data_range_output[0]

        # Change data type
        # If data range is 255 (8 bit)
        if data_range_output[1] == 2**8 - 1 and data_range_output[0] == 0:
            img = img.astype(np.uint8)  # Change data type to unsigned byte
        # If data range is 65535 (16 bit)
        elif data_range_output[1] == 2**16 - 1 and data_range_output[0] == 0:
            img = img.astype(np.uint16)  # Change data type to unsigned short
        # If data range is 1
        elif data_range_output[1] == 1 and data_range_output[0] == 0:
            img = img.astype(np.float32)  # Change data type to float32
        else:
            raise ValueError("Data range not supported. Please use (0, 1), (0, 255) or "
                             "(0, 65535) as data_range_output.")
    else:
        warn("Data is already normalized.", RuntimeWarning)

    return img


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

        .. seealso::
            Scipy documentation

    border_mode : str, default='constant'
        'constant', 'reflect', 'nearest', 'mirror' or 'wrap'

        .. seealso::
            Scipy documentation

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
    `scipy.signal.correlate
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html>`_
    and `scipy.signal.convolve
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve.html>`_

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
        else:
            # 2D images
            img_r_tensor = torch.tensor(img_r).unsqueeze(0).unsqueeze(0)
            img_m_tensor = torch.tensor(img_m).unsqueeze(0).unsqueeze(0)
    else:
        img_r_tensor = torch.tensor(img_r).permute(2, 0, 1).unsqueeze(0)
        img_m_tensor = torch.tensor(img_m).permute(2, 0, 1).unsqueeze(0)
    return img_r_tensor, img_m_tensor


def _visualize_cnr_2d(img, signal_center, background_center, radius):
    fig, axs = plt.subplots(2, 1, figsize=(6, 12), dpi=300)
    fig.suptitle("Regions for CNR Calculation",
                 y=0.92)
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Background")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            background_center[1] - radius,
            background_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0].add_patch(rect_1)

    axs[1].imshow(img, cmap="gray")
    axs[1].set_title("Signal")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1].add_patch(rect_1)
    plt.show()


def _visualize_cnr_3d(img, signal_center, background_center, radius):
    fig, axs = plt.subplots(2, 3, figsize=(14, 10), dpi=300)
    fig.suptitle("Background (Upper) and Signal Region (Lower) for CNR Calculation",
                 y=0.92)
    axs[0][0].imshow(img[background_center[0], ::-1, :],
                     cmap="gray")
    axs[0][0].set_title(f"z-axis, slice {background_center[0]}")
    axs[0][0].set_xlabel("y")
    axs[0][0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            background_center[1] - radius,
            background_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0][0].axhline(
        y=background_center[2], color="g", linestyle="--"
    )
    axs[0][0].axvline(
        x=background_center[1], color="g", linestyle="--"
    )
    axs[0][0].add_patch(rect_1)

    axs[0][1].imshow(img[::-1, background_center[1], :],
                     cmap="gray")
    axs[0][1].set_title(f"y-axis, slice {background_center[1]}")
    axs[0][1].set_xlabel("x")
    axs[0][1].set_ylabel("z")
    rect_2 = patches.Rectangle(
        (
            background_center[2] - radius,
            background_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0][1].axhline(
        y=background_center[0], color="g", linestyle="--"
    )
    axs[0][1].axvline(
        x=background_center[2], color="g", linestyle="--"
    )
    axs[0][1].add_patch(rect_2)

    axs[0][2].imshow(img[::-1, :, background_center[2]],
                     cmap="gray")
    axs[0][2].set_title(f"x-axis, slice {background_center[2]}")
    axs[0][2].set_xlabel("y")
    axs[0][2].set_ylabel("z")
    rect_3 = patches.Rectangle(
        (
            background_center[1] - radius,
            background_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    axs[0][2].axhline(
        y=background_center[0], color="g", linestyle="--"
    )
    axs[0][2].axvline(
        x=background_center[1], color="g", linestyle="--"
    )
    axs[0][2].add_patch(rect_3)

    axs[1][0].imshow(img[signal_center[0], ::-1, :], cmap="gray")
    axs[1][0].set_title(f"z-axis, slice {signal_center[0]}")
    axs[1][0].set_xlabel("y")
    axs[1][0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1][0].axhline(y=signal_center[2], color="g",
                      linestyle="--")
    axs[1][0].axvline(x=signal_center[1], color="g",
                      linestyle="--")
    axs[1][0].add_patch(rect_1)

    axs[1][1].imshow(img[::-1, signal_center[1], :], cmap="gray")
    axs[1][1].set_title(f"y-axis, slice {signal_center[1]}")
    axs[1][1].set_xlabel("x")
    axs[1][1].set_ylabel("z")
    rect_2 = patches.Rectangle(
        (
            signal_center[2] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1][1].axhline(y=signal_center[0], color="g",
                      linestyle="--")
    axs[1][1].axvline(x=signal_center[2], color="g",
                      linestyle="--")
    axs[1][1].add_patch(rect_2)

    axs[1][2].imshow(img[::-1, :, signal_center[2]], cmap="gray")
    axs[1][2].set_title(f"x-axis, slice {signal_center[2]}")
    axs[1][2].set_xlabel("y")
    axs[1][2].set_ylabel("z")
    rect_3 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1][2].axhline(y=signal_center[0], color="g",
                      linestyle="--")
    axs[1][2].axvline(x=signal_center[1], color="g",
                      linestyle="--")
    axs[1][2].add_patch(rect_3)
    plt.show()


def _visualize_snr_2d(img, signal_center, radius):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    fig.suptitle("Signal Region for SNR Calculation", y=0.92)

    ax.imshow(img, cmap="gray")
    ax.set_xlabel("y")
    ax.set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[0] - radius,
            signal_center[1] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    ax.add_patch(rect_1)
    plt.show()


def _visualize_snr_3d(img, signal_center, radius):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6), dpi=300)
    fig.suptitle("Signal Region for SNR Calculation", y=0.92)

    axs[0].imshow(img[signal_center[0], ::-1, :],
                  cmap="gray")
    axs[0].set_title(f"z-axis, slice {signal_center[0]}")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("x")
    rect_1 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[2] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[0].axhline(y=signal_center[2], color="g",
                   linestyle="--")
    axs[0].axvline(x=signal_center[1], color="g",
                   linestyle="--")
    axs[0].add_patch(rect_1)

    axs[1].imshow(img[::-1, signal_center[1], :],
                  cmap="gray")
    axs[1].set_title(f"y-axis, slice {signal_center[1]}")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("z")
    rect_2 = patches.Rectangle(
        (
            signal_center[2] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[1].axhline(y=signal_center[0], color="g",
                   linestyle="--")
    axs[1].axvline(x=signal_center[2], color="g",
                   linestyle="--")
    axs[1].add_patch(rect_2)

    axs[2].imshow(img[::-1, :, signal_center[2]],
                  cmap="gray")
    axs[2].set_title(f"x-axis, slice {signal_center[2]}")
    axs[2].set_xlabel("y")
    axs[2].set_ylabel("z")
    rect_3 = patches.Rectangle(
        (
            signal_center[1] - radius,
            signal_center[0] - radius,
        ),
        radius * 2,
        radius * 2,
        linewidth=1,
        edgecolor="b",
        facecolor="none",
    )
    axs[2].axhline(y=signal_center[0], color="g",
                   linestyle="--")
    axs[2].axvline(x=signal_center[1], color="g",
                   linestyle="--")
    axs[2].add_patch(rect_3)
    plt.show()


def export_csv(metrics, output_path, filename):
    """Export data to a csv file.

    Parameters
    ----------
    metrics : list
        List of metrics
    output_path : str or os.PathLike
        Output path
    filename : str or os.PathLike
        Name of the file

    Examples
    --------
        .. doctest-skip::

            >>> from viqa import export_csv, FSIM, PSNR
            >>> metric1 = FSIM()
            >>> metric2 = PSNR()
            >>> metrics = [metric1, metric2]
            >>> export_csv(metrics, "path/to/output", "filename.csv")
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
