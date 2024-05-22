"""Module for utility functions for data loading.

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

import glob
import os
import re
from typing import Tuple
from warnings import warn

import nibabel as nib
import numpy as np
import skimage as ski
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
    >>> from viqa.load_utils import load_mhd  # doctest: +SKIP
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
    >>> from viqa.load_utils import load_raw  # doctest: +SKIP
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
    >>> from viqa.load_utils import load_nifti  # doctest: +SKIP
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
