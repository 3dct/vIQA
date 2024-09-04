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

import csv
import glob
import os
import re
from typing import Tuple, Union
from warnings import warn

import nibabel as nib
import numpy as np
import skimage as ski
from scipy.stats import kurtosis, skew
from torch import Tensor
from tqdm.autonotebook import tqdm

from viqa.visualization_utils import visualize_2d, visualize_3d


class ImageArray(np.ndarray):
    """
    Class for image arrays.

    This class is a subclass of :py:class:`numpy.ndarray` and adds attributes for image
    statistics.

    Attributes
    ----------
    mean_value : float
        Mean of the image array
    median : float
        Median of the image array
    variance : float
        Variance of the image array
    standarddev : float
        Standard deviation of the image array
    skewness : float
        Skewness of the image array
    kurtosis : float
        Kurtosis of the image array
    histogram : tuple
        Histogram of the image array
    minimum : float
        Minimum value of the image array
    maximum : float
        Maximum value of the image array

    Parameters
    ----------
    input_array : np.ndarray
        Numpy array containing the data

    Returns
    -------
    obj : ImageArray
        New instance of the ImageArray class
    """

    def __new__(cls, input_array):
        """
        Create a new instance of the ImageArray class.

        Parameters
        ----------
        input_array : np.ndarray
            Numpy array containing the data

        Returns
        -------
        obj : ImageArray
            New instance of the ImageArray class
        """
        # Input array is an already formed ndarray instance
        obj = np.asarray(input_array).view(cls)
        # FIXME: Only calculate statistics on method call
        # Add attributes
        obj.mean_value = np.mean(input_array)
        obj.median = np.median(input_array)
        obj.variance = np.var(input_array)
        obj.standarddev = np.std(input_array)
        obj.skewness = skew(input_array, axis=None)
        obj.kurtosis = kurtosis(input_array, axis=None)
        if input_array.dtype.kind in ["u", "i"]:
            obj.histogram = np.histogram(
                input_array, bins=np.iinfo(input_array.dtype).max
            )
        else:
            obj.histogram = np.histogram(input_array, bins=255)
        obj.minimum = np.min(input_array)
        obj.maximum = np.max(input_array)
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize the array.

        Parameters
        ----------
        obj : object
            Object to finalize
        """
        if obj is None:
            return
        # Add attributes
        self.mean_value = getattr(obj, "mean_value", None)
        self.median = getattr(obj, "median", None)
        self.variance = getattr(obj, "variance", None)
        self.standarddev = getattr(obj, "standarddev", None)
        self.skewness = getattr(obj, "skewness", None)
        self.kurtosis = getattr(obj, "kurtosis", None)
        self.histogram = getattr(obj, "histogram", None)
        self.minimum = getattr(obj, "minimum", None)
        self.maximum = getattr(obj, "maximum", None)

    def describe(
        self,
        path: Union[str, os.PathLike, None] = None,
        filename: Union[str, None] = None,
    ) -> dict:
        """
        Export image statistics to a csv file.

        Parameters
        ----------
        path : str or os.PathLike, optional
            Path to the directory where the csv file should be saved
        filename : str, optional
            Name of the csv file

        Returns
        -------
        stats : dict
            Dictionary containing the image statistics

        Warns
        -----
        RuntimeWarning
            If no path or filename is provided. Statistics are not exported.

        Examples
        --------
        >>> import numpy as np
        >>> from viqa import ImageArray
        >>> img = np.random.rand(128, 128)
        >>> img = ImageArray(img)
        >>> img.describe(path="path/to", filename="image_statistics")
        """
        stats = {
            "mean": self.mean_value,
            "median": self.median,
            "variance": self.variance,
            "standarddev": self.standarddev,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }
        if path and filename:
            # export to csv
            if not filename.lower().endswith(".csv"):
                filename += ".csv"
            # Create file path
            file_path = os.path.join(path, filename)
            with open(file_path, mode="w", newline="") as f:  # Open file
                writer = csv.DictWriter(f, stats.keys())
                writer.writeheader()
                writer.writerow(stats)
            print(f"Statistics exported to {file_path}")
        else:
            warn(
                "No path or filename provided. Statistics not exported.", RuntimeWarning
            )
        return stats

    def visualize(
        self, slices: Tuple[int, int, int], export_path=None, **kwargs
    ) -> None:
        """
        Visualize the image array.

        If export_path is provided, the visualization is saved to the directory.

        Parameters
        ----------
        slices : Tuple[int, int, int]
            Slices for the x, y and z axis
        export_path : str or os.PathLike, optional
            Path to the directory where the visualization should be saved
        **kwargs : dict
            Additional keyword arguments for visualization. See
            :py:func:`.viqa.visualization_utils.visualize_3d`.

        Raises
        ------
        ValueError
            If the image is not 2D or 3D.

        Warns
        -----
        UserWarning
            If the image is 2D, the parameter slices will be ignored.

        Examples
        --------
        >>> import numpy as np
        >>> from viqa import ImageArray
        >>> img = np.random.rand(128, 128, 128)
        >>> img = ImageArray(img)
        >>> img.visualize(slices=(64, 64, 64))
        """
        if self.ndim == 3:
            visualize_3d(self, slices, export_path, **kwargs)
        elif self.ndim == 2:
            warn("Image is 2D. Parameter slices will be ignored.", RuntimeWarning)
            visualize_2d(self, export_path, **kwargs)
        else:
            raise ValueError("Image must be 2D or 3D.")


def _load_data_from_disk(file_dir: str | os.PathLike, file_name: str) -> ImageArray:
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
    img_arr : ImageArray
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
        return ImageArray(img_arr)
    elif file_ext == ".raw":  # If file is a .raw file
        img_arr = load_raw(file_dir, file_name)
        return ImageArray(img_arr)
    elif file_ext == ".nii":
        img_arr = load_nifti(file_path)
        return ImageArray(img_arr)
    elif file_ext == ".gz":
        if re.search(".nii", file_name):
            img_arr = load_nifti(file_path)
            return ImageArray(img_arr)
        else:
            raise ValueError("File extension not supported")
    elif file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        img_arr = ski.io.imread(file_path)
        return ImageArray(img_arr)
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

    f = open(file=file_path)  # Open header file

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
        {"DimSize": [int(val) for val in file_header["DimSize"].split()]}  # type: ignore # TODO
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
        data_type = np.float32  # type: ignore # Set data type to float32 # TODO
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
        If the bit depth is not supported. \n
        If no bit depth was found. \n
        If no dimension was found.

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
        bit_depth = bit_depth_search_result.group(1)  # Get the bit depth from file name
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
    This function wraps the nibabel function :py:func:`nibabel.loadsave.load`.
    """
    img = nib.load(file_path)
    img_arr = img.get_fdata()  # type: ignore[attr-defined]
    return img_arr


def _load_binary(data_file_path, data_type, dim_size):
    # Load data
    with open(file=data_file_path, mode="rb") as f:  # Open data file
        img_arr_orig = np.fromfile(
            file=f, dtype=data_type
        )  # Read data file into numpy array according to data type

    if img_arr_orig.size != np.prod(dim_size):
        raise ValueError(
            "Size of data file ("
            + data_file_path
            + ") does not match dimensions ("
            + str(dim_size)
            + ")"
        )
    # Reshape numpy array according to DimSize
    img_arr = img_arr_orig.reshape(*dim_size[::-1])
    # Rotate and flip image
    img_arr = np.rot90(img_arr, axes=(0, 2))
    img_arr = np.flip(img_arr, 0)
    return img_arr


def load_data(
    img: np.ndarray | ImageArray | Tensor | str | os.PathLike,
    data_range: int | None = None,
    normalize: bool = False,
    batch: bool = False,
    roi: Union[list[Tuple[int, int]], None] = None,
) -> list[ImageArray] | ImageArray:
    """
    Load data from a numpy array, a pytorch tensor or a file path.

    Parameters
    ----------
    img : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
        Numpy array, ImageArray, tensor or file path
    data_range : int, optional, default=None
        Maximum value of the returned data. Passed to
        :py:func:`viqa.load_utils.normalize_data`.
    normalize : bool, default False
        If True, data is normalized to (0, ``data_range``) based on min and max of img.
    batch : bool, default False
        If True, img is a file path and all files in the directory are loaded.

        .. caution::
            Currently not tested.

        .. todo::
            Deprecate batch loading as this has no use with the current implementation
            as BatchMetrics class.

    roi : list[Tuple[int, int]], optional, default=None
        Region of interest for cropping the image. The format is a list of tuples
        with the ranges for the x, y and z axis. If not set, the whole image is loaded.

    Returns
    -------
    img_arr : ImageArray or list[ImageArray]
        :py:class:`viqa.load_utils.ImageArray` or list of
        :py:class:`viqa.load_utils.ImageArray` containing the data

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
                for file in tqdm(files):
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
        case ImageArray():  # If input is an ImageArray
            img_arr = img  # Use input as ImageArray
        case np.ndarray():  # If input is a numpy array
            img_arr = img  # Use input as numpy array
        case Tensor():  # If input is a pytorch tensor
            img_arr = img.cpu().numpy()  # Convert tensor to numpy array
        case [np.ndarray()]:  # If input is a list
            # FIXME: This should never get called as the input should not be a list
            #  according to the type hint.
            #  Either add support for list input (and add case [ImageArray()]) or remove
            #  this case.
            img_arr = img  # Use input as list of ImageArrays
            batch = True  # Set batch to True to normalize list of numpy arrays
        case _:
            raise ValueError(
                "Input type not supported"
            )  # Raise exception if input type is not supported

    # Normalize data
    if normalize and data_range:
        if batch:
            img_arr = [
                normalize_data(img=img, data_range_output=(0, data_range))
                for img in img_arr
            ]
        elif not isinstance(img_arr, list):
            img_arr = normalize_data(img=img_arr, data_range_output=(0, data_range))

    if roi:
        # Crop image
        if batch:
            img_arr = [crop_image(img, *roi) for img in img_arr]
        elif not isinstance(img_arr, list):
            img_arr = crop_image(img_arr, *roi)

    img_final: list[ImageArray] | ImageArray
    if isinstance(img_arr, ImageArray):
        img_final = img_arr
    elif isinstance(img_arr, list):
        img_final = [ImageArray(img) for img in img_arr]
    else:
        img_final = ImageArray(img_arr)

    return img_final


def normalize_data(
    img: np.ndarray | ImageArray,
    data_range_output: Tuple[int, int],
    data_range_input: Union[Tuple[int, int], None] = None,
    automatic_data_range: bool = True,
) -> np.ndarray | ImageArray:
    """Normalize a numpy array to a given data range.

    Parameters
    ----------
    img : np.ndarray or ImageArray
        Input image
    data_range_output : Tuple[int]
        Data range of the returned data
    data_range_input : Tuple[int], default=None
        Data range of the input data
    automatic_data_range : bool, default=True
        Automatically determine the input data range

    Returns
    -------
    img_arr : np.ndarray or ImageArray
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
            img_min = data_range_input[0]  # type: ignore # TODO
            img_max = data_range_input[1]  # type: ignore # TODO
        # Normalize numpy array
        img = (
            (img - img_min)
            * (data_range_output[1] - data_range_output[0])
            / (img_max - img_min)
        ) + data_range_output[0]

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
            raise ValueError(
                "Data range not supported. Please use (0, 1), (0, 255) or "
                "(0, 65535) as data_range_output."
            )
    else:
        warn("Data is already normalized.", RuntimeWarning)

    return img


def crop_image(
    img: np.ndarray | ImageArray,
    x: Tuple[int, int],
    y: Tuple[int, int],
    z: Union[Tuple[int, int], None],
) -> np.ndarray | ImageArray:
    """
    Crop the image array.

    Parameters
    ----------
    img : np.ndarray or ImageArray
        Input image
    x : Tuple[int, int]
        Range for the x-axis
    y : Tuple[int, int]
        Range for the y-axis
    z : Tuple[int, int]
        Range for the z-axis

    Returns
    -------
    img_crop : np.ndarray or ImageArray
        Cropped image array

    Raises
    ------
    ValueError
        If the image is not 2D or 3D.

    Warns
    -----
    RuntimeWarning
        If the image is 2D, the parameter z will be ignored.
    """
    if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 3):
        if z is not None:
            warn("Image is 2D. Parameter z will be ignored.", RuntimeWarning)
        img_crop = img[x[0] : x[1], y[0] : y[1]]
    elif img.ndim == 3 and z is not None:
        img_crop = img[x[0] : x[1], y[0] : y[1], z[0] : z[1]]
    else:
        raise ValueError("Image must be 2D or 3D.")
    return img_crop
