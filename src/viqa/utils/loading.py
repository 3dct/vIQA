"""Module for utility functions for data loading.

Examples
--------
    .. doctest-skip::

        >>> from viqa import load_data
        >>> img_path = "path/to/image.mhd"
        >>> img = load_data(img_path)

        >>> import numpy as np
        >>> from viqa import ImageArray, crop_image, normalize_data
        >>> img = np.random.rand(128, 128)
        >>> img.dtype
        dtype('float64')
        >>> type(img)
        <class 'numpy.ndarray'>
        >>> img = ImageArray(img)
        >>> type(img)
        <class 'viqa.utils.loading.ImageArray'>
        >>> img = normalize_data(img, data_range_output=(0, 255))
        >>> img.dtype
        dtype('uint8')
        >>> img = crop_image(img, (0, 64), (0, 64))
        >>> img.shape
        (64, 64)
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
from typing import Any, Tuple, Union
from warnings import warn

import nibabel as nib
import numpy as np
import skimage as ski
from scipy.stats import kurtosis, skew
from torch import Tensor
from tqdm.autonotebook import tqdm

from .deprecation import RemovedInFutureVersionWarning
from .misc import _to_grayscale
from .visualization import visualize_2d, visualize_3d


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
        self.mean_value = getattr(
            obj, "mean_value", "Not set. Run calculate_statistics() first."
        )
        self.median = getattr(
            obj, "median", "Not set. Run calculate_statistics() first."
        )
        self.variance = getattr(
            obj, "variance", "Not set. Run calculate_statistics() first."
        )
        self.standarddev = getattr(
            obj, "standarddev", "Not set. Run calculate_statistics() first."
        )
        self.skewness = getattr(
            obj, "skewness", "Not set. Run calculate_statistics() first."
        )
        self.kurtosis = getattr(
            obj, "kurtosis", "Not set. Run calculate_statistics() first."
        )
        self.histogram = getattr(
            obj, "histogram", "Not set. Run calculate_statistics() first."
        )
        self.minimum = getattr(
            obj, "minimum", "Not set. Run calculate_statistics() first."
        )
        self.maximum = getattr(
            obj, "maximum", "Not set. Run calculate_statistics() first."
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Make sure that an ImageArray is returned when a ufunc operation is performed
        on the ImageArray class.
        """
        # Adapted code by @Thawn from
        # https://stackoverflow.com/questions/51520630/subclassing-numpy-array-propagate-attributes

        # convert inputs and outputs of class ImageArray to np.ndarray to prevent
        # infinite recursion
        args = (
            (i.view(np.ndarray) if isinstance(i, ImageArray) else i) for i in inputs
        )
        outputs = kwargs.pop("out", [])
        if outputs:
            kwargs["out"] = tuple(
                (o.view(np.ndarray) if isinstance(o, ImageArray) else o)
                for o in outputs
            )
        else:
            outputs = (None,) * ufunc.nout
        # call numpys implementation of __array_ufunc__
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented
        if method == "at":
            # method == 'at' means that the operation is performed in-place. Therefore,
            # we are done.
            return
        # now we need to make sure that outputs that where specified with the 'out'
        # argument are handled correctly:
        if ufunc.nout == 1:
            results = (results,)
        results = tuple(
            (result.view(ImageArray) if output is None else output)
            for result, output in zip(results, outputs, strict=False)
        )
        return results[0] if len(results) == 1 else results

    def calculate_statistics(self):
        """Calculate statistics of the image array.

        .. admonition:: The following statistics are calculated:

            * mean
            * median
            * variance
            * standard deviation
            * skewness
            * kurtosis
            * histogram
            * minimum
            * maximum
        """
        # Add attributes
        self.mean_value = np.mean(self)
        self.median = np.median(self.view())
        self.variance = np.var(self.view())
        self.standarddev = np.std(self.view())
        self.skewness = skew(self.view(), axis=None)
        self.kurtosis = kurtosis(self.view(), axis=None)
        if self.view().dtype.kind in ["u", "i"]:
            self.view().histogram = np.histogram(
                self.view(), bins=np.iinfo(self.view().dtype).max
            )
        else:
            self.histogram = np.histogram(self.view(), bins=255)
        self.minimum = np.min(self.view())
        self.maximum = np.max(self.view())

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
            :py:func:`.viqa.utils.visualize_3d`.

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
    >>> from viqa.utils import load_mhd  # doctest: +SKIP
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
    # Get DimSize from header and change to type int
    dim_size = [int(val) for val in file_header["DimSize"].split()]

    # Check bit depth
    bit_depth = file_header["ElementType"]  # Get ElementType from header

    data_type: type[Union[np.floating[Any] | np.integer[Any] | np.unsignedinteger[Any]]]
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
        If the bit depth is not supported. \n
        If no bit depth was found. \n
        If no dimension was found.

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
    >>> from viqa.utils import load_nifti  # doctest: +SKIP
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
        :py:func:`viqa.utils.normalize_data`.
    normalize : bool, default False
        If True, data is normalized to (0, ``data_range``) based on min and max of img.
    batch : bool, default False
        If True, img is a file path and all files in the directory are loaded.

        .. deprecated:: 4.0.0
            This will be deprecated in version 4.0.0.

        .. todo:: Deprecate in version 4.0.0

    roi : list[Tuple[int, int]], optional, default=None
        Region of interest for cropping the image. The format is a list of tuples
        with the ranges for the x, y (and z) axis. First value in the tuple denotes the
        start and the second value the end of the range. If not set, the whole image is
        loaded.

    Returns
    -------
    img_arr : ImageArray or list[ImageArray]
        :py:class:`viqa.utils.ImageArray` or list of
        :py:class:`viqa.utils.ImageArray` containing the data

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
    ``batch`` will be deprecated in version 4.0.0.

    .. todo:: Deprecate in version 4.0.0

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
    # TODO: Deprecate in version 4.0.0
    if batch:
        raise RemovedInFutureVersionWarning(
            "Batch loading is deprecated and will be removed in vIQA 4.0.x."
        )

    # exceptions and warning for data_range and normalize
    if normalize and data_range is None:
        raise ValueError("Parameter data_range must be set if normalize is True.")
    if not normalize and data_range is not None:
        warn(
            "Parameter data_range is set but normalize is False. Parameter "
            "data_range will be ignored.",
            RuntimeWarning,
        )

    img_arr: list[np.ndarray] | np.ndarray  # TODO: list can be removed in version 4.0.0
    # Check input type
    match img:
        case str() | os.PathLike():  # If input is a file path
            # Check if batch
            if batch:
                # TODO: Deprecate in version 4.0.0
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
            # TODO: Deprecate in version 4.0.0
            img_arr = img  # Use input as list of ImageArrays
            batch = True  # Set batch to True to normalize list of numpy arrays
        case _:
            raise ValueError(
                "Input type not supported"
            )  # Raise exception if input type is not supported

    # Normalize data
    if normalize and data_range:
        # TODO: Deprecate in version 4.0.0
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
            # TODO: Deprecate in version 4.0.0
            img_arr = [crop_image(img, *roi) for img in img_arr]
        elif not isinstance(img_arr, list):
            img_arr = crop_image(img_arr, *roi)

    img_final: list[ImageArray] | ImageArray
    if isinstance(img_arr, ImageArray):
        img_final = img_arr
    elif isinstance(img_arr, list):
        # TODO: Deprecate in version 4.0.0
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
    """Normalize an image to a given data range.

    Parameters
    ----------
    img : np.ndarray or ImageArray
        Input image.
    data_range_output : Tuple[int]
        Data range of the returned data.
    data_range_input : Tuple[int], default=None
        Data range of the input data. Needs to be set if ``automatic_data_range`` is
        False.
    automatic_data_range : bool, default=True
        Automatically determine the input data range.

    Returns
    -------
    img_arr : np.ndarray or ImageArray
        Input image normalized to data_range.

    Raises
    ------
    ValueError
        If data type is not supported.
        If ``data_range_output`` is not supported.
        If ``automatic_data_range`` is False and ``data_range_input`` is not set.

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
    info: Union[np.iinfo, np.finfo]
    # Check data type
    if np.issubdtype(img.dtype, np.integer):  # If data type is integer
        info = np.iinfo(img.dtype)
    elif np.issubdtype(img.dtype, np.floating):  # If data type is float
        info = np.finfo(img.dtype)
    else:
        raise ValueError("Data type not supported")

    # Check if data is already normalized
    if info.max is not data_range_output[1] or info.min is not data_range_output[0]:
        # Normalize data
        if automatic_data_range:
            img_min = np.min(img)  # Get minimum value of numpy array
            img_max = np.max(img)  # Get maximum value of numpy array
        else:
            if data_range_input is None:
                raise ValueError(
                    "If automatic_data_range is False, data_range_input must be set."
                )
            else:
                img_min = data_range_input[0]
                img_max = data_range_input[1]
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
    z : Tuple[int, int] or None
        Range for the z-axis

    Returns
    -------
    img_crop : np.ndarray or ImageArray
        Cropped image array

    Raises
    ------
    ValueError
        If the image is not 2D or 3D.
        If the cropped image shape is larger than the original image shape.

    Warns
    -----
    RuntimeWarning
        If the image is 2D, the parameter z will be ignored.
    """
    # Get original shape to check if image is already cropped
    img_shape = np.array(img.shape)

    if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 3):  # If image is 2D
        crop_shape = np.array((x[1] - x[0], y[1] - y[0]))
        if (crop_shape < img_shape).all():  # If cropping is smaller than original image
            if z is not None:
                warn("Image is 2D. Parameter z will be ignored.", RuntimeWarning)
            img_crop = img[x[0] : x[1], y[0] : y[1]]
        elif crop_shape == img_shape:  # If image is already cropped
            warn("Image is already cropped.", RuntimeWarning)
            img_crop = img
        else:  # If cropping is larger than original image
            raise ValueError("Cropped image shape must be smaller than original image.")
    elif img.ndim == 3 and z is not None:  # If image is 3D
        crop_shape = np.array((x[1] - x[0], y[1] - y[0], z[1] - z[0]))
        if (crop_shape < img_shape).all():  # If cropping is smaller than original image
            img_crop = img[x[0] : x[1], y[0] : y[1], z[0] : z[1]]
        elif crop_shape == img_shape:  # If image is already cropped
            warn("Image is already cropped.", RuntimeWarning)
            img_crop = img
        else:  # If cropping is larger than original image
            raise ValueError("Cropped image shape must be smaller than original image.")
    else:  # If image is not 2D or 3D
        raise ValueError("Image must be 2D or 3D.")
    return img_crop


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


def _resize_image(img_r, img_m, scaling_order=1):
    # Resize image if shapes unequal
    if img_r.shape != img_m.shape:
        img_m = ski.transform.resize(
            img_m, img_r.shape, preserve_range=True, order=scaling_order
        )
        img_m = img_m.astype(img_r.dtype)
    return img_m
