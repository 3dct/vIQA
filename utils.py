import os
import re
import glob
import numpy as np
from pathlib import Path
from torch import Tensor


def _load_data_from_disk(file_dir: str or Path or os.PathLike, file_name: str or Path or os.PathLike) -> np.ndarray:
    """
    Loads data from a .mhd file and its corresponding .raw file or a .raw file only and normalizes it.
    :param file_dir: Directory of the file
    :param file_name: Name of the file with extension
    :return: Numpy array containing the data
    """

    # Create file path components
    file_name_split = os.path.splitext(file_name)  # Split file name and extension
    file_name_head = file_name_split[0]  # File name without extension
    file_ext = file_name_split[-1]
    file_path = os.path.join(file_dir, file_name)  # Complete file path

    # Check file extension
    if file_ext == '.mhd':  # If file is a .mhd file
        f = open(file=file_path, mode="rt")  # Open header file

        file_header_txt = f.read().split("\n")  # Extract header lines
        file_header = {key: value for line in file_header_txt[0:-1] for key, value in [line.split(" = ")]}  # Create dictionary from lines

        data_file_path = os.path.join(file_dir, file_header["ElementDataFile"])  # Get data file path from header

        # Extract dimension
        file_header.update({"DimSize": [int(val) for val in file_header["DimSize"].split()]})  # Change DimSize to type int
        dim_size = file_header["DimSize"]  # Get DimSize from header

        # Check bit depth
        bit_depth = file_header["ElementType"]  # Get ElementType from header

        # Set data type according to bit depth
        if bit_depth == 'MET_USHORT':
            data_type = np.ushort  # Set data type to unsigned short
        elif bit_depth == 'MET_UCHAR':
            data_type = np.ubyte  # Set data type to unsigned byte
        else:
            raise Exception("Bit depth not supported")  # Raise exception if bit depth is not supported
    elif file_ext == '.raw':  # If file is a .raw file
        # Check dimension
        dim_search_result = re.search("(\d+(x|_)\d+(x|_)\d+)", file_name_head)  # Search for dimension in file name
        if dim_search_result is not None:  # If dimension was found
            dim = dim_search_result.group(1)  # Get dimension from file name
        else:
            raise Exception("No dimension found")  # Raise exception if no dimension was found

        # Extract dimension
        dim_size = re.split("x|_", dim)  # Split dimension string into list
        dim_size = [int(val) for val in dim_size]  # Change DimSize to type int

        # Check bit depth
        bit_depth_search_result = re.search("(\d{1,2}bit)", file_name_head)  # Search for bit depth in file name
        if bit_depth_search_result is not None:  # If bit depth was found
            bit_depth = bit_depth_search_result.group(1)  # Get bit depth from file name
        else:
            raise Exception("No bit depth found")  # Raise exception if no bit depth was found

        # Set data type according to bit depth
        if bit_depth == '16bit':
            data_type = np.ushort  # Set data type to unsigned short
        elif bit_depth == '8bit':
            data_type = np.ubyte  # Set data type to unsigned byte
        else:
            raise Exception("Bit depth not supported")  # Raise exception if bit depth is not supported

        data_file_path = os.path.join(file_dir, file_name)  # Get data file path
    else:
        raise Exception("File extension not supported")  # Raise exception if file extension is not supported

    # Load data
    f = open(file=data_file_path, mode="rb")  # Open data file
    img_arr_orig = np.fromfile(file=f, dtype=data_type)  # Read data file into numpy array according to data type
    img_arr = img_arr_orig.reshape(*dim_size[::-1])  # Reshape numpy array according to DimSize
    return img_arr


def load_data(img: np.ndarray or Tensor or str or Path or os.PathLike, data_range: int = 255, batch: bool = False,
              normalize: bool = True) -> np.ndarray:
    """
    Loads data from a numpy array, a pytorch tensor or a file path.
    :param img: Numpy array, tensor or file path
    :param data_range: Maximum value of the returned data
    :param batch: If True, img is a file path and all files in the directory are loaded
    :param normalize: If True, data is normalized to data_range
    :return: Numpy array containing the data
    """

    # Check input type
    match img:
        case str() | Path() | os.PathLike():  # If input is a file path
            # Check if batch
            if batch:
                files = glob.glob(img)  # Get all files in directory
                img_arr = []  # Initialize list for numpy arrays
                # Load data from disk for each file
                for file in files:
                    img_arr.append(_load_data_from_disk(file_dir=os.path.dirname(file), file_name=os.path.basename(file)))
            else:
                file_dir = os.path.dirname(img)
                file_name = os.path.basename(img)
                img_arr = _load_data_from_disk(file_dir, file_name)  # Load data from disk
        case np.ndarray():  # If input is a numpy array
            img_arr = img  # Use input as numpy array
        case Tensor():  # If input is a pytorch tensor
            img_arr = img.cpu().numpy()  # Convert tensor to numpy array
        case _:
            raise Exception("Input type not supported")  # Raise exception if input type is not supported

    # Normalize data
    if normalize:
        img_arr = normalize_data(img_arr, data_range)

    return img_arr


def _check_imgs(img_r, img_m, **kwargs) -> (np.ndarray, np.ndarray):
    """
    Checks if two images are of the same type and shape.
    :param img_r: reference image
    :param img_m: modified image
    :return: reference image and modified image
    """

    # load images
    img_r = load_data(img_r, **kwargs)
    img_m = load_data(img_m, **kwargs)

    if img_r.dtype != img_m.dtype:  # If image types do not match
        raise Exception("Image types do not match")

    if img_r.shape != img_m.shape:  # If image shapes do not match
        raise Exception("Image shapes do not match")
    return img_r, img_m


def normalize_data(img_arr, data_range):
    """
    Normalizes a numpy array to a given data range.
    :param img_arr: Input numpy array
    :param data_range: Data range of the returned data
    :return: Input numpy array normalized to data_range
    """
    # Check data type
    if np.issubdtype(img_arr.dtype, np.integer):  # If data type is integer
        info = np.iinfo(img_arr.dtype)
    elif np.issubdtype(img_arr.dtype, np.floating):  # If data type is float
        info = np.finfo(img_arr.dtype)
    else:
        raise Exception("Data type not supported")

    # Check if data is already normalized
    if info.max is not data_range:
        # Normalize data
        img_min = np.min(img_arr)  # Get minimum value of numpy array
        img_max = np.max(img_arr)  # Get maximum value of numpy array
        img_arr = (img_arr - img_min) / (img_max - img_min)  # Normalize numpy array
        img_arr *= data_range  # Scale numpy array to data_range

        # Change data type
        if data_range == 2**8 - 1:  # If data range is 255 (8 bit)
            img_arr = img_arr.astype(np.uint8)  # Change data type to unsigned byte
        elif data_range == 2**16 - 1:  # If data range is 65535 (16 bit)
            img_arr = img_arr.astype(np.uint16)  # Change data type to unsigned short
        elif data_range == 1:  # If data range is 1
            img_arr = img_arr.astype(np.float32)  # Change data type to float32
        else:
            raise Exception("Data range not supported. Please use 1, 255 or 65535.")

    return img_arr


def _to_float(img):
    """
    Converts a numpy array to float.
    :param img: numpy array
    :return: numpy array as float
    """
    match img.dtype:
        case np.float32 | np.float64:
            return img
        case _:
            return img.astype(np.float64)


def correlate_convolve_abs(img, kernel, mode='correlate', border_mode='constant', value=0):
    """
    Correlates or convolves a numpy array with a kernel in the form mean(abs(img * kernel)). Works in 2D and 3D.
    :param img: Input numpy array
    :param kernel: Kernel
    :param mode: 'correlate' or 'convolve'
    :param border_mode: 'constant', 'reflect', 'nearest', 'mirror' or 'wrap'
    :param value: Value for constant border mode
    :return: Convolved result as numpy array
    """
    if mode == 'convolve':  # If mode is convolve
        kernel = np.flip(kernel)  # Flip kernel

    kernel_size = kernel.shape[0]  # Get kernel size
    ndim = len(img.shape)  # Get number of dimensions

    # Pad image
    match border_mode:
        case 'constant':
            origin = np.pad(img, kernel_size, mode='constant', constant_values=value)
        case 'reflect':
            origin = np.pad(img, kernel_size, mode='reflect')
        case 'nearest':
            origin = np.pad(img, kernel_size, mode='edge')
        case 'mirror':
            origin = np.pad(img, kernel_size, mode='symmetric')
        case 'wrap':
            origin = np.pad(img, kernel_size, mode='wrap')
        case _:
            raise Exception("Border mode not supported")

    # Correlate or convolve
    res = np.zeros(img.shape)  # Initialize result array
    for k in range(0, img.shape[0]):
        for m in range(0, img.shape[1]):
            # Check if 2D or 3D
            if ndim == 3:
                for n in range(0, img.shape[2]):
                    res[k, m, n] = np.mean(abs(kernel * origin[k:k + kernel_size,
                                                               m:m + kernel_size,
                                                               n:n + kernel_size]))
            elif ndim == 2:
                res[k, m] = np.mean(abs(kernel * origin[k:k + kernel_size,
                                                        m:m + kernel_size]))
            else:
                raise Exception("Number of dimensions not supported")

    return res
