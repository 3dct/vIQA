#!/usr/bin/python3
"""Crop a volume by a factor via CLI."""

# Authors
# -------
# Author: Lukas Behammer
# Research Center Wels
# University of Applied Sciences Upper Austria, 2023
# CT Research Group
#
# Modifications
# -------------
# Original code, 2025, Lukas Behammer
#
# License
# -------
# BSD-3-Clause License

import os
import re
import sys

import numpy as np

from viqa.utils.loading import _parse_bitdepth


def load_volume(file, size, bitdepth):
    """Load projection images from a binary file."""
    data_type = _parse_bitdepth(bitdepth)
    # Load data
    with open(file=file, mode="rb") as f:  # Open data file
        vol = np.fromfile(
            file=f, dtype=data_type
        )  # Read data file into numpy array according to data type

    if vol.size != np.prod(np.array(size).astype(np.int64)):
        raise ValueError(
            "Size of data file ("
            + file
            + ") does not match dimensions ("
            + str(size)
            + ")"
        )

    slices = []
    for i in range(size[2]):
        slice = vol[i * size[0] * size[1] : (i + 1) * size[0] * size[1]]
        slice = slice.reshape(size[0], size[1])
        slices.append(slice)
    return np.stack(slices, axis=0)


def crop(volume, factor):
    """Crop the volume based on a factor."""
    if factor == 1:
        raise ValueError(f"Error: factor must be greater than 1, got {factor}")
    vol_shape = volume.shape
    center = [dim_shape // 2 for dim_shape in vol_shape]
    new_size = [dim_shape // factor for dim_shape in vol_shape]
    volume_cropped = volume[
        center[0] - (new_size[0] // 2) : center[0] + (new_size[0] // 2),
        center[1] - (new_size[1] // 2) : center[1] + (new_size[1] // 2),
        center[2] - (new_size[2] // 2) : center[2] + (new_size[2] // 2),
    ]
    return volume_cropped


def save_volume(output_folder, file_name, volume, bitdepth):
    """Save the volume to a binary file."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    vol_shape = volume.shape
    file_name = re.sub(
        r"(\d+x\d+x\d+)", f"{vol_shape[2]}x{vol_shape[1]}x{vol_shape[0]}", file_name
    )
    file_name = "crop_" + file_name
    output = os.path.join(output_folder, file_name)
    data_type = _parse_bitdepth(bitdepth)
    volume = volume.astype(data_type)
    # save volume
    volume.tofile(output)


if __name__ == "__main__":
    if len(sys.argv) == 8:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        size = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
        bitdepth = sys.argv[6]
        factor = int(sys.argv[7])
        file_name = os.path.basename(input_path)
        vol = load_volume(input_path, size, bitdepth)
        vol_cropped = crop(vol, factor)
        # vol_resized = ski.transform.resize(
        #     vol_cropped, vol.shape, order=1, preserve_range=True
        # )
        # vol_scaled = ski.transform.rescale(
        #     vol_cropped, factor, order=1, preserve_range=True
        # )
        save_volume(output_path, file_name, vol_cropped, bitdepth)
        # file_name_rescaled = "rescaled_" + file_name
        # save_volume(output_path, file_name_rescaled, vol_scaled, bitdepth)
        # file_name_resized = "resized_" + file_name
        # save_volume(output_path, file_name, vol_resized, bitdepth)
    else:
        print(
            "Usage: crop_image.py <input_file> <output_path> <size_x> "
            "<size_y> <size_z> <bitdepth> <factor>"
        )
        sys.exit(1)
