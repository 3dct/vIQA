#!/usr/bin/python3
"""Convert image stack to a volume via CLI."""

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

import glob
import os
import sys

import numpy as np
from viqa.utils.loading import _load_binary, _parse_bitdepth


def combine_image_stack(input_path, size, bitdepth, file_type="raw"):
    """Combine a stack of single images to a volume."""
    files = glob.glob(os.path.join(input_path, f"*.{file_type}"))
    data_type = _parse_bitdepth(bitdepth)

    if len(files) != 0:
        vol = []
        for file in files:
            vol.append(_load_binary(file, data_type, size))

        vol = np.array(vol)
        return vol


def save_volume(output_path, vol, bitdepth, file_type="raw"):
    """Save a volume to disk."""
    data_type = _parse_bitdepth(bitdepth)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    vol_shape = vol.shape
    vol = vol.astype(data_type)
    # save volume
    vol.tofile(
        os.path.join(
            output_path,
            f"volume_{vol_shape[2]}x{vol_shape[1]}x{vol_shape[0]}_{bitdepth}.{file_type}",
        )
    )


if __name__ == "__main__":
    if len(sys.argv) == 9:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        size = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
        bitdepth = sys.argv[6]
        file_type_input = sys.argv[7]
        file_type_output = sys.argv[8]

        vol = combine_image_stack(input_path, size, bitdepth, file_type_input)
        save_volume(output_path, vol, bitdepth, file_type_output)
    else:
        print(
            "Usage: combine_raw_stack.py <input_path> <output_path> <size_x> "
            "<size_y> <size_z> <bitdepth> <file_type_input> <file_type_output>"
        )
        sys.exit(1)
