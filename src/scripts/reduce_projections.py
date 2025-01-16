#!/usr/bin/python3
"""Reduce the number of projections via CLI."""

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
import sys

from viqa.utils.loading import _load_binary, _parse_bitdepth


def load_projection(file, size, bitdepth):
    """Load projection images from a binary file."""
    data_type = _parse_bitdepth(bitdepth)
    vol = _load_binary(file, data_type, size)
    vol_shape = vol.shape
    vol = vol.reshape(*vol_shape[::-1])
    return vol


def get_n_projections(volume, reducing_factor):
    """Reduce the number of projections."""
    if reducing_factor == 1:
        raise ValueError(f"Error: factor must be greater than 1, got {reducing_factor}")

    volume_new = volume[..., ::reducing_factor]
    return volume_new


def save_volume(output, vol, bitdepth):
    """Save the volume to a binary file."""
    output_folder = os.path.dirname(output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data_type = _parse_bitdepth(bitdepth)
    vol = vol.astype(data_type)
    # save volume
    vol.tofile(output)


if __name__ == "__main__":
    if len(sys.argv) == 8:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        size = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
        bitdepth = sys.argv[6]
        factor = int(sys.argv[7])

        vol = load_projection(input_path, size, bitdepth)
        vol = get_n_projections(vol, factor)
        save_volume(output_path, vol, bitdepth)
    else:
        print(
            "Usage: reduce_projections.py <input_path> <output_path> <size_x> "
            "<size_y> <size_z> <bitdepth> <factor>"
        )
        sys.exit(1)
