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

import numpy as np

from viqa.utils.loading import _parse_bitdepth


def load_projection(file, size, bitdepth):
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

    projections = []
    for i in range(size[2]):
        projection = vol[i * size[0] * size[1] : (i + 1) * size[0] * size[1]]
        projection = projection.reshape(size[0], size[1])
        projections.append(projection)
    return projections


def get_n_projections(slices, reducing_factor):
    """Reduce the number of projections."""
    if reducing_factor == 1:
        raise ValueError(f"Error: factor must be greater than 1, got {reducing_factor}")

    projections_reduced = slices[::reducing_factor]
    volume = np.stack(projections_reduced, axis=0)
    return volume


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
            "Usage: reduce_projections.py <input_file> <output_file> <size_x> "
            "<size_y> <size_z> <bitdepth> <factor>"
        )
        sys.exit(1)
