#!/usr/bin/python3

import glob
import os
import sys

import scipy.io as sio
import viqa


def write_to_mat(data_dir, filename="volume.mhd"):
    """Write the images in the data directory to .mat files."""
    # Get subfolders of the data directory
    subfolders = glob.glob(os.path.join(data_dir, "*"))

    counter = 0
    for subfolder in subfolders:
        print(".", end="")
        img = viqa.load_data(os.path.join(subfolder, filename),
                             normalize=False,
                             batch=False)
        img_norm = viqa.normalize_data(img,
                                       data_range_output=(0, 1),
                                       automatic_data_range=True)
        sio.savemat(os.path.join(subfolder, "image.mat"), {"image": img_norm})
        counter += 1

    print(f"\n{counter} files saved as .mat.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        if len(sys.argv) > 2:
            filename = sys.argv[2]
        else:
            filename = "volume.mhd"
        write_to_mat(data_dir, filename)
        print("Done!")
    else:
        print("Usage: python3 write_to_mat.py <data_dir> [filename]")
        sys.exit(1)
