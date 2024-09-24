#!/usr/bin/python3
"""Write pairs of reference and modified images to a CSV file via CLI."""

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
import shutil
import sys


def write_to_csv(path_modified, path_reference, path_csv, copy=False):
    """Write pairs of reference and modified images to a CSV file."""
    if not os.path.exists(path_csv):
        os.makedirs(path_csv)
        if copy:
            os.makedirs(os.path.join(path_csv, "images"))
    with open(os.path.join(path_csv, "pairs.csv"), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["reference_image", "modified_image"])
        for file_reference in glob.glob(path_reference):
            filename_reference = os.path.basename(file_reference)
            img_ref_name = filename_reference.split(".")[0]
            for file_modified in glob.glob(path_modified):
                filename_modified = os.path.basename(file_modified)
                img_mod_name = filename_modified[:3]
                if img_ref_name == img_mod_name:
                    writer.writerow([filename_reference, filename_modified])
                    if copy:
                        copy_file(
                            file_reference,
                            os.path.join(path_csv, "images", filename_reference),
                        )
                        copy_file(
                            file_modified,
                            os.path.join(path_csv, "images", filename_modified),
                        )


def copy_file(origin_path, destination_path):
    """Copy a file if it does not exist at the destination."""
    if not os.path.isfile(destination_path):
        shutil.copyfile(origin_path, destination_path)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        path_modified = sys.argv[1]
        path_reference = sys.argv[2]
        path_csv = sys.argv[3]
        if sys.argv[4] == "copy":
            copy = True
            print(
                f"Copying images from {path_modified} and {path_reference} "
                f"to {path_csv}/images."
            )
        else:
            copy = False
        path_modified = os.path.join(path_modified, "*")
        path_reference = os.path.join(path_reference, "*")
        write_to_csv(path_modified, path_reference, path_csv, copy=copy)
        print("Done!")
    else:
        print(
            "Usage: python3 write_to_csv.py <path_modified> <path_reference> "
            "<path_csv> [copy]"
        )
        sys.exit(1)
