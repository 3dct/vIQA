"""Module for utility functions for exporting.

Examples
--------
    .. doctest-skip::

        >>> from viqa import export_image, export_metadata, export_results, FSIM, PSNR
        >>> metric1 = FSIM()
        >>> metric2 = PSNR()
        >>> metrics = [metric1, metric2]
        >>> results_dict = export_results(
        >>>    metrics,
        >>>    "path/to/output",
        >>>    "filename.csv",
        >>>    return_dict=True,
        >>> )
        >>> export_metadata(metrics, [{"param1": 1}, {"param2": 2}], "path/to/output")
        >>> export_image(
        >>>    results_dict,
        >>>    "path/to/reference/image",
        >>>    "path/to/modified/image",
        >>>    x=50,
        >>> )
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
import os
from datetime import datetime
from importlib.metadata import version
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from .loading import _check_imgs, _resize_image, load_data


def export_results(metrics, output_path, filename, return_dict=False):
    """Export data to a csv file.

    Parameters
    ----------
    metrics : list
        List of metrics
    output_path : str or os.PathLike
        Output path
    filename : str or os.PathLike
        Name of the file
    return_dict : bool, optional
        If True, the results are returned as a dictionary. Default is False.

    Notes
    -----
    This function just writes the ``score_val`` attribute of instanced metrics
    to a csv file. Therefore, the metrics must have been calculated before exporting and
    no-reference metrics cannot be distinguished between reference and modified image.

    .. attention::

        The csv file will be overwritten if it already exists.

    Examples
    --------
        .. doctest-skip::

            >>> from viqa import export_results, FSIM, PSNR
            >>> metric1 = FSIM()
            >>> metric2 = PSNR()
            >>> metrics = [metric1, metric2]
            >>> export_results(metrics, "path/to/output", "filename.csv")
    """
    # Check if filename has the correct extension
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    results_dict = {}
    # Create file path
    file_path = os.path.join(output_path, filename)
    with open(file_path, mode="w", newline="") as f:  # Open file
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric in metrics:
            if metric.score_val is None:
                metric.score_val = "n/a"
            else:
                writer.writerow([metric._name, metric.score_val])
            results_dict[metric._name] = metric.score_val
    if return_dict:
        return results_dict


def export_metadata(metrics, metrics_parameters, file_path, file_name="metadata.txt"):
    """Export the metadata (custom parameters and package version) to a txt file.

    Parameters
    ----------
    metrics : list
        List of metric instances.
    metrics_parameters : list
        List of dictionaries containing the parameters for the metrics.
    file_path : str
        Path to the directory where the txt file should be saved.
    file_name : str, default='metadata.txt'
        Name of the txt file. Default is 'metadata.txt'.

    Notes
    -----
        .. attention::

            The txt file will be overwritten if it already exists.
    """
    if os.path.splitext(file_name)[1] != ".txt":
        raise ValueError(f"The file name {file_name} must have the extension '.txt'.")
    path = os.path.join(file_path, file_name)
    with open(path, mode="w") as txtfile:
        txtfile.write("vIQA_version: " + version("viqa") + "\n")
        txtfile.write("Time: " + datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        txtfile.write("\n")
        txtfile.write("\n")
        txtfile.write("custom metric parameters: \n")
        txtfile.write("========================= \n")
        for metric_num, metric in enumerate(metrics):
            txtfile.write(metric.__str__().split("(")[0])
            txtfile.write("\n")
            [txtfile.write("-") for _ in metric.__str__().split("(")[0]]
            txtfile.write("\n")
            txtfile.write("data_range: " + str(metric.parameters["data_range"]) + "\n")
            for key, value in metrics_parameters[metric_num].items():
                txtfile.write(key + ": " + str(value))
                txtfile.write("\n")
            txtfile.write("\n")


def export_image(
    results,
    img_r,
    img_m,
    x=None,
    y=None,
    z=None,
    file_path=None,
    file_name="image_comparison.png",
    show_image=True,
    **kwargs,
):
    """Print the reference and modified image side by side with the metric values.

    Parameters
    ----------
    results : dict
        Dictionary containing the metric values.
    img_r : str or np.ndarray
        Path to the reference image or the image itself.
    img_m : str or np.ndarray
        Path to the modified image or the image itself.
    x, y, z : int, optional
        The index of the slice to be plotted. Only one axis can be specified.
    file_path : str, optional
        Path to the directory where the image should be saved. If None, the image
        will be displayed only.
    file_name : str, optional
        Name of the image file. Default is 'image_comparison.png'.
    show_image : bool, optional
        If True, the image will be displayed. Default is True.
    kwargs : dict
        Additional parameters. Passed to :py:func:`matplotlib.pyplot.subplots`.

    Other Parameters
    ----------------
    dpi : int, default=300
        Dots per inch of the figure.
    scaling_order : int, default=1
        Order of the spline interpolation used for image resizing. Default is 1.
        Passed to :py:func:`skimage.transform.resize`

    Raises
    ------
    ValueError
        If the area to be plotted was not correctly specified.
        If the image is not 2D or 3D.
        If no axis or more than one axis was specified.
        If the images have different number of dimensions.

    Warns
    -----
    UserWarning
        If no results are available to plot.
    """
    if len(results) == 0:
        warn("No results to plot. Only the images are plotted.", UserWarning)

    dpi = kwargs.pop("dpi", 300)

    img_r = load_data(img_r)
    img_m = load_data(img_m)
    # Check if images have the same no of dimensions
    if not img_r.ndim == img_m.ndim:
        raise ValueError("Images must have the same number of dimensions.")
    scaling_order = kwargs.pop("scaling_order", 1)
    img_m = _resize_image(img_r, img_m, scaling_order=scaling_order)
    img_r, img_m = _check_imgs(img_r, img_m)

    if img_r.ndim == 2 or (img_r.ndim == 3 and img_r.shape[-1] == 3):
        # For 2D (color) images, flip the image to match the imshow orientation
        img_r_plot = np.flip(img_r, 1)
        img_m_plot = np.flip(img_m, 1)
    elif img_r.ndim == 3:
        if {x, y, z} == {None}:
            raise ValueError("One axis must be specified")
        if len({x, y, z} - {None}) != 1:
            raise ValueError("Only one axis can be specified")

        # For 3D images, plot the specified area
        x_1, x_2 = 0, img_r.shape[0]
        y_1, y_2 = 0, img_r.shape[1]
        z_1, z_2 = 0, img_r.shape[2]

        if x is not None:
            img_r_plot = np.rot90(np.flip(img_r[x, y_1:y_2, z_1:z_2], 1))
            img_m_plot = np.rot90(np.flip(img_m[x, y_1:y_2, z_1:z_2], 1))
        elif y is not None:
            img_r_plot = np.rot90(np.flip(img_r[x_1:x_2, y, z_1:z_2], 1))
            img_m_plot = np.rot90(np.flip(img_m[x_1:x_2, y, z_1:z_2], 1))
        elif z is not None:
            img_r_plot = np.rot90(np.flip(img_r[x_1:x_2, y_1:y_2, z], 0), -1)
            img_m_plot = np.rot90(np.flip(img_m[x_1:x_2, y_1:y_2, z], 0), -1)
        else:
            raise ValueError("Area to be plotted was not correctly specified")
    else:
        raise ValueError("Image must be 2D or 3D")

    fig, axs = plt.subplots(1, 2, dpi=dpi, **kwargs)
    axs[0].imshow(img_r_plot, cmap="gray")
    axs[0].invert_yaxis()
    axs[1].imshow(img_m_plot, cmap="gray")
    axs[1].invert_yaxis()

    fig.suptitle("Image Comparison and IQA metric values", y=0.92)
    axs[0].set_title("Reference image")
    axs[1].set_title("Modified image")

    num_full_reference = 0
    num_no_reference = 0
    for metric in results.keys():
        if not (metric.endswith("_r") or metric.endswith("_m")):
            num_full_reference += 1
        else:
            num_no_reference += 1

    cols = ((num_full_reference - 1) // 4) + 1  # 4 metrics per column
    if num_no_reference != 0:
        cols += 1

    # Split the results into full-reference and no-reference metrics
    results_fr = {
        k: v for k, v in results.items() if not (k.endswith("_r") or k.endswith("_m"))
    }
    results_nr = {
        k: v for k, v in results.items() if k.endswith("_r") or k.endswith("_m")
    }

    # Plot full-reference metrics
    counter = 0
    for i in range(cols - 1):
        x_pos = 1.0 / (cols + 1)
        lines = 4  # 4 metrics per column
        x_pos = x_pos * (i + 1)
        for j in range(lines):
            if counter < num_full_reference:
                y_pos = 0.09 - 0.03 * j  # 0.09 is the top of the plot
                metric, result = list(results_fr.items())[counter]
                fig.text(
                    x_pos, y_pos, f"{metric}: {result:.2f}", ha="center", fontsize=8
                )
                counter += 1
    # Plot no-reference metrics
    x_pos = 1.0 / (cols + 1) * cols  # last column
    for j in range(num_no_reference):
        y_pos = 0.09 - 0.03 * j
        metric, result = list(results_nr.items())[j]
        fig.text(x_pos, y_pos, f"{metric}: {result:.2f}", ha="center", fontsize=8)

    axs[0].axis("off")
    axs[1].axis("off")

    # Check if filename has the correct extension
    if os.path.splitext(file_name)[1] != ".png":
        raise ValueError(f"The file name {file_name} must have the extension '.png'.")

    if file_path:
        file_path = os.path.join(file_path, file_name)
        plt.savefig(file_path, bbox_inches="tight", pad_inches=0.5)
        if show_image:
            plt.show()
    else:
        plt.show()
