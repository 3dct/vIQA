"""Module for functions to calculate multiple metrics.

This modules contains classes to calculate multiple metrics in batch mode or for one

Examples
--------
.. doctest-skip::

    >>> from viqa import BatchMetrics, MultipleMetrics, PSNR, QMeasure
    >>> metrics = [PSNR(data_range=1), QMeasure(data_range=1)]
    >>> metrics_parameters = [{}, {'hist_bins': 16, 'num_peaks': 2}]
    >>> batch = BatchMetrics(
    ...     file_dir='path/to/images',
    ...     pairs_csv='path/to/pairs.csv',
    ...     metrics=metrics,
    ...     metrics_parameters=metrics_parameters
    ... )
    >>> batch.calculate()
    >>> batch.export_results(file_path='path/to/results', file_name='results.csv')
    >>> img_r = 'path/to/reference_image'
    >>> img_m = 'path/to/modified_image'
    >>> multiple = MultipleMetrics(metrics, metrics_parameters)
    >>> multiple.calculate(img_r, img_m)
    >>> multiple.report(
    ...     csv=True,
    ...     metadata=True,
    ...     text=False,
    ...     image=False,
    ...     file_path='path/to/results'
    ... )

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
from abc import ABC, abstractmethod
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm

from viqa._metrics import Metric
from viqa.load_utils import load_data
from viqa.utils import _check_imgs, _resize_image, export_metadata


class _MultipleInterface(ABC):
    def __init__(self, metrics, metrics_parameters):
        if len(metrics) != len(metrics_parameters):
            raise ValueError(
                "The number of metrics and metric parameters must be equal."
            )
        if not all(isinstance(metric, Metric) for metric in metrics):
            raise ValueError("Metric list contains non-metric objects.")
        if not all(isinstance(parameters, dict) for parameters in metrics_parameters):
            raise ValueError("Parameters list contains non-dictionary objects.")
        self.metrics = metrics
        self.metrics_parameters = metrics_parameters
        self.results = {}

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass

    @abstractmethod
    def report(self, csv, metadata, *args):
        pass

    @abstractmethod
    def export_results(self, file_path, file_name):
        pass

    def export_metadata(self, file_path=".", file_name="metadata.txt"):
        """Export the metadata (custom parameters and package version) to a txt file.

        Parameters
        ----------
        file_path : str
            Path to the directory where the txt file should be saved.
        file_name : str, default='metadata.txt'
            Name of the txt file. Default is 'metadata.txt'.

        Notes
        -----
            .. attention::

                The txt file will be overwritten if it already exists.
        """
        export_metadata(
            self.metrics,
            self.metrics_parameters,
            file_path=file_path,
            file_name=file_name,
        )


class BatchMetrics(_MultipleInterface):
    """Class to calculate metrics in batch mode.

    Attributes
    ----------
    results : dict
        Dictionary containing the results of the metrics.
    file_dir : str
        Directory where the images are stored.
    metrics : list
        List of metric instances.
    metrics_parameters : list
        List of dictionaries containing the parameters for the metrics.
    pairs_csv : str
        Path to the csv file containing the image pairs.
    pairs : list
        List of dictionaries containing the image pairs.

    Parameters
    ----------
    file_dir : str
        Directory where the images are stored.
    pairs_csv : str
        Path to the csv file containing the image pairs.

        .. admonition:: CSV file layout

            +-----------------+----------------+
            | reference_image | modified_image |
            +=================+================+
            | image_path      | image_path     |
            +-----------------+----------------+
            | ...             | ...            |
            +-----------------+----------------+

    metrics : list
        List of metric instances.
    metrics_parameters : list
        List of dictionaries containing the parameters for the metrics.

    Raises
    ------
    ValueError
        If the number of metrics and metric parameters is not equal.
        If the metric list contains non-metric objects.
        If the parameters list contains non-dictionary objects
        If the CSV file does not contain the columns 'reference_image' and
        'modified_image'.

    Notes
    -----
    Make sure to use a well-structured CSV file as performance is better with e.g. the
    same reference image in multiple consecutive rows.

    .. attention::

        In image pairs with unequal shapes, the modified image will be resized to the
        shape of the reference image in the :py:meth:`calculate` method.

    Examples
    --------
    .. doctest-skip::

        >>> from viqa import BatchMetrics, PSNR, QMeasure
        >>> metrics = [PSNR(data_range=1), QMeasure(data_range=1)]
        >>> metrics_parameters = [{}, {'hist_bins': 16, 'num_peaks': 2}]
        >>> batch = BatchMetrics(
        ...     file_dir='path/to/images',
        ...     pairs_csv='path/to/pairs.csv',
        ...     metrics=metrics,
        ...     metrics_parameters=metrics_parameters
        ... )
        >>> batch.calculate()
        >>> batch.export_results(file_path='path/to/results', file_name='results.csv')
    """

    def __init__(self, file_dir, pairs_csv, metrics, metrics_parameters):
        """Construct method."""
        super().__init__(metrics, metrics_parameters)

        self.file_dir = file_dir
        self.pairs_csv = pairs_csv
        self.pairs = _read_csv(self.pairs_csv)

    def calculate(self, **kwargs):
        """Calculate the metrics in batch mode.

        Parameters
        ----------
        kwargs : dict
            Additional parameters. Passed to :py:func:`viqa.load_utils.load_data`.

        Other Parameters
        ----------------
        scaling_order : int, default=1
            Order of the spline interpolation used for image resizing. Default is 1.
            Passed to :py:func:`skimage.transform.resize`.

        Returns
        -------
        results : dict
            Dictionary containing the results of the metrics.

        Warns
        -----
        UserWarning
            If the images are the same as in the previous pair.
        """
        reference_img = None
        prev_ref_path = None
        modified_img = None
        prev_mod_path = None
        metric_results = None
        for pair_num, pair in enumerate(tqdm(self.pairs)):
            reference_path = os.path.join(self.file_dir, pair["reference_image"])
            modified_path = os.path.join(self.file_dir, pair["modified_image"])
            # Skip calculation if the images are the same as in the previous pair
            if reference_path == prev_ref_path and modified_path == prev_mod_path:
                self.results[str(pair_num)] = metric_results
                warn("Skipping calculation for identical image pair.", UserWarning)
                continue
            # Load the images only once if it is the same for multiple pairs
            if reference_path != prev_ref_path:
                reference_img = load_data(reference_path, **kwargs)
                prev_ref_path = reference_path
                prev_result_reference = None
            else:
                prev_result_reference = metric_results
            if modified_path != prev_mod_path:
                modified_img = load_data(modified_path, **kwargs)
                prev_mod_path = modified_path
                prev_result_modified = None
            else:
                prev_result_modified = metric_results

            metric_results = _calc(
                self.metrics,
                self.metrics_parameters,
                reference_img,
                modified_img,
                prev_result_reference=prev_result_reference,
                prev_result_modified=prev_result_modified,
                **kwargs,
            )
            self.results[str(pair_num)] = metric_results
        return self.results

    def report(self, csv=True, metadata=True, file_path="."):
        """Report the results and metadata.

        Parameters
        ----------
        csv : bool, default=True
            If True, the results will be exported to a csv file.
            :py:meth:`export_results` will be called.
        metadata : bool, default=True
            If True, the metadata will be exported to a txt file.
            :py:meth:`export_metadata` will be called.
        file_path : str, optional
            Path to the directory where the files should be saved. If None, the files
            will be saved in the current working directory.
        """
        if csv:
            self.export_results(file_path=file_path, file_name="results.csv")
        if metadata:
            self.export_metadata(file_path=file_path, file_name="metadata.txt")

    def export_results(self, file_path=".", file_name="results.csv"):
        """Export the results to a csv file.

        Parameters
        ----------
        file_path : str
            Path to the directory where the csv file should be saved.
        file_name : str, default='results.csv'
            Name of the csv file. Default is 'results.csv'.

        Notes
        -----
            .. attention::

                The csv file will be overwritten if it already exists.
        """
        if os.path.splitext(file_name)[1] != ".csv":
            raise ValueError(
                f"The file name {file_name} must have the " f"extension '.csv'."
            )
        path = os.path.join(file_path, file_name)
        with open(path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(
                ["pair_num"]
                + ["reference_image"]
                + ["modified_image"]
                + list(self.results[str(0)].keys())
            )
            # Write data
            for pair_num, results in self.results.items():
                writer.writerow(
                    [pair_num]
                    + [self.pairs[int(pair_num)]["reference_image"]]
                    + [self.pairs[int(pair_num)]["modified_image"]]
                    + list(results.values())
                )


class MultipleMetrics(_MultipleInterface):
    """Class to calculate metrics in batch mode.

    Attributes
    ----------
    results : dict
        Dictionary containing the results of the metrics.
    metrics : list
        List of metric instances.
    metrics_parameters : list
        List of dictionaries containing the parameters for the metrics.

    Parameters
    ----------
    metrics : list
        List of metric instances.
    metrics_parameters : list
        List of dictionaries containing the parameters for the metrics.

    Raises
    ------
    ValueError
        If the number of metrics and metric parameters is not equal.
        If the metric list contains non-metric objects.
        If the parameters list contains non-dictionary objects

    Notes
    -----
    .. attention::

        In image pairs with unequal shapes, the modified image will be resized to the
        shape of the reference image.

    Examples
    --------
    .. doctest-skip::

        >>> from viqa import MultipleMetrics, PSNR, QMeasure
        >>> metrics = [PSNR(data_range=1), QMeasure(data_range=1)]
        >>> metrics_parameters = [{}, {'hist_bins': 16, 'num_peaks': 2}]
        >>> multiple = MultipleMetrics(
        ...     metrics=metrics,
        ...     metrics_parameters=metrics_parameters
        ... )
        >>> img_r = 'path/to/reference_image'
        >>> img_m = 'path/to/modified_image'
        >>> multiple.calculate(img_r, img_m)
        >>> multiple.report(
        ...     csv=True,
        ...     metadata=True,
        ...     text=False,
        ...     image=False,
        ...     file_path='path/to/results'
        ... )
    """

    def __init__(self, metrics, metrics_parameters):
        super().__init__(metrics, metrics_parameters)

    def calculate(self, img_r, img_m, **kwargs):
        """Calculate multiple metrics for an image pair.

        Parameters
        ----------
        img_r : str or np.ndarray
            Path to the reference image or the image itself.
        img_m : str or np.ndarray
            Path to the modified image or the image itself.
        kwargs : dict
            Additional parameters. Passed to :py:func:`viqa.load_utils.load_data`.

        Other Parameters
        ----------------
        scaling_order : int, default=1
            Order of the spline interpolation used for image resizing. Default is 1.
            Passed to :py:func:`skimage.transform.resize`.

        Returns
        -------
        results : dict
            Dictionary containing the results of the metrics.
        """
        metric_results = _calc(
            self.metrics, self.metrics_parameters, img_r, img_m, leave=True, **kwargs
        )
        self.results = metric_results
        return self.results

    def report(
        self, csv=True, metadata=True, text=True, image=False, file_path=".", **kwargs
    ):
        """Report the results and metadata.

        Parameters
        ----------
        csv : bool, default=True
            If True, the results will be exported to a csv file.
            :py:meth:`export_results` will be called.
        metadata : bool, default=True
            If True, the metadata will be exported to a txt file.
            :py:meth:`export_metadata` will be called.
        text : bool, default=True
            If True, the metric values will be printed to the console.
            :py:meth:`print_values` will be called.
        image : bool, default=False
            If True, the reference and modified image will be plotted side by side.
            :py:meth:`print_image` will be called.
        file_path : str, optional
            Path to the directory where the files should be saved. If None, the files
            will be saved in the current working directory.
        kwargs : dict
            Additional parameters. Passed to :py:func:`print_image`.

        Other Parameters
        ----------------
        decimals : int, default=2
            Number of decimal places for the printed metric values in the console.
        export_image : bool, default=False
            If True, the image will be saved as a file. Default is False.
        img_r : str or np.ndarray
            Path to the reference image or the image itself.
        img_m : str or np.ndarray
            Path to the modified image or the image itself.
        x, y, z : int, optional
            The index of the slice to be plotted. Only one axis can be specified.

        Raises
        ------
        ValueError
            If the reference and modified image are not provided
        """
        decimals = kwargs.pop("decimals", 2)
        export_image = kwargs.pop("export_image", False)
        img_r = kwargs.pop("img_r", None)
        img_m = kwargs.pop("img_m", None)
        x = kwargs.pop("x", None)
        y = kwargs.pop("y", None)
        z = kwargs.pop("z", None)

        if export_image:
            img_file_path = file_path
        else:
            img_file_path = None

        if text:
            self.print_values(decimals)
        if image:
            if img_r is None or img_m is None:
                raise ValueError("Reference and modified image must be provided.")
            self.print_image(
                img_r=img_r,
                img_m=img_m,
                file_path=img_file_path,
                x=x,
                y=y,
                z=z,
                **kwargs,
            )
        if csv:
            self.export_results(file_path=file_path, file_name="results.csv")
        if metadata:
            self.export_metadata(file_path=file_path, file_name="metadata.txt")

    def print_values(self, decimals=2):
        """Print the metric values to the console.

        Parameters
        ----------
        decimals : int, default=2
            Number of decimal places for the printed metric values.
        """
        for metric, result in self.results.items():
            print(f"{metric}: {result:.{decimals}f}")

    def print_image(
        self, img_r, img_m, x=None, y=None, z=None, file_path=None, **kwargs
    ):
        """Print the reference and modified image side by side with the metric values.

        Parameters
        ----------
        img_r : str or np.ndarray
            Path to the reference image or the image itself.
        img_m : str or np.ndarray
            Path to the modified image or the image itself.
        x, y, z : int, optional
            The index of the slice to be plotted. Only one axis can be specified.
        file_path : str, optional
            Path to the directory where the image should be saved. If None, the image
            will be displayed only.
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
        Exception
            If the area to be plotted was not correctly specified.
            If the image is not 2D or 3D.
            If no axis or more than one axis was specified.
        """
        if len(self.results) == 0:
            warn("No results to plot. Only the images are plotted.", UserWarning)

        dpi = kwargs.pop("dpi", 300)

        img_r = load_data(img_r)
        img_m = load_data(img_m)
        scaling_order = kwargs.pop("scaling_order", 1)
        img_m = _resize_image(img_r, img_m, scaling_order=scaling_order)
        img_r, img_m = _check_imgs(img_r, img_m)

        if img_r.ndim == 2 or (img_r.ndim == 3 and img_r.shape[-1] == 3):
            # For 2D (color) images, flip the image to match the imshow orientation
            img_r_plot = np.flip(img_r, 1)
            img_m_plot = np.flip(img_m, 1)
        elif img_r.ndim == 3:
            if {x, y, z} == {None}:
                raise Exception("One axis must be specified")
            if len({x, y, z} - {None}) != 1:
                raise Exception("Only one axis can be specified")

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
                raise Exception("Area to be plotted was not correctly specified")
        else:
            raise Exception("Image must be 2D or 3D")

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
        for metric in self.results.keys():
            if not (metric.endswith("_r") or metric.endswith("_m")):
                num_full_reference += 1
            else:
                num_no_reference += 1

        cols = ((num_full_reference - 1) // 4) + 1  # 4 metrics per column
        if num_no_reference != 0:
            cols += 1

        # Split the results into full-reference and no-reference metrics
        results_fr = {
            k: v
            for k, v in self.results.items()
            if not (k.endswith("_r") or k.endswith("_m"))
        }
        results_nr = {
            k: v
            for k, v in self.results.items()
            if k.endswith("_r") or k.endswith("_m")
        }

        # Plot full-reference metrics
        counter = 0
        for i in range(cols - 1):
            x_pos = 1.0 / (cols + 1)
            lines = 4  # 4 metrics per column
            x_pos = x_pos * (i + 1)
            for j in range(lines):
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

        if file_path:
            file_name = "image_comparison.png"
            file_path = os.path.join(file_path, file_name)
            plt.savefig(file_path, bbox_inches="tight", pad_inches=0.5)
        plt.show()

    def export_results(self, file_path, file_name="results.csv"):
        """Export the results to a csv file.

        Parameters
        ----------
        file_path : str
            Path to the directory where the csv file should be saved.
        file_name : str, default='results.csv'
            Name of the csv file. Default is 'results.csv'.

        Notes
        -----
            .. attention::

                The csv file will be overwritten if it already exists.
        """
        if os.path.splitext(file_name)[1] != ".csv":
            raise ValueError(
                f"The file name {file_name} must have the " f"extension '.csv'."
            )
        path = os.path.join(file_path, file_name)
        with open(path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(list(self.results.keys()))
            # Write data
            writer.writerow(list(self.results.values()))


def _read_csv(file_path):
    with open(file_path, newline="") as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.DictReader(csvfile, dialect=dialect)
        if (
            "reference_image" not in reader.fieldnames
            or "modified_image" not in reader.fieldnames
        ):
            raise ValueError(
                "CSV file must contain the columns 'reference_image' and "
                "'modified_image'."
            )
        return list(reader)


def _calc(metrics, metrics_parameters, img_r, img_m, **kwargs):
    scaling_order = kwargs.pop("scaling_order", 1)
    leave = kwargs.pop("leave", False)
    prev_result_reference = kwargs.pop("prev_result_reference", None)
    prev_result_modified = kwargs.pop("prev_result_modified", None)

    img_r = load_data(img_r, **kwargs)
    img_m = load_data(img_m, **kwargs)

    img_m = _resize_image(img_r, img_m, scaling_order)

    img_r, img_m = _check_imgs(img_r, img_m, **kwargs)

    metric_results = {}
    for metric, parameters in tqdm(
        zip(metrics, metrics_parameters, strict=False), total=len(metrics), leave=leave
    ):
        if metric.type == "no-reference":
            if prev_result_reference is not None and isinstance(
                prev_result_reference, dict
            ):
                metric_results[name] = prev_result_reference[
                    name := metric._name + "_r"
                ]
            else:
                result_r = metric.score(img_r, **parameters)
                metric_results[metric._name + "_r"] = float(result_r)
            if prev_result_modified is not None and isinstance(
                prev_result_modified, dict
            ):
                metric_results[name] = prev_result_modified[name := metric._name + "_m"]
            else:
                result_m = metric.score(img_m, **parameters)
                metric_results[metric._name + "_m"] = float(result_m)
        elif metric.type == "full-reference":
            result = metric.score(img_r, img_m, **parameters)
            metric_results[metric._name] = float(result)
    return metric_results
