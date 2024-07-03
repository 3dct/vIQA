"""Module for batch functions.

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

from skimage.transform import resize

from viqa.load_utils import load_data


class BatchMetrics:
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

    Notes
    -----
    .. attention::

        In image pairs with unequal shapes, the modified image will be resized to the
        shape of the reference image.

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
        """Constructor method."""
        if len(metrics) != len(metrics_parameters):
            raise ValueError("The number of metrics and metric parameters must be "
                             "equal.")

        self.results = {}
        self.file_dir = file_dir
        self.metrics = metrics
        self.metrics_parameters = metrics_parameters
        self.pairs_csv = pairs_csv
        self.pairs = _read_csv(self.pairs_csv)

    def calculate(self):
        """Calculate the metrics in batch mode."""
        for pair_num, pair in enumerate(self.pairs):
            reference_path = os.path.join(self.file_dir, pair['reference_image'])
            modified_path = os.path.join(self.file_dir, pair['modified_image'])
            img_r = load_data(reference_path)
            img_m = load_data(modified_path)

            # Resize image if shapes unequal
            if img_r.shape != img_m.shape:
                img_m = resize(img_m, img_r.shape, preserve_range=True, order=1)
                img_m = img_m.astype(img_r.dtype)

            metric_results = {}
            for metric_num, metric in enumerate(self.metrics):
                if metric._name not in ["CNR", "SNR", "Q-Measure"]:
                    result = metric.score(
                        img_r=img_r,
                        img_m=img_m,
                        **self.metrics_parameters[metric_num]
                    )
                    metric_results[metric._name] = float(result)
                else:
                    result_r = metric.score(
                        img=img_r,
                        **self.metrics_parameters[metric_num]
                    )
                    result_m = metric.score(
                        img=img_m,
                        **self.metrics_parameters[metric_num]
                    )
                    metric_results[metric._name + '_r'] = float(result_r)
                    metric_results[metric._name + '_m'] = float(result_m)
            self.results[str(pair_num)] = metric_results

    def export_results(self, file_path, file_name='results.csv'):
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
        path = os.path.join(file_path, file_name)
        with open(path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(
                ['pair_num']
                + ['reference_image']
                + ['modified_image']
                + list(self.results[str(0)].keys())
            )
            # Write data
            for pair_num, results in self.results.items():
                writer.writerow(
                    [pair_num]
                    + [self.pairs[int(pair_num)]['reference_image']]
                    + [self.pairs[int(pair_num)]['modified_image']]
                    + list(results.values())
                )


def _read_csv(file_path):
    with open(file_path, mode='r', newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.DictReader(csvfile, dialect=dialect)
        if ('reference_image' not in reader.fieldnames
                or 'modified_image' not in reader.fieldnames):
            raise ValueError("CSV file must contain the columns 'reference_image' and "
                             "'modified_image'.")
        return list(reader)
