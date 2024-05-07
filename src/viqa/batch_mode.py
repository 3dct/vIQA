"""Module for batch functions.

Examples
--------
    .. todo::
        Add examples

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

from viqa.utils import load_data


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

    Examples
    --------
    .. todo::
        Add examples
    """

    def __init__(self, file_dir, pairs_csv, metrics, metrics_parameters):
        self.results = {}
        self.file_dir = file_dir
        self.metrics = metrics
        self.metrics_parameters = metrics_parameters
        self.pairs_csv = pairs_csv
        self.pairs = _read_csv(self.pairs_csv)

    def calculate(self):
        """Calculate the metrics in batch mode."""

        # TODO: Add support for no-reference metrics
        # if pair['modified_image'] == None:
        #     img_r = load_data(reference_image)

        # self._type = 'fr'
        # if metric._type == 'nr':

        for pair_num, pair in enumerate(self.pairs):
            reference_path = os.path.join(self.file_dir, pair['reference_image'])
            modified_path = os.path.join(self.file_dir, pair['modified_image'])
            img_r = load_data(reference_path)
            img_m = load_data(modified_path)
            metric_results = {}
            for metric_num, metric in enumerate(self.metrics):
                result = metric.score(
                    img_r=img_r,
                    img_m=img_m,
                    **self.metrics_parameters[metric_num]
                )
                metric_results[metric._name] = result
            self.results[str(pair_num)] = metric_results

    def export_results(self, file_path, file_name='results.csv'):
        """Export the results to a csv file.

        Parameters
        ----------
        file_path : str
            Path to the directory where the csv file should be saved.
        file_name : str, default='results.csv'
            Name of the csv file. Default is 'results.csv'.
        """
        path = os.path.join(file_path, file_name)
        with open(path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(
                ['pair_num']
                + ['reference_image']
                + ['modified_image']
                + [metric._name for metric in self.metrics]
            )
            # Write data
            for pair_num, results in self.results.items():
                writer.writerow(
                    [pair_num]
                    + [self.pairs[int(pair_num)]['reference_image']]
                    + [self.pairs[int(pair_num)]['modified_image']]
                    + list(results.values()))


def _read_csv(file_path):
    with open(file_path, mode='r', newline='') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.DictReader(csvfile, dialect=dialect)
        return list(reader)
