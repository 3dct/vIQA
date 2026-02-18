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

"""Module for the internal metrics classes.

This module contains the abstract classes for the metrics.
"""

from abc import ABC, abstractmethod

from viqa.utils import _check_imgs, export_results, load_data


class Metric:
    def __init__(self, data_range, normalize, **kwargs):
        self.parameters = {
            "data_range": data_range,
            "normalize": normalize,
            "chromatic": False,
            "roi": None,
            **kwargs,
        }
        self.score_val = None
        self._name = None
        if self.parameters["normalize"] and not self.parameters["data_range"]:
            raise ValueError("If normalize is True, data_range must be specified")

    @abstractmethod
    def score(self, *args):
        """Calculate the score."""
        pass

    @abstractmethod
    def print_score(self, *args):
        """Print the score."""
        pass

    def export_results(self, path, filename):
        """Export the score to a csv file.

        Parameters
        ----------
        path : str
            The path where the csv file should be saved.
        filename : str
            The name of the csv file.

        Notes
        -----
        The arguments get passed to :py:func:`.viqa.utils.export_results`.
        """
        export_results([self], path, filename)

    def __eq__(self, other):
        if isinstance(other, Metric):
            return self.score_val == other.score_val
        else:
            return self.score_val == other

    def __lt__(self, other):
        if isinstance(other, Metric):
            return self.score_val < other.score_val
        else:
            return self.score_val < other

    def __gt__(self, other):
        if isinstance(other, Metric):
            return self.score_val > other.score_val
        else:
            return self.score_val > other

    def __le__(self, other):
        if isinstance(other, Metric):
            return self.score_val <= other.score_val
        else:
            return self.score_val <= other

    def __ge__(self, other):
        if isinstance(other, Metric):
            return self.score_val >= other.score_val
        else:
            return self.score_val >= other

    def __ne__(self, other):
        if isinstance(other, Metric):
            return self.score_val != other.score_val
        else:
            return self.score_val != other

    def __repr__(self):
        return f"{self.__class__.__name__}(result={self.score_val})"


class FullReferenceMetricsInterface(ABC, Metric):
    def __init__(self, data_range, normalize, **kwargs):
        super().__init__(data_range, normalize, **kwargs)
        self.type = "full-reference"

    def load_images(self, img_r, img_m):
        """Load the images and perform checks.

        Parameters
        ----------
        img_r : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
            The reference image.
        img_m : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
            The modified image.

        Returns
        -------
        img_r : viqa.ImageArray
            The loaded reference image as an :py:class:`viqa.utils.ImageArray`.
        img_m : viqa.ImageArray
            The loaded modified image as an :py:class:`viqa.utils.ImageArray`.
        """
        img_r, img_m = _check_imgs(
            img_r=img_r,
            img_m=img_m,
            data_range=self.parameters["data_range"],
            normalize=self.parameters["normalize"],
            chromatic=self.parameters["chromatic"],
            roi=self.parameters["roi"],
        )
        return img_r, img_m


class NoReferenceMetricsInterface(ABC, Metric):
    def __init__(self, data_range, normalize, **kwargs):
        super().__init__(data_range, normalize, **kwargs)
        self.type = "no-reference"

    def load_images(self, img):
        """Load the image.

        Uses the :py:func:`.viqa.utils.load_data` function to load the image.

        Parameters
        ----------
        img : np.ndarray, viqa.ImageArray, torch.Tensor, str or os.PathLike
            The image to load.

        Returns
        -------
        img : viqa.ImageArray
            The loaded image as an :py:class:`viqa.utils.ImageArray`.
        """
        # Load image
        img = load_data(
            img=img,
            data_range=self.parameters["data_range"],
            normalize=self.parameters["normalize"],
            roi=self.parameters["roi"],
        )
        return img
