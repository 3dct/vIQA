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

from abc import ABC, abstractmethod

from viqa.load_utils import load_data
from viqa.utils import _check_imgs, export_results


class Metric:
    def __init__(self, data_range, normalize, **kwargs):
        self._parameters = {
            "data_range": data_range,
            "normalize": normalize,
            "chromatic": False,
            "roi": None,
            **kwargs,
        }
        self.score_val = None
        self._name = None
        if self._parameters["normalize"] and not self._parameters["data_range"]:
            raise ValueError("If normalize is True, data_range must be specified")

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


class FullReferenceMetricsInterface(ABC, Metric):
    def __init__(self, data_range, normalize, **kwargs):
        super().__init__(data_range, normalize, **kwargs)
        self.type = "full-reference"

    @abstractmethod
    def score(self, img_r, img_m):
        img_r, img_m = _check_imgs(
            img_r=img_r,
            img_m=img_m,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            chromatic=self._parameters["chromatic"],
            roi=self._parameters["roi"],
        )
        return img_r, img_m

    @abstractmethod
    def print_score(self):
        pass

    def __eq__(self, other):
        return self.score_val == other.score_val

    def __lt__(self, other):
        return self.score_val < other.score_val

    def __gt__(self, other):
        return self.score_val > other.score_val

    def __le__(self, other):
        return self.score_val <= other.score_val

    def __ge__(self, other):
        return self.score_val >= other.score_val

    def __ne__(self, other):
        return self.score_val != other.score_val

    def __repr__(self):
        return f"{self.__class__.__name__}(score_val={self.score_val})"


class NoReferenceMetricsInterface(ABC, Metric):
    def __init__(self, data_range, normalize, **kwargs):
        super().__init__(data_range, normalize, **kwargs)
        self.type = "no-reference"

    @abstractmethod
    def score(self, img):
        # Load image
        img = load_data(
            img=img,
            data_range=self._parameters["data_range"],
            normalize=self._parameters["normalize"],
            roi=self._parameters["roi"],
        )
        return img

    @abstractmethod
    def print_score(self):
        pass

    def __eq__(self, other):
        return self.score_val == other.score_val

    def __lt__(self, other):
        return self.score_val < other.score_val

    def __gt__(self, other):
        return self.score_val > other.score_val

    def __le__(self, other):
        return self.score_val <= other.score_val

    def __ge__(self, other):
        return self.score_val >= other.score_val

    def __ne__(self, other):
        return self.score_val != other.score_val

    def __repr__(self):
        return f"{self.__class__.__name__}(score_val={self.score_val})"
