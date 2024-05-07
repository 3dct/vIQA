from abc import ABC, abstractmethod

from viqa.utils import export_csv


class FullReferenceMetricsInterface(ABC):
    def __init__(self, data_range, normalize, **kwargs):
        self._parameters = {
            "data_range": data_range,
            "normalize": normalize,
            "chromatic": False,
            **kwargs,
        }
        self.score_val = None
        self._name = None
        if self._parameters["normalize"] and not self._parameters["data_range"]:
            raise ValueError("If normalize is True, data_range must be specified")

    @abstractmethod
    def score(self, img_r, img_m):
        pass

    @abstractmethod
    def print_score(self):
        pass

    def export_csv(self, path, filename):
        """Export the score to a csv file.

        Parameters
        ----------
        path : str
            The path where the csv file should be saved.
        filename : str
            The name of the csv file.

        Notes
        -----
        The arguments get passed to :py:func:`.viqa.utils.export_csv`.
        """
        export_csv([self], path, filename)

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


class NoReferenceMetricsInterface(ABC):
    def __init__(self, data_range, normalize, **kwargs):
        self._parameters = {
            "data_range": data_range,
            "normalize": normalize,
            "chromatic": False,
            **kwargs,
        }
        self.score_val = None
        self._name = None
        if self._parameters["normalize"] and not self._parameters["data_range"]:
            raise ValueError("If normalize is True, data_range must be specified")

    @abstractmethod
    def score(self, img):
        pass

    @abstractmethod
    def print_score(self):
        pass

    def export_csv(self, path, filename):
        """Export the score to a csv file.

        Parameters
        ----------
        path : str
            The path where the csv file should be saved.
        filename : str
            The name of the csv file.

        Notes
        -----
        The arguments get passed to :py:func:`.viqa.utils.export_csv`.
        """
        export_csv([self], path, filename)

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
