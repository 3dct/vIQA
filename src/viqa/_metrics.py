from abc import ABC, abstractmethod


class FullReferenceMetricsInterface(ABC):
    def __init__(self, data_range, normalize, batch, **kwargs):
        self._parameters = {
            "data_range": data_range,
            "normalize": normalize,
            "batch": batch,
            "chromatic": False,
            **kwargs,
        }
        self.score_val = None
        if self._parameters["normalize"] and not self._parameters["data_range"]:
            raise ValueError("If normalize is True, data_range must be specified")

    @abstractmethod
    def score(self, img_r, img_m):
        pass

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
