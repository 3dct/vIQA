from abc import ABC, abstractmethod


class Full_Reference_Metrics_Interface(ABC):
    def __init__(self, data_range, **kwargs):
        self._parameters = {'data_range': data_range, 'normalize': True, 'batch': None, 'chromatic': False, **kwargs}
        self.score_val = None

    @abstractmethod
    def score(self, img_r, img_m):
        pass

    @abstractmethod
    def print_score(self):
        pass
