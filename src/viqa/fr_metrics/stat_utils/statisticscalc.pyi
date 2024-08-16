from typing import Tuple

import numpy as np

def minstd(image: np.ndarray, blocksize: int, stride: int) -> np.ndarray:
    """Calculate the minimum standard deviation of blocks of a given image."""
    ...

def getstatistics(
    image: np.ndarray, blocksize: int, stride: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the statistics of blocks of a given image."""
    ...
