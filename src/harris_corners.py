import numpy as np


def harris_corners(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray, k: float = 0.4) -> np.ndarray:
    """Compute the Harris corner response function.

    The function takes in elements of a Structure Matrix.

    M = [[ xx, xy ]
         [ xy, yy ]]

    k is a constant in the range [0.04, 0.06].

    It return the Harris corner response function R.
    """
    R = xx * yy - xy**2 - k * (xx + yy) ** 2
    return R
