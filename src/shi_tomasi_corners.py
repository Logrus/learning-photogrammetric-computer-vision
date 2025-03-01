#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#
import numpy as np


def shi_tomasi_corners(xx: np.ndarray, yy: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """
    Compute the Shi-Tomasi corner response for given image gradients.

    The function takes in elements of a Structure Matrix.

        M = [[ xx, xy ]
            [ xy, yy ]]

    Returns:
        ndarray: The Shi-Tomasi corner response.
    """
    trace = xx + yy
    det = xx * yy - xy**2

    R = trace / 2.0 - 0.5 * np.sqrt(trace**2 - 4 * det)
    return R
