#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

from scipy.ndimage import maximum_filter
import numpy as np


def nms_2d(scores: np.ndarray, neighborhood_size: int = 3) -> np.ndarray:
    """
    Performs non-maximum suppression on a 2D score array.

    Parameters:
        scores (2D array): Input score map (higher values are kept).
        neighborhood_size (int): Suppression window size.

    Returns:
        suppressed_scores (2D array): Scores after NMS.
    """

    # Find local maxima using a max filter
    # max_filter is a function that replaces each element in an array with the local maximum value
    # within a given neighborhood.
    # We then compare scores with the maximum values, this produces a mask of True values at local maxima
    # and False elsewhere.
    max_mask = scores == maximum_filter(scores, size=neighborhood_size, mode="constant", cval=0)

    # Suppress all but local maxima
    # by multiplying local maxima with True (1) and other values with False (0)
    suppressed_scores = scores * max_mask

    return suppressed_scores
