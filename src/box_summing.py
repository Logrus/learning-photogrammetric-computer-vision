#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np


def box_summing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Fast box summing convolution using integral images."""

    h, w = image.shape
    k = kernel_size // 2

    # Create output image for the sum
    output = np.zeros_like(image, dtype=np.float64)

    # Compute integral image
    integral = np.cumsum(np.cumsum(np.asarray(image, dtype=np.float64), axis=0), axis=1)

    for y in range(k, h - k):
        for x in range(k, w - k):
            # left x, top y
            lx, ty = x - k, y - k
            # right x, bottom y
            rx, by = x + k, y + k

            sum_value = float(integral[by, rx])
            if lx > 0:
                sum_value -= integral[by, lx - 1]
            if ty > 0:
                sum_value -= integral[ty - 1, rx]
            if lx > 0 and ty > 0:
                sum_value += integral[ty - 1, lx - 1]

            output[y, x] = sum_value

    return output
