#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np


def sobel_kernel_Dx() -> np.ndarray:
    """Generate a 2D Sobel kernel along X dimension."""
    return 1.0 / 8 * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)


def sobel_kernel_Dy() -> np.ndarray:
    """Generate a 2D Sobel kernel along Y dimension."""
    return sobel_kernel_Dx().T
