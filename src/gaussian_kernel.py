#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np


def gaussian_kernel(kernel_size: int = 3, sigma: float = 0.5) -> np.ndarray:
    """Generate a 2D Gaussian kernel."""

    # Generate values around zero mean with given kernel size
    half_size = (kernel_size - 1) // 2
    x = np.linspace(-half_size, half_size, kernel_size)

    # Create a grid of coordinates in 2D space
    xx, yy = np.meshgrid(x, x)

    # Compute Gaussian kernel
    kernel = np.exp(-1 / 2 * (xx**2 + yy**2) / sigma**2)

    # Normalize kernel to sum to 1
    kernel = kernel / np.sum(kernel)

    # Make sure the returned kernel is of expected type
    return kernel.astype(np.float32)
