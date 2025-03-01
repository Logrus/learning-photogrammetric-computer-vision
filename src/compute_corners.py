#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np
from scipy import ndimage
from src.gaussian_kernel import gaussian_kernel
from src.sobel_kernel import sobel_kernel_Dx, sobel_kernel_Dy
from src.harris_corners import harris_corners
from src.shi_tomasi_corners import shi_tomasi_corners
from src.box_summing import box_summing
from src.non_maximum_suppression import nms_2d
from enum import Enum


# Enum for corner types
class CornerCriterion(Enum):
    HARRIS = "harris"
    SHI_TOMASI = "shi-tomasi"


def compute_corners(image: np.ndarray, criterion: CornerCriterion, rel_threshold=0.1) -> np.ndarray:
    """Compute corner keypoints using Harris or Shi-Tomasi criteria."""

    # Prepare gaussian and Sobel kernels
    G = gaussian_kernel(3, 0.5)
    Dx = sobel_kernel_Dx()
    Dy = sobel_kernel_Dy()

    # Convolve kernels to do pre-smoothing
    Kx = ndimage.convolve(G, Dx)
    Ky = ndimage.convolve(G, Dy)

    # Calculate image derivatives
    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)

    # Compute elements
    Ix2 = Ix**2
    Iy2 = Iy**2
    IxIy = Ix * Iy

    # Sum patches to compute elements
    # of a structured matrix
    kernel_size = 3
    xx = box_summing(Ix2, kernel_size)
    xy = box_summing(IxIy, kernel_size)
    yy = box_summing(Iy2, kernel_size)

    if criterion == CornerCriterion.HARRIS:
        R = harris_corners(xx, yy, xy, k=0.04)
    elif criterion == CornerCriterion.SHI_TOMASI:
        R = shi_tomasi_corners(xx, yy, xy)
    else:
        raise RuntimeError(f"Unknown corner detector {type}")

    # Perform non maximum suppression
    nms_image = nms_2d(R, 3)

    # Apply additional thresholding
    rel_T = np.max(R) * rel_threshold
    corners = np.column_stack(np.where(nms_image > rel_T))

    # Revert corners from row, col to x, y format
    return corners[:, ::-1]
