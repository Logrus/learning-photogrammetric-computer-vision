#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np
from scipy import ndimage
from src.gaussian_kernel import gaussian_kernel
from src.sobel_kernel import sobel_kernel_Dx, sobel_kernel_Dy


def compute_descriptors(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Computes SIFT-inspired 128 bit descriptor."""

    G = gaussian_kernel(3, 0.5)
    Dx = sobel_kernel_Dx()
    Dy = sobel_kernel_Dy()

    Kx = ndimage.convolve(G, Dx)
    Ky = ndimage.convolve(G, Dy)

    Ix = ndimage.convolve(image, Kx)
    Iy = ndimage.convolve(image, Ky)

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(Ix**2 + Iy**2)
    orientation = np.rad2deg(np.arctan2(Iy, Ix) + 2 * np.pi) % 360

    d_radius = 8
    padded_magnitude = np.pad(magnitude, d_radius, constant_values=0)
    padded_orientation = np.pad(orientation, d_radius, constant_values=0)

    descriptors = []
    for keypoint in corners:
        x, y = keypoint[0] + d_radius, keypoint[1] + d_radius

        patch_mag = magnitude[y - d_radius : y + d_radius, x - d_radius : x + d_radius]
        patch_ori = orientation[y - d_radius : y + d_radius, x - d_radius : x + d_radius]

        descriptor = []
        for r in range(0, 16, 4):
            for c in range(0, 16, 4):
                subpatch_mag = patch_mag[r : r + 4, c : c + 4]
                subpatch_ori = patch_ori[r : r + 4, c : c + 4]

                hist, _ = np.histogram(subpatch_ori, bins=8, range=(0, 360), weights=subpatch_mag)
                assert ((0.0 <= subpatch_ori) & (subpatch_ori <= 360.0)).all()
                descriptor.append(hist)

        descriptor = np.array(descriptor).reshape(-1)

        # Descriptor normalization
        norm = np.linalg.norm(descriptor)
        if norm > 0:  # prevent divide by zero
            descriptor = descriptor / norm
        descriptor[descriptor > 0.2] = 0.2
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm

        descriptors.append(descriptor)
    return descriptors
