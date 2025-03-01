#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np


def compute_homography_ransac(C1, C2, M):
    """Implements a RANSAC scheme to estimate the homography and the set of inliers."""

    max_iter = 100
    min_inlier_ratio = 0.6
    inlier_thres = 5

    T = int(np.log(1 - 0.999) / np.log(1 - min_inlier_ratio**4))
    n_iter = min(max_iter, T)

    H_final = None
    M_final = None

    best_count = 0

    for i in range(n_iter):

        # Select 4 random correspondencies
        randi = np.random.choice(M.shape[0], 4, replace=False)
        idx = M[randi]

        # Select points from image 1 and image 2 with correspondencies
        P1_4 = C1[idx[:, 0]]
        P2_4 = C2[idx[:, 1]]

        # Compute homography
        H = calculate_homography_four_matches(P1_4, P2_4)

        # compute residual
        P1 = C1[M[:, 0]]
        P2 = C2[M[:, 1]]
        res = compute_residual(P1, P2, H)

        count = np.sum(res <= inlier_thres)
        if count > best_count:
            best_count = count
            H_final = H
            M_final = M[res <= inlier_thres]

    return H_final, M_final, best_count


# Calculate the geometric distance between estimated points and original points, namely residuals.
def compute_residual(P1, P2, H):
    """Compute the residual given the Homography H"""
    P1T = np.pad(P1, ((0, 0), (0, 1)), constant_values=1) @ H.T
    P1T = P1T / P1T[:, 2:3]
    P1T = P1T[:, :2]
    residual = np.linalg.norm(P2 - P1T, axis=1)
    return residual


def calculate_homography_four_matches(P1, P2):
    """Estimate the homography given four correspondening keypoints in the two images."""

    if P1.shape[0] != 4 or P2.shape[0] != 4:
        print("Four corresponding points needed to compute Homography")
        return None

    # loop through correspondences and create assemble matrix
    # A * h = 0, where A(2n,9), h(9,1)

    A = []
    for i in range(P1.shape[0]):
        p1 = np.array([P1[i, 0], P1[i, 1], 1])
        p2 = np.array([P2[i, 0], P2[i, 1], 1])

        a2 = [
            0,
            0,
            0,
            -p2[2] * p1[0],
            -p2[2] * p1[1],
            -p2[2] * p1[2],
            p2[1] * p1[0],
            p2[1] * p1[1],
            p2[1] * p1[2],
        ]
        a1 = [
            -p2[2] * p1[0],
            -p2[2] * p1[1],
            -p2[2] * p1[2],
            0,
            0,
            0,
            p2[0] * p1[0],
            p2[0] * p1[1],
            p2[0] * p1[2],
        ]
        A.append(a1)
        A.append(a2)

    A = np.array(A)

    # svd composition
    # the singular value is sorted in descending order
    u, s, v = np.linalg.svd(A)

    # we take the “right singular vector” (a column from V )
    # which corresponds to the smallest singular value
    H = np.reshape(v[8], (3, 3))

    # normalize and now we have H
    H = (1 / H[2, 2]) * H

    return H
