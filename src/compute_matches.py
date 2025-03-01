#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np


def compute_matches(D1, D2):
    """Computes matches for two images using the descriptors.
    Uses the Lowe's criteria to determine the best match."""
    N1 = len(D1)
    N2 = len(D2)

    scores = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            d1 = D1[i]
            d2 = D2[j]
            scores[i, j] = np.linalg.norm(d1 - d2)
    idx = np.argsort(scores, axis=1)
    indices = idx[:, :2]
    two_scores = scores[np.arange(indices.shape[0])[:, None], indices]

    correspondences = np.array([np.arange(indices.shape[0]), idx[:, 0]])
    if two_scores.shape[1] > 1:
        passed = two_scores[:, 0] < 0.75 * two_scores[:, 1]
        correspondences = correspondences[:, passed].T
    else:
        correspondences = correspondences.T
    return correspondences
