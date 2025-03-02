#
# Author: Vladislav Tananaev
# Date: 01.03.2025
#

import numpy as np
from scipy.spatial.distance import cdist


def compute_matches(D1, D2):
    """Computes matches for two images using the descriptors.
    Uses the Lowe's criteria to determine the best match."""

    scores = cdist(np.asarray(D1), np.asarray(D2), metric='euclidean')

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
