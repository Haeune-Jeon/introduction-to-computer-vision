import numpy as np

def linear_triangulate_points(P1: np.ndarray, P2: np.ndarray,
                              pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """DLT linear triangulation for multiple correspondences.
    P1, P2: (3x4) projection matrices in pixels; pts1, pts2: (N,2) pixel coords.
    Returns: (N,3) points in world coordinates.
    """
    assert P1.shape == (3,4) and P2.shape == (3,4)
    N = pts1.shape[0]
    X = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        u1, v1 = pts1[i]
        u2, v2 = pts2[i]
        A = np.zeros((4, 4), dtype=np.float64)
        A[0] = u1 * P1[2] - P1[0]
        A[1] = v1 * P1[2] - P1[1]
        A[2] = u2 * P2[2] - P2[0]
        A[3] = v2 * P2[2] - P2[1]
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        X[i] = (Xh[:3] / (Xh[3] + 1e-12))
    return X


