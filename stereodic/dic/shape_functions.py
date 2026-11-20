"""Subset shape functions (warps) in centred local coordinates (dx, dy).

Parameter vectors
    order 0: p = (u, v)
    order 1: p = (u, u_x, u_y, v, v_x, v_y)
    order 2: second-order (Gao et al. 2015, 6x6 lifted matrix) — planned for M2.

Warps are represented as 3x3 homogeneous matrices acting on (dx, dy, 1):
    [x', y', 1]^T = M(p) [dx, dy, 1]^T,   M = [[1+u_x, u_y, u], [v_x, 1+v_y, v], [0, 0, 1]].
The IC-GN update is W(p) <- W(p) o W(dp)^-1, i.e. M <- M(p) @ inv(M(dp)).
All functions accept batched parameters with shape (..., P).
"""

from __future__ import annotations

import numpy as np

_N_PARAMS = {0: 2, 1: 6}


def _check(order: int) -> None:
    if order == 2:
        raise NotImplementedError("second-order shape function is scheduled for M2 (Gao et al. 2015)")
    if order not in _N_PARAMS:
        raise ValueError(f"unsupported shape function order {order}")


def n_params(order: int) -> int:
    _check(order)
    return _N_PARAMS[order]


def identity(order: int) -> np.ndarray:
    return np.zeros(n_params(order))


def to_matrix(p: np.ndarray, order: int) -> np.ndarray:
    _check(order)
    p = np.asarray(p, dtype=np.float64)
    M = np.zeros(p.shape[:-1] + (3, 3))
    M[..., 0, 0] = M[..., 1, 1] = M[..., 2, 2] = 1.0
    if order == 0:
        M[..., 0, 2], M[..., 1, 2] = p[..., 0], p[..., 1]
    else:
        M[..., 0, 0] += p[..., 1]
        M[..., 0, 1] = p[..., 2]
        M[..., 0, 2] = p[..., 0]
        M[..., 1, 0] = p[..., 4]
        M[..., 1, 1] += p[..., 5]
        M[..., 1, 2] = p[..., 3]
    return M


def from_matrix(M: np.ndarray, order: int) -> np.ndarray:
    _check(order)
    M = np.asarray(M, dtype=np.float64)
    if order == 0:
        return np.stack([M[..., 0, 2], M[..., 1, 2]], axis=-1)
    return np.stack(
        [M[..., 0, 2], M[..., 0, 0] - 1.0, M[..., 0, 1], M[..., 1, 2], M[..., 1, 0], M[..., 1, 1] - 1.0], axis=-1
    )


def warp(p: np.ndarray, dx: np.ndarray, dy: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Warped local coordinates (dx', dy') for a single parameter vector p."""
    M = to_matrix(p, order)
    return M[0, 0] * dx + M[0, 1] * dy + M[0, 2], M[1, 0] * dx + M[1, 1] * dy + M[1, 2]


def jacobian(dx: np.ndarray, dy: np.ndarray, order: int) -> np.ndarray:
    """d(x', y')/dp at p = 0, shape (Q, 2, P)."""
    _check(order)
    dx = np.asarray(dx, dtype=np.float64).ravel()
    dy = np.asarray(dy, dtype=np.float64).ravel()
    q = dx.size
    J = np.zeros((q, 2, n_params(order)))
    if order == 0:
        J[:, 0, 0] = 1.0
        J[:, 1, 1] = 1.0
    else:
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = 1.0, dx, dy
        J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = 1.0, dx, dy
    return J


def compose_inverse(p: np.ndarray, dp: np.ndarray, order: int) -> np.ndarray:
    """IC-GN update: parameters of W(p) o W(dp)^-1."""
    return from_matrix(to_matrix(p, order) @ np.linalg.inv(to_matrix(dp, order)), order)


def center_displacement(p: np.ndarray, order: int) -> np.ndarray:
    """Displacement (u, v) of the subset centre, shape (..., 2)."""
    _check(order)
    p = np.asarray(p, dtype=np.float64)
    return p[..., [0, 1]] if order == 0 else p[..., [0, 3]]


def from_displacement(uv: np.ndarray, order: int) -> np.ndarray:
    """Parameters of a pure translation (u, v), shape (..., P)."""
    uv = np.asarray(uv, dtype=np.float64)
    p = np.zeros(uv.shape[:-1] + (n_params(order),))
    if order == 0:
        p[..., 0], p[..., 1] = uv[..., 0], uv[..., 1]
    else:
        p[..., 0], p[..., 3] = uv[..., 0], uv[..., 1]
    return p


def max_corner_motion(dp: np.ndarray, half_size: float, order: int) -> float:
    """Largest displacement of the four subset corners under W(dp) (convergence measure)."""
    M = to_matrix(dp, order)
    h = float(half_size)
    corners = np.array([[-h, -h, 1.0], [h, -h, 1.0], [-h, h, 1.0], [h, h, 1.0]])
    moved = corners @ M.T
    return float(np.max(np.hypot(moved[:, 0] - corners[:, 0], moved[:, 1] - corners[:, 1])))
