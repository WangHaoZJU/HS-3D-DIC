"""Subset shape functions (warps) in centred local coordinates (dx, dy).

Parameter vectors
    order 0: p = (u, v)
    order 1: p = (u, u_x, u_y, v, v_x, v_y)
    order 2: p = (u, u_x, u_y, u_xx, u_xy, u_yy, v, v_x, v_y, v_xx, v_xy, v_yy)

Warped local position (order 2; lower orders drop terms):
    x' = dx + u + u_x dx + u_y dy + 1/2 u_xx dx^2 + u_xy dx dy + 1/2 u_yy dy^2
    y' = dy + v + v_x dx + v_y dy + 1/2 v_xx dx^2 + v_xy dx dy + 1/2 v_yy dy^2

Orders 0/1 are represented by 3x3 homogeneous matrices acting on xi = (dx, dy, 1).
Order 2 follows Gao et al. (2015): the warp is lifted to a 6x6 matrix acting on
xi = (dx^2, dx dy, dy^2, dx, dy, 1); the rows for x'^2, x'y', y'^2 keep terms up to
second order. Composition/inversion in that space is the standard second-order
IC-GN update (also used by OpenCorr). The truncation does not move the IC-GN fixed
point (dp = 0 <=> J^T r = 0), it only affects the path to convergence.

The IC-GN update is W(p) <- W(p) o W(dp)^-1, i.e. M <- M(p) @ inv(M(dp)), never p -= dp.
All functions accept batched parameters with shape (..., P).
"""

from __future__ import annotations

import numpy as np

_N_PARAMS = {0: 2, 1: 6, 2: 12}
_MATRIX_DIM = {0: 3, 1: 3, 2: 6}
# rows of the matrix that hold x' and y' (the last column is the constant term)
_ROW_X = {0: 0, 1: 0, 2: 3}
_ROW_Y = {0: 1, 1: 1, 2: 4}
# index of v in p
_V_INDEX = {0: 1, 1: 3, 2: 6}


def _check(order: int) -> None:
    if order not in _N_PARAMS:
        raise ValueError(f"unsupported shape function order {order}")


def n_params(order: int) -> int:
    _check(order)
    return _N_PARAMS[order]


def matrix_dim(order: int) -> int:
    _check(order)
    return _MATRIX_DIM[order]


def identity(order: int) -> np.ndarray:
    return np.zeros(n_params(order))


def basis(dx: np.ndarray, dy: np.ndarray, order: int) -> np.ndarray:
    """Lifted local coordinates xi, shape (Q, 3) for orders 0/1, (Q, 6) for order 2."""
    _check(order)
    dx = np.asarray(dx, dtype=np.float64).ravel()
    dy = np.asarray(dy, dtype=np.float64).ravel()
    one = np.ones_like(dx)
    if order < 2:
        return np.column_stack([dx, dy, one])
    return np.column_stack([dx * dx, dx * dy, dy * dy, dx, dy, one])


def _to_matrix2(p: np.ndarray) -> np.ndarray:
    u, ux, uy, uxx, uxy, uyy = (p[..., k] for k in range(6))
    v, vx, vy, vxx, vxy, vyy = (p[..., k] for k in range(6, 12))
    A, B, C, D, E, U = 1.0 + ux, uy, 0.5 * uxx, uxy, 0.5 * uyy, u
    a, b, c, d, e, V = vx, 1.0 + vy, 0.5 * vxx, vxy, 0.5 * vyy, v
    M = np.zeros(p.shape[:-1] + (6, 6))
    # x'^2 (truncated to second order)
    M[..., 0, :] = np.stack([A * A + 2 * U * C, 2 * A * B + 2 * U * D, B * B + 2 * U * E, 2 * U * A, 2 * U * B, U * U], -1)
    # x' y'
    M[..., 1, :] = np.stack(
        [
            A * a + U * c + V * C,
            A * b + B * a + U * d + V * D,
            B * b + U * e + V * E,
            U * a + V * A,
            U * b + V * B,
            U * V,
        ],
        -1,
    )
    # y'^2
    M[..., 2, :] = np.stack([a * a + 2 * V * c, 2 * a * b + 2 * V * d, b * b + 2 * V * e, 2 * V * a, 2 * V * b, V * V], -1)
    M[..., 3, :] = np.stack([C, D, E, A, B, U], -1)
    M[..., 4, :] = np.stack([c, d, e, a, b, V], -1)
    M[..., 5, 5] = 1.0
    return M


def to_matrix(p: np.ndarray, order: int) -> np.ndarray:
    _check(order)
    p = np.asarray(p, dtype=np.float64)
    if p.shape[-1] != _N_PARAMS[order]:
        raise ValueError(f"expected {_N_PARAMS[order]} parameters for order {order}, got {p.shape[-1]}")
    if order == 2:
        return _to_matrix2(p)
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
    if order == 1:
        return np.stack(
            [M[..., 0, 2], M[..., 0, 0] - 1.0, M[..., 0, 1], M[..., 1, 2], M[..., 1, 0], M[..., 1, 1] - 1.0], axis=-1
        )
    rx, ry = M[..., 3, :], M[..., 4, :]
    return np.stack(
        [
            rx[..., 5], rx[..., 3] - 1.0, rx[..., 4], 2.0 * rx[..., 0], rx[..., 1], 2.0 * rx[..., 2],
            ry[..., 5], ry[..., 3], ry[..., 4] - 1.0, 2.0 * ry[..., 0], ry[..., 1], 2.0 * ry[..., 2],
        ],
        axis=-1,
    )


def warp(p: np.ndarray, dx: np.ndarray, dy: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Warped local coordinates (dx', dy') for a single parameter vector p."""
    M = to_matrix(p, order)
    xi = basis(dx, dy, order)
    shape = np.shape(dx)
    return (xi @ M[_ROW_X[order]]).reshape(shape), (xi @ M[_ROW_Y[order]]).reshape(shape)


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
    elif order == 1:
        J[:, 0, 0], J[:, 0, 1], J[:, 0, 2] = 1.0, dx, dy
        J[:, 1, 3], J[:, 1, 4], J[:, 1, 5] = 1.0, dx, dy
    else:
        row = np.column_stack([np.ones_like(dx), dx, dy, 0.5 * dx * dx, dx * dy, 0.5 * dy * dy])
        J[:, 0, 0:6] = row
        J[:, 1, 6:12] = row
    return J


def compose_inverse(p: np.ndarray, dp: np.ndarray, order: int) -> np.ndarray:
    """IC-GN update: parameters of W(p) o W(dp)^-1."""
    return from_matrix(to_matrix(p, order) @ np.linalg.inv(to_matrix(dp, order)), order)


def center_displacement(p: np.ndarray, order: int) -> np.ndarray:
    """Displacement (u, v) of the subset centre, shape (..., 2)."""
    _check(order)
    p = np.asarray(p, dtype=np.float64)
    return p[..., [0, _V_INDEX[order]]]


def from_displacement(uv: np.ndarray, order: int) -> np.ndarray:
    """Parameters of a pure translation (u, v), shape (..., P)."""
    uv = np.asarray(uv, dtype=np.float64)
    p = np.zeros(uv.shape[:-1] + (n_params(order),))
    p[..., 0], p[..., _V_INDEX[order]] = uv[..., 0], uv[..., 1]
    return p


def from_affine(A: np.ndarray, order: int) -> np.ndarray:
    """Parameters from a local affine map x' = A[:, :2] (dx, dy) + A[:, 2], A shape (..., 2, 3).

    For order 0 only the translation is kept.
    """
    A = np.asarray(A, dtype=np.float64)
    if order == 0:
        return np.stack([A[..., 0, 2], A[..., 1, 2]], axis=-1)
    p1 = np.stack(
        [A[..., 0, 2], A[..., 0, 0] - 1.0, A[..., 0, 1], A[..., 1, 2], A[..., 1, 0], A[..., 1, 1] - 1.0], axis=-1
    )
    return convert(p1, 1, order)


def convert(p: np.ndarray, from_order: int, to_order: int) -> np.ndarray:
    """Change the shape-function order; higher-order terms are dropped or set to zero."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape[-1] != n_params(from_order):
        raise ValueError("parameter length does not match from_order")
    if from_order == to_order:
        return p.copy()
    # expand to order 2 as the common representation
    full = np.zeros(p.shape[:-1] + (12,))
    if from_order == 0:
        full[..., 0], full[..., 6] = p[..., 0], p[..., 1]
    elif from_order == 1:
        full[..., 0:3], full[..., 6:9] = p[..., 0:3], p[..., 3:6]
    else:
        full = p.copy()
    if to_order == 2:
        return full
    if to_order == 1:
        return np.concatenate([full[..., 0:3], full[..., 6:9]], axis=-1)
    _check(to_order)
    return np.stack([full[..., 0], full[..., 6]], axis=-1)


def shift_origin(p: np.ndarray, delta: np.ndarray, order: int) -> np.ndarray:
    """Re-express a warp about a new subset centre c_j = c_i + delta (exact for polynomials).

    If the subset at c_i maps reference point c_i + d to c_i + d + disp_i(d), the subset at
    c_j gets disp_j(d) = disp_i(d + delta). Used for RG-DIC propagation of initial guesses.
    """
    _check(order)
    p = np.asarray(p, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    ex, ey = delta[..., 0], delta[..., 1]
    q = p.copy()
    if order == 0:
        return q
    if order == 1:
        for k in (0, 3):
            q[..., k] = p[..., k] + p[..., k + 1] * ex + p[..., k + 2] * ey
        return q
    for k in (0, 6):
        g, gx, gy, gxx, gxy, gyy = (p[..., k + j] for j in range(6))
        q[..., k] = g + gx * ex + gy * ey + 0.5 * gxx * ex * ex + gxy * ex * ey + 0.5 * gyy * ey * ey
        q[..., k + 1] = gx + gxx * ex + gxy * ey
        q[..., k + 2] = gy + gxy * ex + gyy * ey
    return q


def max_corner_motion(dp: np.ndarray, half_size: float, order: int) -> float:
    """Largest displacement of the four subset corners under W(dp) (convergence measure)."""
    h = float(half_size)
    cx = np.array([-h, h, -h, h])
    cy = np.array([-h, -h, h, h])
    wx, wy = warp(dp, cx, cy, order)
    return float(np.max(np.hypot(wx - cx, wy - cy)))
