"""Spherical parametrization for correlation matrices.

Based on Pinheiro & Bates (1996). Transforms between correlation matrices
and unconstrained angle parameters, ensuring positive-definiteness during
optimization without bound constraints.

A d×d correlation matrix has d(d-1)/2 free parameters. We parametrize via
angles θ ∈ (0, π), which map to a Cholesky factor L such that R = LL'.
The angles are unconstrained on the real line via logit-like transformation.
"""

import numpy as np
from numpy.typing import NDArray


def angles_to_cholesky(angles: NDArray, d: int) -> NDArray:
    """Convert angle parameters to lower-triangular Cholesky factor.

    Parameters
    ----------
    angles : array of shape (d*(d-1)/2,)
        Angle parameters in (0, pi).
    d : int
        Dimension of the correlation matrix.

    Returns
    -------
    L : array of shape (d, d)
        Lower-triangular Cholesky factor such that L @ L.T is a
        correlation matrix.
    """
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    L = np.zeros((d, d))
    idx = 0
    for i in range(d):
        if i == 0:
            L[0, 0] = 1.0
            continue
        # Row i uses angles[idx : idx + i]
        row_sin = sin_a[idx : idx + i]
        row_cos = cos_a[idx : idx + i]
        # Cumulative sine products: cumprod[k] = prod(sin_a[idx..idx+k])
        cum_sin = np.cumprod(row_sin)
        # j == 0: cos(angles[idx])
        L[i, 0] = row_cos[0]
        # j == 1..i-1: prod(sin[idx..idx+j-1]) * cos(angles[idx+j])
        for j in range(1, i):
            L[i, j] = cum_sin[j - 1] * row_cos[j]
        # j == i (last): prod(sin[idx..idx+i-1])
        L[i, i] = cum_sin[i - 1]
        idx += i
    return L


def cholesky_to_corr(L: NDArray) -> NDArray:
    """Convert Cholesky factor to correlation matrix."""
    R = L @ L.T
    # Ensure exact ones on diagonal (numerical stability)
    d = np.sqrt(np.diag(R))
    R = R / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


def angles_to_corr(angles: NDArray, d: int) -> NDArray:
    """Convert unconstrained angles to a correlation matrix.

    Parameters
    ----------
    angles : array of shape (d*(d-1)/2,)
        Angle parameters in (0, pi).
    d : int
        Dimension of the correlation matrix.

    Returns
    -------
    R : array of shape (d, d)
        Positive-definite correlation matrix.
    """
    L = angles_to_cholesky(angles, d)
    return cholesky_to_corr(L)


def corr_to_angles(R: NDArray) -> NDArray:
    """Convert a correlation matrix to angle parameters.

    Parameters
    ----------
    R : array of shape (d, d)
        Positive-definite correlation matrix.

    Returns
    -------
    angles : array of shape (d*(d-1)/2,)
        Angle parameters in (0, pi).
    """
    d = R.shape[0]
    L = np.linalg.cholesky(R)
    # Normalize rows to unit length
    norms = np.sqrt(np.sum(L ** 2, axis=1))
    L = L / norms[:, np.newaxis]

    n_angles = d * (d - 1) // 2
    angles = np.zeros(n_angles)
    idx = 0
    for i in range(1, d):
        # j == 0
        angles[idx] = np.arccos(np.clip(L[i, 0], -1, 1))
        # j == 1..i-1: product uses preceding angles in this row
        for j in range(1, i):
            prod = np.prod(np.sin(angles[idx : idx + j]))
            if abs(prod) < 1e-15:
                angles[idx + j] = np.pi / 2
            else:
                angles[idx + j] = np.arccos(np.clip(L[i, j] / prod, -1, 1))
        idx += i
    return angles


def unconstrained_to_angles(params: NDArray) -> NDArray:
    """Map unconstrained parameters to (0, pi) via scaled sigmoid."""
    return np.pi / (1 + np.exp(-params))


def angles_to_unconstrained(angles: NDArray) -> NDArray:
    """Map angles in (0, pi) to unconstrained parameters."""
    # Clip to avoid log(0)
    ratio = np.clip(angles / np.pi, 1e-10, 1 - 1e-10)
    return np.log(ratio / (1 - ratio))


def unconstrained_to_corr(params: NDArray, d: int) -> NDArray:
    """Map unconstrained parameters directly to a correlation matrix."""
    angles = unconstrained_to_angles(params)
    return angles_to_corr(angles, d)


def corr_to_unconstrained(R: NDArray) -> NDArray:
    """Map a correlation matrix to unconstrained parameters."""
    angles = corr_to_angles(R)
    return angles_to_unconstrained(angles)
