"""AR(1) correlation structure."""

import numpy as np
from numpy.typing import NDArray

from python_gls.correlation.base import CorStruct


class CorAR1(CorStruct):
    """First-order autoregressive correlation.

    R[i,j] = phi^|i-j| for equally-spaced observations.

    Equivalent to R's `corAR1()`.

    Parameters
    ----------
    phi : float, optional
        Autoregressive parameter, |phi| < 1.
    """

    def __init__(self, phi: float | None = None):
        super().__init__()
        if phi is not None:
            if not isinstance(phi, (int, float)):
                raise TypeError(f"phi must be a number, got {type(phi).__name__}")
            if not -1 < phi < 1:
                raise ValueError(
                    f"phi must be in (-1, 1) for stationarity, got {phi}"
                )
            self._params = np.array([float(phi)])

    @property
    def n_params(self) -> int:
        return 1

    def get_correlation_matrix(self, group_size: int, **kwargs) -> NDArray:
        if self._params is None:
            return np.eye(group_size)
        phi = self._params[0]
        indices = np.arange(group_size)
        R = phi ** np.abs(indices[:, None] - indices[None, :])
        return R

    def get_correlation_matrix_inverse(self, group_size: int, **kwargs) -> NDArray:
        if self._params is None:
            return np.eye(group_size)
        phi = self._params[0]
        if abs(phi) < 1e-15:
            return np.eye(group_size)
        m = group_size
        if m == 1:
            return np.array([[1.0]])
        phi2 = phi * phi
        denom = 1.0 - phi2
        R_inv = np.zeros((m, m))
        # Diagonal entries
        R_inv[0, 0] = 1.0 / denom
        R_inv[m - 1, m - 1] = 1.0 / denom
        for i in range(1, m - 1):
            R_inv[i, i] = (1.0 + phi2) / denom
        # Off-diagonal entries
        off = -phi / denom
        for i in range(m - 1):
            R_inv[i, i + 1] = off
            R_inv[i + 1, i] = off
        return R_inv

    def get_log_determinant(self, group_size: int, **kwargs) -> float:
        if self._params is None:
            return 0.0
        phi = self._params[0]
        if abs(phi) < 1e-15:
            return 0.0
        m = group_size
        if m <= 1:
            return 0.0
        return (m - 1) * np.log(1.0 - phi * phi)

    def _get_init_params(self, residuals_by_group: list[NDArray]) -> NDArray:
        # Estimate phi from lag-1 autocorrelation
        lag1_corrs = []
        for r in residuals_by_group:
            if len(r) < 2:
                continue
            r_centered = r - np.mean(r)
            var = np.var(r_centered)
            if var > 1e-10:
                lag1 = np.sum(r_centered[:-1] * r_centered[1:]) / (len(r) * var)
                lag1_corrs.append(lag1)
        if lag1_corrs:
            phi = np.clip(np.mean(lag1_corrs), -0.99, 0.99)
        else:
            phi = 0.0
        return np.array([phi])

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        phi = np.clip(params[0], -0.999, 0.999)
        return np.array([np.arctanh(phi)])

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        return np.array([np.tanh(uparams[0])])
