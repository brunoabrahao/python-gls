"""ARMA(p,q) correlation structure."""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz

from python_gls.correlation.base import CorStruct


class CorARMA(CorStruct):
    """ARMA(p,q) correlation structure.

    Defines correlation via an autoregressive moving-average process.
    The autocorrelation function is computed from AR and MA coefficients.

    Equivalent to R's `corARMA(p=p, q=q)`.

    Parameters
    ----------
    p : int
        Order of the AR component.
    q : int
        Order of the MA component.
    """

    def __init__(self, p: int = 0, q: int = 0):
        super().__init__()
        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(
                f"p and q must be integers, got p={type(p).__name__}, q={type(q).__name__}"
            )
        if p < 0 or q < 0:
            raise ValueError(f"p and q must be non-negative, got p={p}, q={q}")
        if p == 0 and q == 0:
            raise ValueError("At least one of p or q must be > 0.")
        self.p = p
        self.q = q

    @property
    def n_params(self) -> int:
        return self.p + self.q

    def _compute_acf(self, max_lag: int) -> NDArray:
        """Compute autocorrelation function from ARMA parameters."""
        ar = self._params[:self.p] if self.p > 0 else np.array([])
        ma = self._params[self.p:] if self.q > 0 else np.array([])

        # Compute ACF of ARMA(p,q) process via Yule-Walker-like recursion
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        # For pure AR
        if self.q == 0 and self.p > 0:
            # Yule-Walker: gamma(h) = sum_i phi_i * gamma(h-i)
            for h in range(1, max_lag + 1):
                for i in range(min(self.p, h)):
                    if h - i - 1 >= 0:
                        acf[h] += ar[i] * acf[abs(h - i - 1)]
            return acf

        # For pure MA
        if self.p == 0 and self.q > 0:
            theta = np.concatenate([[1.0], ma])
            for h in range(min(self.q + 1, max_lag + 1)):
                num = sum(
                    theta[j] * theta[j + h]
                    for j in range(self.q + 1 - h)
                )
                denom = sum(theta[j] ** 2 for j in range(self.q + 1))
                acf[h] = num / denom
            return acf

        # General ARMA: use impulse response function
        n_impulse = max(max_lag + 1, 100)
        psi = np.zeros(n_impulse)
        psi[0] = 1.0
        ma_full = np.zeros(n_impulse)
        ma_full[:self.q] = ma
        ar_full = np.zeros(n_impulse)
        ar_full[:self.p] = ar

        for i in range(1, n_impulse):
            if i <= self.q:
                psi[i] = ma_full[i - 1]
            for j in range(min(self.p, i)):
                psi[i] += ar_full[j] * psi[i - j - 1]

        # ACF from impulse response
        for h in range(max_lag + 1):
            num = sum(psi[j] * psi[j + h] for j in range(n_impulse - h))
            denom = sum(psi[j] ** 2 for j in range(n_impulse))
            acf[h] = num / denom

        return acf

    def get_correlation_matrix(self, group_size: int, **kwargs) -> NDArray:
        if self._params is None:
            return np.eye(group_size)
        acf = self._compute_acf(group_size - 1)
        R = toeplitz(acf)
        # Ensure positive-definiteness
        eigvals = np.linalg.eigvalsh(R)
        if np.min(eigvals) < 1e-10:
            R += (1e-10 - np.min(eigvals) + 1e-10) * np.eye(group_size)
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)
        return R

    def _get_init_params(self, residuals_by_group: list[NDArray]) -> NDArray:
        # Small initial values
        return np.zeros(self.n_params) + 0.1

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        # Use tanh transform for stability
        return np.arctanh(np.clip(params, -0.999, 0.999))

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        return np.tanh(uparams)
