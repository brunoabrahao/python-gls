"""Compound symmetry (exchangeable) correlation structure."""

import numpy as np
from numpy.typing import NDArray

from python_gls.correlation.base import CorStruct


class CorCompSymm(CorStruct):
    """Compound symmetry (exchangeable) correlation.

    All pairwise correlations are equal to rho. The correlation matrix is:
        R[i,j] = rho for i != j, 1 for i == j.

    Equivalent to R's `corCompSymm()`.

    Parameters
    ----------
    rho : float, optional
        Initial correlation value. Must be in (-1/(d-1), 1) for
        positive-definiteness.
    """

    def __init__(self, rho: float | None = None):
        super().__init__()
        if rho is not None:
            if not isinstance(rho, (int, float)):
                raise TypeError(f"rho must be a number, got {type(rho).__name__}")
            if not -1 < rho < 1:
                raise ValueError(
                    f"rho must be in (-1, 1) for positive-definiteness, got {rho}"
                )
            self._params = np.array([float(rho)])

    @property
    def n_params(self) -> int:
        return 1

    def get_correlation_matrix(self, group_size: int, **kwargs) -> NDArray:
        if self._params is None:
            return np.eye(group_size)
        rho = self._params[0]
        R = np.full((group_size, group_size), rho)
        np.fill_diagonal(R, 1.0)
        return R

    def _get_init_params(self, residuals_by_group: list[NDArray]) -> NDArray:
        # Estimate rho from average pairwise correlation of residuals
        corrs = []
        for r in residuals_by_group:
            d = len(r)
            if d < 2:
                continue
            for i in range(d):
                for j in range(i + 1, d):
                    corrs.append(r[i] * r[j] / (np.std(r) ** 2 + 1e-10))
        if corrs:
            rho = np.clip(np.mean(corrs), -0.9, 0.9)
        else:
            rho = 0.0
        return np.array([rho])

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        # Fisher z-transform: rho -> atanh(rho)
        rho = np.clip(params[0], -0.999, 0.999)
        return np.array([np.arctanh(rho)])

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        return np.array([np.tanh(uparams[0])])
