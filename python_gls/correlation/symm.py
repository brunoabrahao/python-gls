"""Unstructured (general symmetric) correlation structure."""

import numpy as np
from numpy.typing import NDArray

from python_gls.correlation.base import CorStruct
from python_gls._parametrization import (
    unconstrained_to_corr,
    corr_to_unconstrained,
)


class CorSymm(CorStruct):
    """Unstructured (general symmetric) correlation.

    Estimates all d(d-1)/2 unique off-diagonal correlations freely.
    Uses spherical parametrization to ensure positive-definiteness.

    Equivalent to R's `corSymm()`.

    Parameters
    ----------
    dim : int or None
        Dimension of the correlation matrix. If None, inferred from data.
    """

    def __init__(self, dim: int | None = None):
        super().__init__()
        if dim is not None:
            if not isinstance(dim, int):
                raise TypeError(f"dim must be an integer, got {type(dim).__name__}")
            if dim < 2:
                raise ValueError(f"dim must be >= 2 for a correlation matrix, got {dim}")
        self._dim = dim

    @property
    def n_params(self) -> int:
        if self._dim is None:
            raise ValueError("Dimension not set. Call initialize() first or pass dim=.")
        return self._dim * (self._dim - 1) // 2

    def get_correlation_matrix(self, group_size: int, **kwargs) -> NDArray:
        """Build unstructured correlation matrix from current parameters."""
        if self._params is None:
            return np.eye(group_size)
        return unconstrained_to_corr(self._params, group_size)

    def _get_init_params(self, residuals_by_group: list[NDArray]) -> NDArray:
        """Initialize from sample correlation of residuals."""
        # Determine dimension from data
        sizes = [len(r) for r in residuals_by_group]
        if len(set(sizes)) != 1:
            d = max(sizes)
        else:
            d = sizes[0]
        self._dim = d

        # Compute sample correlation from residuals
        # Stack residuals into matrix (n_groups x d)
        equal_groups = [r for r in residuals_by_group if len(r) == d]
        if len(equal_groups) >= 2:
            resid_mat = np.vstack(equal_groups)
            if resid_mat.shape[0] > d:
                R_sample = np.corrcoef(resid_mat.T)
            else:
                R_sample = np.eye(d)
        else:
            R_sample = np.eye(d)

        # Ensure positive-definiteness
        eigvals = np.linalg.eigvalsh(R_sample)
        if np.min(eigvals) < 1e-6:
            R_sample = R_sample + (1e-6 - np.min(eigvals) + 1e-6) * np.eye(d)
            # Renormalize to correlation
            d_inv = np.diag(1.0 / np.sqrt(np.diag(R_sample)))
            R_sample = d_inv @ R_sample @ d_inv

        return corr_to_unconstrained(R_sample)

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        # Params are already unconstrained (spherical parametrization)
        return params.copy()

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        return uparams.copy()
