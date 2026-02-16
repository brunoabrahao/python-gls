"""VarConstPower: constant plus power variance function."""

import numpy as np
from numpy.typing import NDArray

from python_gls.variance.base import VarFunc


class VarConstPower(VarFunc):
    """Variance as constant + power of covariate.

    Models: sd(y_i) = (c + |v_i|^delta) where c > 0, and v_i is the covariate.

    Equivalent to R's `varConstPower(form = ~covariate)`.

    Parameters
    ----------
    covariate : str
        Name of the covariate in the data.
    """

    def __init__(self, covariate: str):
        super().__init__()
        if not isinstance(covariate, str):
            raise TypeError(
                f"covariate must be a string (column name), got {type(covariate).__name__}"
            )
        self.covariate = covariate

    @property
    def n_params(self) -> int:
        return 2  # c (constant) and delta (power)

    def get_weights(self, data: dict, group_indices: NDArray) -> NDArray:
        if self._params is None:
            return np.ones(len(group_indices))
        v = np.abs(np.asarray(data[self.covariate], dtype=float)[group_indices])
        c = np.exp(self._params[0])  # Ensure positive
        delta = self._params[1]
        weights = c + np.where(v > 0, v ** delta, 0.0)
        return np.maximum(weights, 1e-10)

    def _get_init_params(self, residuals: NDArray, data: dict) -> NDArray:
        return np.array([0.0, 0.5])  # c=1 (log scale), delta=0.5

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        return params.copy()  # c is already on log scale

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        return uparams.copy()
