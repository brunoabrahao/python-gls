"""VarExp: exponential variance function."""

import numpy as np
from numpy.typing import NDArray

from python_gls.variance.base import VarFunc


class VarExp(VarFunc):
    """Variance as exponential of a covariate.

    Models: sd(y_i) = exp(delta * v_i) where v_i is the covariate value.

    Equivalent to R's `varExp(form = ~covariate)`.

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
        return 1

    def get_weights(self, data: dict, group_indices: NDArray) -> NDArray:
        if self._params is None:
            return np.ones(len(group_indices))
        v = np.asarray(data[self.covariate], dtype=float)[group_indices]
        delta = self._params[0]
        # Clip exponent to avoid overflow
        exponent = np.clip(delta * v, -500, 500)
        return np.exp(exponent)

    def _get_init_params(self, residuals: NDArray, data: dict) -> NDArray:
        return np.array([0.0])

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        return params.copy()

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        return uparams.copy()
