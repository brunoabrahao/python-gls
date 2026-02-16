"""VarFixed: pre-specified variance weights."""

import numpy as np
from numpy.typing import NDArray

from python_gls.variance.base import VarFunc


class VarFixed(VarFunc):
    """Fixed (pre-specified) variance weights.

    Weights are provided by the user and not estimated from data.

    Equivalent to R's `varFixed(~covariate)` where the covariate values
    are used directly as variance weights.

    Parameters
    ----------
    weights_var : str
        Name of the variable in the data containing the weights.
        The variance is proportional to these values, so
        sd_i = sqrt(w_i).
    """

    def __init__(self, weights_var: str):
        super().__init__()
        if not isinstance(weights_var, str):
            raise TypeError(
                f"weights_var must be a string (column name), got {type(weights_var).__name__}"
            )
        self.weights_var = weights_var
        self._params = np.array([])  # No parameters to estimate

    @property
    def n_params(self) -> int:
        return 0

    def get_weights(self, data: dict, group_indices: NDArray) -> NDArray:
        w = np.asarray(data[self.weights_var], dtype=float)[group_indices]
        return np.sqrt(np.maximum(w, 1e-10))

    def _get_init_params(self, residuals: NDArray, data: dict) -> NDArray:
        return np.array([])

    def initialize(self, residuals: NDArray, data: dict) -> None:
        self._params = np.array([])
