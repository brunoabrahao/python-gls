"""VarComb: combination of multiple variance functions."""

import numpy as np
from numpy.typing import NDArray

from python_gls.variance.base import VarFunc


class VarComb(VarFunc):
    """Combination (product) of multiple variance functions.

    The total variance weight is the product of weights from each
    component function.

    Equivalent to R's `varComb(varFunc1, varFunc2, ...)`.

    Parameters
    ----------
    *varfuncs : VarFunc
        Variance function instances to combine.
    """

    def __init__(self, *varfuncs: VarFunc):
        super().__init__()
        if len(varfuncs) < 2:
            raise ValueError("VarComb requires at least 2 variance functions.")
        for i, vf in enumerate(varfuncs):
            if not isinstance(vf, VarFunc):
                raise TypeError(
                    f"All arguments must be VarFunc instances, but argument {i} "
                    f"is {type(vf).__name__}"
                )
        self.varfuncs = list(varfuncs)

    @property
    def n_params(self) -> int:
        return sum(vf.n_params for vf in self.varfuncs)

    def get_weights(self, data: dict, group_indices: NDArray) -> NDArray:
        weights = np.ones(len(group_indices))
        for vf in self.varfuncs:
            weights *= vf.get_weights(data, group_indices)
        return weights

    def get_params(self) -> NDArray:
        return np.concatenate([vf.get_params() for vf in self.varfuncs])

    def set_params(self, params: NDArray) -> None:
        idx = 0
        for vf in self.varfuncs:
            n = vf.n_params
            vf.set_params(params[idx : idx + n])
            idx += n

    def get_unconstrained_params(self) -> NDArray:
        parts = []
        for vf in self.varfuncs:
            up = vf.get_unconstrained_params()
            if len(up) > 0:
                parts.append(up)
        return np.concatenate(parts) if parts else np.array([])

    def set_unconstrained_params(self, uparams: NDArray) -> None:
        idx = 0
        for vf in self.varfuncs:
            n = vf.n_params
            if n > 0:
                vf.set_unconstrained_params(uparams[idx : idx + n])
            idx += n

    def _get_init_params(self, residuals: NDArray, data: dict) -> NDArray:
        parts = []
        for vf in self.varfuncs:
            p = vf._get_init_params(residuals, data)
            if len(p) > 0:
                parts.append(p)
        return np.concatenate(parts) if parts else np.array([])

    def initialize(self, residuals: NDArray, data: dict) -> None:
        for vf in self.varfuncs:
            vf.initialize(residuals, data)
        self._params = self.get_params()
