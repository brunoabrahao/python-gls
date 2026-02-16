"""VarIdent: heterogeneous variance by group."""

import numpy as np
from numpy.typing import NDArray

from python_gls.variance.base import VarFunc


class VarIdent(VarFunc):
    """Different variance per group level.

    Models Var(y_i) = sigma^2 * delta_g^2 where g is the group that
    observation i belongs to. One group is the reference (delta = 1).

    Equivalent to R's `varIdent(form = ~1 | group)`.

    Parameters
    ----------
    group_var : str
        Name of the grouping variable in the data.
    """

    def __init__(self, group_var: str):
        super().__init__()
        if not isinstance(group_var, str):
            raise TypeError(
                f"group_var must be a string (column name), got {type(group_var).__name__}"
            )
        self.group_var = group_var
        self._levels: list | None = None
        self._ref_level = None

    @property
    def n_params(self) -> int:
        if self._levels is None:
            return 0
        return len(self._levels) - 1  # One level is reference

    def get_weights(self, data: dict, group_indices: NDArray) -> NDArray:
        if self._params is None or self._levels is None:
            return np.ones(len(group_indices))

        group_vals = np.asarray(data[self.group_var])[group_indices]

        # Build mapping: reference level -> 1.0, others -> exp(param)
        weights = np.ones(len(group_indices))
        param_idx = 0
        for level in self._levels:
            if level == self._ref_level:
                continue
            mask = group_vals == level
            weights[mask] = np.exp(self._params[param_idx])
            param_idx += 1

        return weights

    def _get_init_params(self, residuals: NDArray, data: dict) -> NDArray:
        group_vals = np.asarray(data[self.group_var])
        self._levels = sorted(set(group_vals), key=str)
        self._ref_level = self._levels[0]

        # Initialize from ratio of group std devs
        ref_mask = group_vals == self._ref_level
        ref_std = np.std(residuals[ref_mask]) if np.sum(ref_mask) > 1 else 1.0

        params = []
        for level in self._levels:
            if level == self._ref_level:
                continue
            mask = group_vals == level
            if np.sum(mask) > 1:
                level_std = np.std(residuals[mask])
                ratio = level_std / (ref_std + 1e-10)
                params.append(np.log(max(ratio, 1e-6)))
            else:
                params.append(0.0)

        return np.array(params) if params else np.array([])

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        return params.copy()  # Already unconstrained (log scale)

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        return uparams.copy()
