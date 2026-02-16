"""Continuous-time AR(1) correlation structure."""

import numpy as np
from numpy.typing import NDArray

from python_gls.correlation.base import CorStruct


class CorCAR1(CorStruct):
    """Continuous-time first-order autoregressive correlation.

    R[i,j] = phi^|t_i - t_j| where t_i are (possibly irregular) time points.
    Unlike CorAR1, this handles irregularly-spaced observations.

    Equivalent to R's `corCAR1()`.

    Parameters
    ----------
    phi : float, optional
        Decay parameter, 0 < phi < 1.
    """

    def __init__(self, phi: float | None = None):
        super().__init__()
        if phi is not None:
            if not isinstance(phi, (int, float)):
                raise TypeError(f"phi must be a number, got {type(phi).__name__}")
            if not 0 < phi < 1:
                raise ValueError(
                    f"phi must be in (0, 1) for continuous-time AR(1), got {phi}"
                )
            self._params = np.array([float(phi)])
        self._time_points: dict[int, NDArray] = {}

    @property
    def n_params(self) -> int:
        return 1

    def set_time_points(self, group_id: int, times: NDArray) -> None:
        """Set time points for a specific group.

        Parameters
        ----------
        group_id : int
            Group index.
        times : array
            Time points for this group.
        """
        self._time_points[group_id] = np.asarray(times, dtype=float)

    def get_correlation_matrix(self, group_size: int, **kwargs) -> NDArray:
        if self._params is None:
            return np.eye(group_size)

        phi = self._params[0]
        group_id = kwargs.get("group_id", None)

        if group_id is not None and group_id in self._time_points:
            times = self._time_points[group_id]
        else:
            # Default to equally-spaced
            times = np.arange(group_size, dtype=float)

        time_diffs = np.abs(times[:, None] - times[None, :])
        R = phi ** time_diffs
        return R

    def _get_init_params(self, residuals_by_group: list[NDArray]) -> NDArray:
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
            phi = np.clip(np.mean(lag1_corrs), 0.01, 0.99)
        else:
            phi = 0.5
        return np.array([phi])

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        # phi in (0, 1) -> logit
        phi = np.clip(params[0], 1e-6, 1 - 1e-6)
        return np.array([np.log(phi / (1 - phi))])

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        # Clip to avoid overflow in exp for very large negative values
        u = np.clip(uparams[0], -500, 500)
        return np.array([1 / (1 + np.exp(-u))])
