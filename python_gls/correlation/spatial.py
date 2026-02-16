"""Spatial correlation structures.

Implements isotropic spatial correlation functions parameterized by
range and optional nugget effect.
"""

import numpy as np
from numpy.typing import NDArray

from python_gls.correlation.base import CorStruct


class _SpatialCorStruct(CorStruct):
    """Base class for spatial correlation structures.

    All spatial structures share:
    - A range parameter controlling decay
    - An optional nugget parameter (discontinuity at distance 0)
    - Distance matrices stored per group

    Parameters
    ----------
    range_param : float, optional
        Initial range parameter (> 0).
    nugget : bool
        Whether to include a nugget effect.
    """

    def __init__(self, range_param: float | None = None, nugget: bool = False):
        super().__init__()
        if range_param is not None:
            if not isinstance(range_param, (int, float)):
                raise TypeError(
                    f"range_param must be a number, got {type(range_param).__name__}"
                )
            if range_param <= 0:
                raise ValueError(
                    f"range_param must be positive, got {range_param}"
                )
        if not isinstance(nugget, bool):
            raise TypeError(f"nugget must be a boolean, got {type(nugget).__name__}")
        self._nugget = nugget
        self._distances: dict[int, NDArray] = {}
        if range_param is not None:
            if nugget:
                self._params = np.array([float(range_param), 0.0])
            else:
                self._params = np.array([float(range_param)])

    @property
    def n_params(self) -> int:
        return 2 if self._nugget else 1

    def set_distances(self, group_id: int, dist_matrix: NDArray) -> None:
        """Set the distance matrix for a group."""
        self._distances[group_id] = np.asarray(dist_matrix, dtype=float)

    def set_coordinates(self, group_id: int, coords: NDArray) -> None:
        """Set coordinates for a group; distances computed automatically."""
        coords = np.asarray(coords, dtype=float)
        if coords.ndim == 1:
            coords = coords[:, None]
        from scipy.spatial.distance import cdist
        self._distances[group_id] = cdist(coords, coords)

    def _correlation_function(self, d: NDArray, range_param: float) -> NDArray:
        """Compute correlation from distances. Override in subclasses."""
        raise NotImplementedError

    def get_correlation_matrix(self, group_size: int, **kwargs) -> NDArray:
        if self._params is None:
            return np.eye(group_size)

        range_param = self._params[0]
        group_id = kwargs.get("group_id", None)

        if group_id is not None and group_id in self._distances:
            dist = self._distances[group_id]
        else:
            # Default: unit-spaced
            idx = np.arange(group_size, dtype=float)
            dist = np.abs(idx[:, None] - idx[None, :])

        R = self._correlation_function(dist, range_param)

        if self._nugget and len(self._params) > 1:
            nug = 1 / (1 + np.exp(-self._params[1]))  # sigmoid to (0, 1)
            R = (1 - nug) * R
            np.fill_diagonal(R, 1.0)

        return R

    def _get_init_params(self, residuals_by_group: list[NDArray]) -> NDArray:
        # Heuristic: set range to median distance
        if self._distances:
            all_dists = []
            for d in self._distances.values():
                mask = np.triu(np.ones(d.shape, dtype=bool), k=1)
                all_dists.extend(d[mask].tolist())
            if all_dists:
                range_init = np.median(all_dists)
            else:
                range_init = 1.0
        else:
            range_init = 1.0
        if self._nugget:
            return np.array([range_init, 0.0])
        return np.array([range_init])

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        # range > 0: use log transform
        u = np.zeros_like(params)
        u[0] = np.log(max(params[0], 1e-10))
        if self._nugget and len(params) > 1:
            u[1] = params[1]  # already unconstrained (sigmoid applied in get_corr)
        return u

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        p = np.zeros_like(uparams)
        p[0] = np.exp(uparams[0])
        if self._nugget and len(uparams) > 1:
            p[1] = uparams[1]
        return p


class CorExp(_SpatialCorStruct):
    """Exponential spatial correlation.

    R(d) = exp(-d / range).

    Equivalent to R's `corExp()`.
    """

    def _correlation_function(self, d: NDArray, range_param: float) -> NDArray:
        return np.exp(-d / range_param)


class CorGaus(_SpatialCorStruct):
    """Gaussian spatial correlation.

    R(d) = exp(-(d/range)^2).

    Equivalent to R's `corGaus()`.
    """

    def _correlation_function(self, d: NDArray, range_param: float) -> NDArray:
        return np.exp(-(d / range_param) ** 2)


class CorLin(_SpatialCorStruct):
    """Linear spatial correlation.

    R(d) = max(1 - d/range, 0).

    Equivalent to R's `corLin()`.
    """

    def _correlation_function(self, d: NDArray, range_param: float) -> NDArray:
        return np.maximum(1 - d / range_param, 0)


class CorRatio(_SpatialCorStruct):
    """Rational quadratic spatial correlation.

    R(d) = 1 / (1 + (d/range)^2).

    Equivalent to R's `corRatio()`.
    """

    def _correlation_function(self, d: NDArray, range_param: float) -> NDArray:
        return 1.0 / (1.0 + (d / range_param) ** 2)


class CorSpher(_SpatialCorStruct):
    """Spherical spatial correlation.

    R(d) = 1 - 1.5*(d/range) + 0.5*(d/range)^3  for d < range
    R(d) = 0  for d >= range

    Equivalent to R's `corSpher()`.
    """

    def _correlation_function(self, d: NDArray, range_param: float) -> NDArray:
        ratio = d / range_param
        R = np.where(
            ratio < 1,
            1 - 1.5 * ratio + 0.5 * ratio ** 3,
            0.0,
        )
        return R
