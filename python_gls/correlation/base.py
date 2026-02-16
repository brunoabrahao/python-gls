"""Base class for correlation structures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class CorStruct(ABC):
    """Abstract base class for correlation structures.

    A correlation structure defines within-group correlations for GLS
    estimation. Each group (e.g., subject, cluster) has its own
    correlation matrix, but all groups share the same parameters.

    Subclasses must implement:
        - get_correlation_matrix(group_size, **kwargs)
        - n_params (property)
        - _get_init_params(residuals_by_group)
    """

    def __init__(self) -> None:
        self._params: NDArray | None = None
        self._unconstrained_params: NDArray | None = None

    @abstractmethod
    def get_correlation_matrix(self, group_size: int, **kwargs) -> NDArray:
        """Return the correlation matrix for a group of given size.

        Parameters
        ----------
        group_size : int
            Number of observations in this group.
        **kwargs
            Additional context (e.g., time points, positions).

        Returns
        -------
        R : (group_size, group_size) correlation matrix.
        """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of correlation parameters."""

    @abstractmethod
    def _get_init_params(self, residuals_by_group: list[NDArray]) -> NDArray:
        """Compute initial parameter values from OLS residuals.

        Parameters
        ----------
        residuals_by_group : list of arrays
            Residuals split by group.

        Returns
        -------
        params : array of initial parameter values.
        """

    def get_params(self) -> NDArray:
        """Get current parameter values."""
        if self._params is None:
            raise ValueError(
                f"{type(self).__name__} parameters not yet initialized. "
                f"Call initialize() first or fit the model."
            )
        return self._params.copy()

    def set_params(self, params: NDArray) -> None:
        """Set parameter values."""
        params = np.asarray(params, dtype=float)
        if params.ndim != 1:
            raise ValueError(
                f"params must be a 1-D array, got shape {params.shape}"
            )
        self._params = params

    def get_unconstrained_params(self) -> NDArray:
        """Get unconstrained (transformed) parameters for optimization."""
        if self._unconstrained_params is None:
            if self._params is None:
                raise ValueError(
                    f"{type(self).__name__} parameters not yet initialized. "
                    f"Call initialize() first or fit the model."
                )
            return self._params_to_unconstrained(self._params)
        return self._unconstrained_params.copy()

    def set_unconstrained_params(self, uparams: NDArray) -> None:
        """Set parameters from unconstrained (transformed) values."""
        uparams = np.asarray(uparams, dtype=float)
        self._unconstrained_params = uparams
        self._params = self._unconstrained_to_params(uparams)

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        """Transform natural parameters to unconstrained space.

        Default: identity (override for constrained parameters).
        """
        return params.copy()

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        """Transform unconstrained parameters to natural space.

        Default: identity (override for constrained parameters).
        """
        return uparams.copy()

    def initialize(self, residuals_by_group: list[NDArray]) -> None:
        """Initialize parameters from OLS residuals.

        Parameters
        ----------
        residuals_by_group : list of arrays
            Residuals split by group.
        """
        if not residuals_by_group:
            raise ValueError(
                "residuals_by_group must be a non-empty list of arrays"
            )
        self._params = self._get_init_params(residuals_by_group)
        self._unconstrained_params = self._params_to_unconstrained(self._params)
