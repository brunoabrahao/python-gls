"""Base class for variance functions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class VarFunc(ABC):
    """Abstract base class for variance functions.

    A variance function models heteroscedasticity by defining how the
    standard deviation of observations varies as a function of covariates
    or group membership.

    The variance model is: Var(y_i) = sigma^2 * g(v_i, delta)^2
    where g is the variance function, v_i is a covariate or group indicator,
    and delta are the parameters.

    The weights returned by get_weights() are g(v_i, delta), i.e., the
    standard deviation multipliers.

    Subclasses must implement:
        - get_weights(data, group_indices)
        - n_params (property)
        - _get_init_params(residuals, data)
    """

    def __init__(self) -> None:
        self._params: NDArray | None = None

    @abstractmethod
    def get_weights(self, data: dict, group_indices: NDArray) -> NDArray:
        """Return standard deviation multipliers for each observation in a group.

        Parameters
        ----------
        data : dict
            Data dictionary with relevant columns.
        group_indices : array
            Row indices for this group in the original data.

        Returns
        -------
        weights : array of shape (group_size,)
            Standard deviation multipliers (always positive).
        """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of variance parameters."""

    @abstractmethod
    def _get_init_params(self, residuals: NDArray, data: dict) -> NDArray:
        """Initialize parameters from OLS residuals."""

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
        if self._params is None:
            raise ValueError(
                f"{type(self).__name__} parameters not yet initialized. "
                f"Call initialize() first or fit the model."
            )
        return self._params_to_unconstrained(self._params)

    def set_unconstrained_params(self, uparams: NDArray) -> None:
        """Set parameters from unconstrained (transformed) values."""
        self._params = self._unconstrained_to_params(np.asarray(uparams, dtype=float))

    def _params_to_unconstrained(self, params: NDArray) -> NDArray:
        """Transform natural parameters to unconstrained space."""
        return params.copy()

    def _unconstrained_to_params(self, uparams: NDArray) -> NDArray:
        """Transform unconstrained parameters to natural space."""
        return uparams.copy()

    def initialize(self, residuals: NDArray, data: dict) -> None:
        """Initialize parameters from OLS residuals and data."""
        self._params = self._get_init_params(residuals, data)
