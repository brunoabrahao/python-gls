"""GLS model class -- main entry point for the library.

Implements Generalized Least Squares with learned correlation and variance
structures, equivalent to R's nlme::gls().
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from python_gls.correlation.base import CorStruct
from python_gls.variance.base import VarFunc
from python_gls.likelihood import (
    profile_loglik_ml,
    profile_loglik_reml,
    compute_gls_estimates,
)
from python_gls.results import GLSResults


def _validate_array(arr: NDArray, name: str) -> None:
    """Check an array for NaN and Inf values."""
    if np.any(np.isnan(arr)):
        n_nan = int(np.sum(np.isnan(arr)))
        raise ValueError(
            f"{name} contains {n_nan} NaN value(s). "
            f"Remove or impute missing values before fitting."
        )
    if np.any(np.isinf(arr)):
        n_inf = int(np.sum(np.isinf(arr)))
        raise ValueError(
            f"{name} contains {n_inf} infinite value(s). "
            f"Check for overflow or division by zero in your data."
        )


class GLS:
    """Generalized Least Squares with learned correlation and variance structures.

    Equivalent to R's ``nlme::gls()``. Estimates fixed effects along with
    correlation and variance parameters via maximum likelihood (ML) or
    restricted maximum likelihood (REML).

    Parameters
    ----------
    formula : str or None
        R-style formula (e.g., ``"y ~ x1 + x2"``). Use ``from_formula()``
        for formula-based construction.
    data : DataFrame or None
        Data for formula-based construction.
    endog : array-like or None
        Response variable (if not using formula).
    exog : array-like or None
        Design matrix (if not using formula). Should include intercept column.
    correlation : CorStruct or None
        Correlation structure. If None, assumes independence.
    variance : VarFunc or None
        Variance function. If None, assumes homoscedasticity.
    groups : str or array-like or None
        Grouping variable name (str) or array of group labels.
        Required if correlation is specified.
    method : str
        Estimation method: ``'REML'`` (default) or ``'ML'``.

    Examples
    --------
    >>> from python_gls import GLS
    >>> from python_gls.correlation import CorSymm
    >>> from python_gls.variance import VarIdent
    >>>
    >>> result = GLS.from_formula(
    ...     "y ~ x1 + x2",
    ...     data=df,
    ...     correlation=CorSymm(),
    ...     variance=VarIdent("group"),
    ...     groups="subject",
    ... ).fit()
    >>> print(result.summary())
    """

    def __init__(
        self,
        endog: NDArray | None = None,
        exog: NDArray | None = None,
        correlation: CorStruct | None = None,
        variance: VarFunc | None = None,
        groups: NDArray | str | None = None,
        data: pd.DataFrame | None = None,
        method: str = "REML",
    ) -> None:
        if correlation is not None and not isinstance(correlation, CorStruct):
            raise TypeError(
                f"correlation must be a CorStruct instance, got {type(correlation).__name__}. "
                f"Use one of: CorAR1(), CorCompSymm(), CorSymm(), etc."
            )
        if variance is not None and not isinstance(variance, VarFunc):
            raise TypeError(
                f"variance must be a VarFunc instance, got {type(variance).__name__}. "
                f"Use one of: VarIdent(), VarPower(), VarExp(), etc."
            )
        if not isinstance(method, str):
            raise TypeError(f"method must be a string, got {type(method).__name__}")

        self.correlation = correlation
        self.variance = variance
        self.method = method.upper()
        if self.method not in ("ML", "REML"):
            raise ValueError(
                f"method must be 'ML' or 'REML', got '{method}'"
            )

        self._X: NDArray | None = None
        self._y: NDArray | None = None
        self._groups: NDArray | None = None
        self._data: dict | None = None
        self._feature_names: list[str] = []
        self._formula: str | None = None

        if endog is not None and exog is not None:
            self._y = np.asarray(endog, dtype=float).ravel()
            self._X = np.asarray(exog, dtype=float)
            if self._X.ndim == 1:
                self._X = self._X[:, None]

            _validate_array(self._y, "endog")
            _validate_array(self._X, "exog")

            if len(self._y) != self._X.shape[0]:
                raise ValueError(
                    f"endog and exog have incompatible shapes: "
                    f"endog has {len(self._y)} observations but exog has "
                    f"{self._X.shape[0]} rows"
                )
            if len(self._y) == 0:
                raise ValueError("endog and exog must not be empty")
            if self._X.shape[1] > self._X.shape[0]:
                warnings.warn(
                    f"More predictors ({self._X.shape[1]}) than observations "
                    f"({self._X.shape[0]}). Model may be unidentifiable.",
                    stacklevel=2,
                )

            self._feature_names = [f"x{i}" for i in range(self._X.shape[1])]
        elif (endog is None) != (exog is None):
            raise ValueError(
                "Both endog and exog must be provided together, or neither. "
                "Got endog={} and exog={}".format(
                    "provided" if endog is not None else "None",
                    "provided" if exog is not None else "None",
                )
            )

        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(
                    f"data must be a pandas DataFrame, got {type(data).__name__}"
                )
            self._data = {col: np.asarray(data[col]) for col in data.columns}

        if groups is not None:
            if isinstance(groups, str):
                if data is not None:
                    if groups not in data.columns:
                        raise ValueError(
                            f"groups column '{groups}' not found in data. "
                            f"Available columns: {list(data.columns)}"
                        )
                    self._groups = np.asarray(data[groups])
                elif self._data is not None:
                    if groups not in self._data:
                        raise ValueError(
                            f"groups column '{groups}' not found in data. "
                            f"Available columns: {list(self._data.keys())}"
                        )
                    self._groups = np.asarray(self._data[groups])
                else:
                    raise ValueError(
                        f"groups='{groups}' is a column name but no data was provided. "
                        f"Pass data= or provide groups as an array."
                    )
            else:
                self._groups = np.asarray(groups)
                if self._y is not None and len(self._groups) != len(self._y):
                    raise ValueError(
                        f"groups array length ({len(self._groups)}) does not match "
                        f"number of observations ({len(self._y)})"
                    )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        correlation: CorStruct | None = None,
        variance: VarFunc | None = None,
        groups: str | None = None,
        method: str = "REML",
    ) -> GLS:
        """Construct a GLS model from an R-style formula.

        Parameters
        ----------
        formula : str
            R-style formula, e.g., ``"y ~ x1 + x2"`` or ``"y ~ C(treatment) * time"``.
        data : DataFrame
            Data containing the variables referenced in the formula.
        correlation : CorStruct, optional
            Correlation structure.
        variance : VarFunc, optional
            Variance function.
        groups : str, optional
            Name of the grouping variable in ``data``.
        method : str
            ``'REML'`` or ``'ML'``.

        Returns
        -------
        GLS
            Model instance ready for ``.fit()``.
        """
        if not isinstance(formula, str):
            raise TypeError(
                f"formula must be a string, got {type(formula).__name__}"
            )
        if "~" not in formula:
            raise ValueError(
                f"formula must contain '~' separating response and predictors, "
                f"e.g. 'y ~ x1 + x2'. Got: '{formula}'"
            )
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(data).__name__}"
            )
        if len(data) == 0:
            raise ValueError("data must not be empty")
        if groups is not None and not isinstance(groups, str):
            raise TypeError(
                f"groups must be a column name (string), got {type(groups).__name__}"
            )

        import formulaic

        model_spec = formulaic.model_matrix(formula, data)

        # formulaic returns ModelMatrices with .lhs and .rhs attributes
        if hasattr(model_spec, "lhs"):
            y = np.asarray(model_spec.lhs).ravel()
            X_mm = model_spec.rhs
            X = np.asarray(X_mm, dtype=float)
            feature_names = list(X_mm.columns)
        elif isinstance(model_spec, tuple) and len(model_spec) == 2:
            y = np.asarray(model_spec[0]).ravel()
            X_mm = model_spec[1]
            X = np.asarray(X_mm, dtype=float)
            feature_names = list(X_mm.columns)
        else:
            X_mm = model_spec
            X = np.asarray(X_mm, dtype=float)
            feature_names = list(X_mm.columns)
            lhs = formula.split("~")[0].strip()
            y = np.asarray(data[lhs], dtype=float).ravel()

        obj = cls(
            endog=y,
            exog=X,
            correlation=correlation,
            variance=variance,
            groups=groups,
            data=data,
            method=method,
        )
        obj._feature_names = feature_names
        obj._formula = formula
        return obj

    def _split_by_groups(self) -> tuple[list[NDArray], list[NDArray], list[NDArray]]:
        """Split X, y, and indices by group.

        Returns
        -------
        X_groups : list of (m_g, k) arrays
        y_groups : list of (m_g,) arrays
        idx_groups : list of index arrays (row indices into original data)
        """
        if self._groups is None:
            return [self._X], [self._y], [np.arange(len(self._y))]

        unique_groups = np.unique(self._groups)
        X_groups = []
        y_groups = []
        idx_groups = []
        for g in unique_groups:
            mask = self._groups == g
            idx = np.where(mask)[0]
            X_groups.append(self._X[idx])
            y_groups.append(self._y[idx])
            idx_groups.append(idx)
        return X_groups, y_groups, idx_groups

    def _get_corr_matrices(
        self, group_sizes: list[int]
    ) -> list[NDArray]:
        """Get correlation matrices for each group."""
        if self.correlation is None:
            return [np.eye(s) for s in group_sizes]
        return [
            self.correlation.get_correlation_matrix(s, group_id=i)
            for i, s in enumerate(group_sizes)
        ]

    def _get_corr_inverses(
        self, group_sizes: list[int]
    ) -> list[NDArray]:
        """Get inverse correlation matrices for each group."""
        if self.correlation is None:
            return [np.eye(s) for s in group_sizes]
        return [
            self.correlation.get_correlation_matrix_inverse(s, group_id=i)
            for i, s in enumerate(group_sizes)
        ]

    def _get_corr_logdets(
        self, group_sizes: list[int]
    ) -> list[float]:
        """Get log-determinants of correlation matrices for each group."""
        if self.correlation is None:
            return [0.0 for _ in group_sizes]
        return [
            self.correlation.get_log_determinant(s, group_id=i)
            for i, s in enumerate(group_sizes)
        ]

    def _get_var_weights(
        self, idx_groups: list[NDArray]
    ) -> list[NDArray]:
        """Get variance weights for each group."""
        if self.variance is None:
            return [np.ones(len(idx)) for idx in idx_groups]
        return [
            self.variance.get_weights(self._data, idx)
            for idx in idx_groups
        ]

    def fit(
        self,
        maxiter: int = 200,
        tol: float = 1e-8,
        verbose: bool = False,
        n_jobs: int = 1,
    ) -> GLSResults:
        """Fit the GLS model.

        Parameters
        ----------
        maxiter : int
            Maximum number of optimization iterations.
        tol : float
            Convergence tolerance.
        verbose : bool
            If True, print optimization progress.
        n_jobs : int
            Number of threads for parallel computation. Use 1 for sequential
            (default, zero overhead). Use -1 to use all available CPU cores.
            Threading helps most with unbalanced panels; balanced panels
            already use batched NumPy operations. For BLAS-level parallelism,
            set ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, or
            ``OPENBLAS_NUM_THREADS`` environment variables.

        Returns
        -------
        GLSResults
            Fitted model results.
        """
        if self._X is None or self._y is None:
            raise ValueError(
                "No data provided. Use GLS.from_formula('y ~ x', data=df) "
                "or pass endog= and exog= to the constructor."
            )

        # Resolve n_jobs
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        elif n_jobs < 1:
            n_jobs = 1

        N = len(self._y)
        k = self._X.shape[1]

        if N <= k:
            warnings.warn(
                f"Number of observations ({N}) is not greater than the number "
                f"of predictors ({k}). Estimates may be unreliable.",
                stacklevel=2,
            )

        if self.variance is not None and self._data is None:
            raise ValueError(
                "A variance function was specified but no data dictionary is available. "
                "Use GLS.from_formula() or pass data= to the constructor."
            )

        # Step 1: OLS initial fit
        beta_ols = np.linalg.lstsq(self._X, self._y, rcond=None)[0]
        residuals_ols = self._y - self._X @ beta_ols

        X_groups, y_groups, idx_groups = self._split_by_groups()
        group_sizes = [len(yg) for yg in y_groups]

        # If no correlation or variance structure, just do OLS/GLS with identity
        if self.correlation is None and self.variance is None:
            corr_inverses = [np.eye(s) for s in group_sizes]
            corr_logdets = [0.0 for _ in group_sizes]
            var_weights = [np.ones(s) for s in group_sizes]

            beta_hat, cov_beta, sigma2_hat, loglik = compute_gls_estimates(
                X_groups, y_groups, corr_inverses, corr_logdets,
                var_weights, N, self.method, n_jobs
            )

            return GLSResults(
                model=self,
                params=beta_hat,
                cov_params=cov_beta,
                sigma2=sigma2_hat,
                loglik=loglik,
                method=self.method,
                nobs=N,
                df_model=k - 1,
                df_resid=N - k,
                feature_names=self._feature_names,
                n_iter=0,
                converged=True,
            )

        # Step 2: Initialize correlation and variance params from OLS residuals
        residuals_by_group = [residuals_ols[idx] for idx in idx_groups]

        if self.correlation is not None:
            self.correlation.initialize(residuals_by_group)

        if self.variance is not None:
            if self._data is None:
                raise ValueError("Data dictionary required for variance functions.")
            self.variance.initialize(residuals_ols, self._data)

        # Step 3: Optimize profile log-likelihood
        def _pack_params() -> NDArray:
            parts = []
            if self.correlation is not None and self.correlation.n_params > 0:
                parts.append(self.correlation.get_unconstrained_params())
            if self.variance is not None and self.variance.n_params > 0:
                parts.append(self.variance.get_unconstrained_params())
            return np.concatenate(parts) if parts else np.array([])

        def _unpack_params(theta: NDArray) -> None:
            idx = 0
            if self.correlation is not None and self.correlation.n_params > 0:
                n_corr = self.correlation.n_params
                self.correlation.set_unconstrained_params(theta[idx : idx + n_corr])
                idx += n_corr
            if self.variance is not None and self.variance.n_params > 0:
                n_var = self.variance.n_params
                self.variance.set_unconstrained_params(theta[idx : idx + n_var])
                idx += n_var

        loglik_func = (
            profile_loglik_reml if self.method == "REML" else profile_loglik_ml
        )

        n_eval = [0]

        def neg_loglik(theta: NDArray) -> float:
            _unpack_params(theta)
            try:
                corr_inverses = self._get_corr_inverses(group_sizes)
                corr_logdets = self._get_corr_logdets(group_sizes)
                var_weights = self._get_var_weights(idx_groups)
                ll = loglik_func(
                    X_groups, y_groups, corr_inverses, corr_logdets,
                    var_weights, N, n_jobs
                )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return 1e15
            n_eval[0] += 1
            if verbose and n_eval[0] % 10 == 0:
                print(f"  Iteration {n_eval[0]}: loglik = {ll:.4f}")
            if np.isnan(ll) or np.isinf(ll):
                return 1e15
            return -ll

        theta0 = _pack_params()

        if len(theta0) > 0:
            result = minimize(
                neg_loglik,
                theta0,
                method="L-BFGS-B",
                options={"maxiter": maxiter, "ftol": tol, "disp": verbose},
            )
            _unpack_params(result.x)
            converged = result.success
            n_iter = result.nit
            if not converged:
                warnings.warn(
                    f"Optimization did not converge after {n_iter} iterations: "
                    f"{result.message}. Results may be unreliable.",
                    stacklevel=2,
                )
        else:
            converged = True
            n_iter = 0

        # Step 4: Compute final estimates at converged parameters
        corr_inverses = self._get_corr_inverses(group_sizes)
        corr_logdets = self._get_corr_logdets(group_sizes)
        var_weights = self._get_var_weights(idx_groups)

        beta_hat, cov_beta, sigma2_hat, loglik = compute_gls_estimates(
            X_groups, y_groups, corr_inverses, corr_logdets,
            var_weights, N, self.method, n_jobs
        )

        # Collect estimated parameters
        corr_params = (
            self.correlation.get_params()
            if self.correlation is not None and self.correlation.n_params > 0
            else None
        )
        var_params = (
            self.variance.get_params()
            if self.variance is not None and self.variance.n_params > 0
            else None
        )

        return GLSResults(
            model=self,
            params=beta_hat,
            cov_params=cov_beta,
            sigma2=sigma2_hat,
            loglik=loglik,
            method=self.method,
            nobs=N,
            df_model=k - 1,
            df_resid=N - k,
            feature_names=self._feature_names,
            correlation_params=corr_params,
            variance_params=var_params,
            n_iter=n_iter,
            converged=converged,
        )
