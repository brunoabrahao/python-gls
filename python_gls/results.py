"""GLS estimation results."""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats


class GLSResults:
    """Results from a fitted GLS model.

    Provides statsmodels-compatible interface for accessing parameter
    estimates, standard errors, test statistics, and model diagnostics.

    Parameters
    ----------
    model : GLS
        The fitted model instance.
    params : array
        Estimated coefficients.
    cov_params : array
        Covariance matrix of parameter estimates.
    sigma2 : float
        Estimated residual variance (sigma^2).
    loglik : float
        Maximized log-likelihood value.
    method : str
        Estimation method ('ML' or 'REML').
    nobs : int
        Number of observations.
    df_model : int
        Number of estimated parameters (excluding intercept).
    df_resid : int
        Residual degrees of freedom (nobs - k).
    feature_names : list of str
        Names of the features/columns.
    correlation_params : array or None
        Estimated correlation structure parameters.
    variance_params : array or None
        Estimated variance structure parameters.
    n_iter : int
        Number of optimization iterations.
    converged : bool
        Whether the optimization converged.
    """

    def __init__(
        self,
        model,
        params: NDArray,
        cov_params: NDArray,
        sigma2: float,
        loglik: float,
        method: str,
        nobs: int,
        df_model: int,
        df_resid: int,
        feature_names: list[str],
        correlation_params: NDArray | None = None,
        variance_params: NDArray | None = None,
        n_iter: int = 0,
        converged: bool = True,
    ):
        self.model = model
        self._params = np.asarray(params)
        self._cov_params = np.asarray(cov_params)
        self.sigma2 = sigma2
        self.loglik = loglik
        self.method = method
        self.nobs = nobs
        self.df_model = df_model
        self.df_resid = df_resid
        self.feature_names = list(feature_names)
        self.correlation_params = correlation_params
        self.variance_params = variance_params
        self.n_iter = n_iter
        self.converged = converged

    @property
    def params(self) -> pd.Series:
        """Estimated coefficients as a named Series."""
        return pd.Series(self._params, index=self.feature_names, name="params")

    @property
    def bse(self) -> pd.Series:
        """Standard errors of the estimated coefficients."""
        se = np.sqrt(np.diag(self._cov_params))
        return pd.Series(se, index=self.feature_names, name="bse")

    @property
    def tvalues(self) -> pd.Series:
        """t-statistics for the estimated coefficients."""
        return pd.Series(
            self._params / np.sqrt(np.diag(self._cov_params)),
            index=self.feature_names,
            name="tvalues",
        )

    @property
    def pvalues(self) -> pd.Series:
        """Two-sided p-values from t-distribution."""
        t = self.tvalues.values
        p = 2 * stats.t.sf(np.abs(t), df=self.df_resid)
        return pd.Series(p, index=self.feature_names, name="pvalues")

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals for the estimated coefficients.

        Parameters
        ----------
        alpha : float
            Significance level. Default 0.05 gives 95% CI.

        Returns
        -------
        DataFrame with columns ['lower', 'upper'].
        """
        t_crit = stats.t.ppf(1 - alpha / 2, df=self.df_resid)
        se = np.sqrt(np.diag(self._cov_params))
        lower = self._params - t_crit * se
        upper = self._params + t_crit * se
        return pd.DataFrame(
            {"lower": lower, "upper": upper},
            index=self.feature_names,
        )

    @property
    def resid(self) -> NDArray:
        """Residuals (y - X @ params)."""
        return self.model._y - self.model._X @ self._params

    @property
    def fittedvalues(self) -> NDArray:
        """Fitted values (X @ params)."""
        return self.model._X @ self._params

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        k = len(self._params)
        if self.correlation_params is not None:
            k += len(self.correlation_params)
        if self.variance_params is not None:
            k += len(self.variance_params)
        k += 1  # sigma2
        return -2 * self.loglik + 2 * k

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        k = len(self._params)
        if self.correlation_params is not None:
            k += len(self.correlation_params)
        if self.variance_params is not None:
            k += len(self.variance_params)
        k += 1  # sigma2
        return -2 * self.loglik + k * np.log(self.nobs)

    def cov_params_func(self) -> pd.DataFrame:
        """Covariance matrix of estimated coefficients as DataFrame."""
        return pd.DataFrame(
            self._cov_params,
            index=self.feature_names,
            columns=self.feature_names,
        )

    def summary(self, title: str | None = None) -> str:
        """Generate a text summary of the estimation results.

        Parameters
        ----------
        title : str, optional
            Custom title for the summary table.

        Returns
        -------
        str
            Formatted summary string.
        """
        if title is None:
            title = "Generalized Least Squares Results"

        ci = self.conf_int()
        lines = []
        lines.append("=" * 78)
        lines.append(f"{title:^78}")
        lines.append("=" * 78)
        lines.append(f"Method:          {self.method:<20} Log-Likelihood:   {self.loglik:>14.4f}")
        lines.append(f"No. Observations:{self.nobs:<20} AIC:              {self.aic:>14.4f}")
        lines.append(f"Df Model:        {self.df_model:<20} BIC:              {self.bic:>14.4f}")
        lines.append(f"Df Residuals:    {self.df_resid:<20} Sigma^2:          {self.sigma2:>14.6f}")
        lines.append(f"Converged:       {'Yes' if self.converged else 'No':<20} Iterations:       {self.n_iter:>14}")
        lines.append("-" * 78)

        # Coefficient table
        header = f"{'':>20} {'coef':>10} {'std err':>10} {'t':>10} {'P>|t|':>10} {'[0.025':>10} {'0.975]':>10}"
        lines.append(header)
        lines.append("-" * 78)
        for name in self.feature_names:
            coef = self._params[self.feature_names.index(name)]
            se = self.bse[name]
            t = self.tvalues[name]
            p = self.pvalues[name]
            lo = ci.loc[name, "lower"]
            hi = ci.loc[name, "upper"]
            lines.append(
                f"{name:>20} {coef:>10.4f} {se:>10.4f} {t:>10.4f} {p:>10.4f} {lo:>10.4f} {hi:>10.4f}"
            )

        lines.append("=" * 78)

        if self.correlation_params is not None:
            lines.append(f"Correlation Structure: {type(self.model.correlation).__name__}")
            lines.append(f"  Parameters: {self.correlation_params}")

        if self.variance_params is not None:
            lines.append(f"Variance Function: {type(self.model.variance).__name__}")
            lines.append(f"  Parameters: {self.variance_params}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
