"""ML and REML log-likelihood functions for GLS estimation.

Implements profile log-likelihood where fixed effects (beta) are profiled out,
leaving only correlation and variance parameters to be optimized.
"""

import warnings

import numpy as np
from numpy.typing import NDArray


def _build_omega_block(
    corr_matrix: NDArray,
    var_weights: NDArray,
    sigma2: float,
) -> NDArray:
    """Build covariance block: sigma^2 * A^{1/2} R A^{1/2}.

    Parameters
    ----------
    corr_matrix : (m, m) correlation matrix for this group.
    var_weights : (m,) variance weights for this group (standard deviations).
    sigma2 : scalar residual variance.

    Returns
    -------
    Omega block of shape (m, m).
    """
    A_half = np.diag(var_weights)
    return sigma2 * A_half @ corr_matrix @ A_half


def _build_omega_inv_block(
    corr_matrix: NDArray,
    var_weights: NDArray,
) -> NDArray:
    """Build Omega^{-1} block (up to 1/sigma^2 scaling).

    Returns (1/sigma^2) * A^{-1/2} R^{-1} A^{-1/2}.
    We drop the sigma^2 factor since it cancels in the profile likelihood.
    """
    # Guard against zero/near-zero weights
    safe_weights = np.where(np.abs(var_weights) < 1e-15, 1e-15, var_weights)
    A_inv_half = np.diag(1.0 / safe_weights)
    R_inv = np.linalg.solve(corr_matrix, np.eye(corr_matrix.shape[0]))
    return A_inv_half @ R_inv @ A_inv_half


def _safe_log_weights(var_weights: NDArray) -> float:
    """Compute sum of log(weights) with protection against zero/negative."""
    safe_weights = np.maximum(np.abs(var_weights), 1e-300)
    return float(np.sum(np.log(safe_weights)))


def profile_loglik_ml(
    X_groups: list[NDArray],
    y_groups: list[NDArray],
    corr_matrices: list[NDArray],
    var_weights_groups: list[NDArray],
    nobs: int,
) -> float:
    """Profile log-likelihood under ML estimation.

    Beta and sigma^2 are profiled out. The returned value is the
    concentrated log-likelihood as a function of correlation and variance
    parameters only.

    Parameters
    ----------
    X_groups : list of (m_g, k) design matrices per group.
    y_groups : list of (m_g,) response vectors per group.
    corr_matrices : list of (m_g, m_g) correlation matrices per group.
    var_weights_groups : list of (m_g,) variance weight vectors per group.
    nobs : int, total number of observations.

    Returns
    -------
    float : profile log-likelihood value.
    """
    k = X_groups[0].shape[1]
    N = nobs

    # Accumulate X'Omega^{-1}X and X'Omega^{-1}y and log|Omega| across groups
    XtOiX = np.zeros((k, k))
    XtOiy = np.zeros(k)
    log_det_omega = 0.0

    for Xg, yg, Rg, wg in zip(X_groups, y_groups, corr_matrices, var_weights_groups):
        Omega_inv = _build_omega_inv_block(Rg, wg)
        XtOiX += Xg.T @ Omega_inv @ Xg
        XtOiy += Xg.T @ Omega_inv @ yg

        # log|Omega_g| = log|R_g| + 2*sum(log(w_g)) (sigma^2 factor added later)
        sign, logdet_R = np.linalg.slogdet(Rg)
        if sign <= 0:
            return -np.inf
        log_det_omega += logdet_R + 2 * _safe_log_weights(wg)

    # Profile beta: beta_hat = (X'Omega^{-1}X)^{-1} X'Omega^{-1}y
    try:
        beta_hat = np.linalg.solve(XtOiX, XtOiy)
    except np.linalg.LinAlgError:
        return -np.inf

    # Profile sigma^2: sigma^2_hat = (1/N) * sum_g (y_g - X_g beta)' Omega_inv_g (y_g - X_g beta)
    rss_weighted = 0.0
    for Xg, yg, Rg, wg in zip(X_groups, y_groups, corr_matrices, var_weights_groups):
        Omega_inv = _build_omega_inv_block(Rg, wg)
        resid = yg - Xg @ beta_hat
        rss_weighted += resid @ Omega_inv @ resid

    sigma2_hat = rss_weighted / N

    if sigma2_hat <= 0:
        return -np.inf

    # Log-likelihood: -N/2 * log(2*pi) - N/2 * log(sigma^2) - 1/2 * log|Omega/sigma^2| - N/2
    # where log|Omega| = N*log(sigma^2) + log_det_omega (without sigma^2)
    loglik = (
        -0.5 * N * np.log(2 * np.pi)
        - 0.5 * N * np.log(sigma2_hat)
        - 0.5 * log_det_omega
        - 0.5 * N
    )

    if np.isnan(loglik) or np.isinf(loglik):
        return -np.inf

    return loglik


def profile_loglik_reml(
    X_groups: list[NDArray],
    y_groups: list[NDArray],
    corr_matrices: list[NDArray],
    var_weights_groups: list[NDArray],
    nobs: int,
) -> float:
    """Profile log-likelihood under REML estimation.

    Like ML, but integrates out the fixed effects for unbiased variance
    estimation. The REML adjustment adds -0.5 * log|X'Omega^{-1}X|.

    Parameters
    ----------
    X_groups : list of (m_g, k) design matrices per group.
    y_groups : list of (m_g,) response vectors per group.
    corr_matrices : list of (m_g, m_g) correlation matrices per group.
    var_weights_groups : list of (m_g,) variance weight vectors per group.
    nobs : int, total number of observations.

    Returns
    -------
    float : profile REML log-likelihood value.
    """
    k = X_groups[0].shape[1]
    N = nobs

    XtOiX = np.zeros((k, k))
    XtOiy = np.zeros(k)
    log_det_omega = 0.0

    for Xg, yg, Rg, wg in zip(X_groups, y_groups, corr_matrices, var_weights_groups):
        Omega_inv = _build_omega_inv_block(Rg, wg)
        XtOiX += Xg.T @ Omega_inv @ Xg
        XtOiy += Xg.T @ Omega_inv @ yg

        sign, logdet_R = np.linalg.slogdet(Rg)
        if sign <= 0:
            return -np.inf
        log_det_omega += logdet_R + 2 * _safe_log_weights(wg)

    try:
        beta_hat = np.linalg.solve(XtOiX, XtOiy)
    except np.linalg.LinAlgError:
        return -np.inf

    # REML uses N-k for sigma^2
    rss_weighted = 0.0
    for Xg, yg, Rg, wg in zip(X_groups, y_groups, corr_matrices, var_weights_groups):
        Omega_inv = _build_omega_inv_block(Rg, wg)
        resid = yg - Xg @ beta_hat
        rss_weighted += resid @ Omega_inv @ resid

    N_reml = N - k
    if N_reml <= 0:
        return -np.inf
    sigma2_hat = rss_weighted / N_reml

    if sigma2_hat <= 0:
        return -np.inf

    # REML log-likelihood
    sign_xtox, logdet_xtox = np.linalg.slogdet(XtOiX)
    if sign_xtox <= 0:
        return -np.inf

    loglik = (
        -0.5 * N_reml * np.log(2 * np.pi)
        - 0.5 * N_reml * np.log(sigma2_hat)
        - 0.5 * log_det_omega
        - 0.5 * logdet_xtox
        - 0.5 * N_reml
    )

    if np.isnan(loglik) or np.isinf(loglik):
        return -np.inf

    return loglik


def compute_gls_estimates(
    X_groups: list[NDArray],
    y_groups: list[NDArray],
    corr_matrices: list[NDArray],
    var_weights_groups: list[NDArray],
    nobs: int,
    method: str = "REML",
) -> tuple[NDArray, NDArray, float, float]:
    """Compute GLS beta, covariance, sigma^2, and log-likelihood.

    Given the estimated correlation and variance parameters, compute the
    final GLS estimates.

    Returns
    -------
    beta_hat : (k,) estimated coefficients
    cov_beta : (k, k) covariance of beta estimates
    sigma2_hat : estimated residual variance
    loglik : log-likelihood at these estimates
    """
    k = X_groups[0].shape[1]
    N = nobs

    XtOiX = np.zeros((k, k))
    XtOiy = np.zeros(k)
    log_det_omega = 0.0

    for Xg, yg, Rg, wg in zip(X_groups, y_groups, corr_matrices, var_weights_groups):
        Omega_inv = _build_omega_inv_block(Rg, wg)
        XtOiX += Xg.T @ Omega_inv @ Xg
        XtOiy += Xg.T @ Omega_inv @ yg

        sign, logdet_R = np.linalg.slogdet(Rg)
        log_det_omega += logdet_R + 2 * _safe_log_weights(wg)

    try:
        beta_hat = np.linalg.solve(XtOiX, XtOiy)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "X'Omega^{-1}X is singular. This may indicate perfect collinearity "
            "in the design matrix or a degenerate correlation structure."
        ) from e

    rss_weighted = 0.0
    for Xg, yg, Rg, wg in zip(X_groups, y_groups, corr_matrices, var_weights_groups):
        Omega_inv = _build_omega_inv_block(Rg, wg)
        resid = yg - Xg @ beta_hat
        rss_weighted += resid @ Omega_inv @ resid

    if method == "REML":
        denom = N - k
        if denom <= 0:
            warnings.warn(
                f"N - k = {denom} <= 0 for REML. Using ML denominator (N={N}) instead.",
                stacklevel=2,
            )
            denom = N
        sigma2_hat = rss_weighted / denom
    else:
        sigma2_hat = rss_weighted / N

    # Covariance of beta: sigma^2 * (X'Omega^{-1}X)^{-1}
    try:
        cov_beta = sigma2_hat * np.linalg.inv(XtOiX)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Failed to invert X'Omega^{-1}X for covariance estimation. "
            "The design matrix may be singular or near-singular."
        ) from e

    # Warn if covariance has issues
    cov_diag = np.diag(cov_beta)
    if np.any(cov_diag < 0):
        warnings.warn(
            "Some variance estimates are negative, indicating numerical instability. "
            "Standard errors may be unreliable.",
            stacklevel=2,
        )

    # Compute log-likelihood at these estimates
    if method == "REML":
        loglik = profile_loglik_reml(
            X_groups, y_groups, corr_matrices, var_weights_groups, nobs
        )
    else:
        loglik = profile_loglik_ml(
            X_groups, y_groups, corr_matrices, var_weights_groups, nobs
        )

    return beta_hat, cov_beta, sigma2_hat, loglik
