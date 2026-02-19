"""ML and REML log-likelihood functions for GLS estimation.

Implements profile log-likelihood where fixed effects (beta) are profiled out,
leaving only correlation and variance parameters to be optimized.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray


def _build_omega_inv_from_corr_inv(
    corr_inv: NDArray,
    var_weights: NDArray,
) -> NDArray:
    """Build Omega^{-1} block from pre-inverted correlation matrix.

    Returns A^{-1/2} R^{-1} A^{-1/2} (sigma^2 factor omitted since it
    cancels in the profile likelihood).
    """
    safe_weights = np.where(np.abs(var_weights) < 1e-15, 1e-15, var_weights)
    inv_w = 1.0 / safe_weights
    # A_inv_half @ corr_inv @ A_inv_half  =  diag(1/w) @ corr_inv @ diag(1/w)
    return (inv_w[:, None] * corr_inv) * inv_w[None, :]


def _safe_log_weights(var_weights: NDArray) -> float:
    """Compute sum of log(weights) with protection against zero/negative."""
    safe_weights = np.maximum(np.abs(var_weights), 1e-300)
    return float(np.sum(np.log(safe_weights)))


def _accumulate_gls_sums(
    X_groups: list[NDArray],
    y_groups: list[NDArray],
    corr_inverses: list[NDArray],
    corr_logdets: list[float],
    var_weights_groups: list[NDArray],
    n_jobs: int = 1,
) -> tuple[NDArray, NDArray, float, list[NDArray]]:
    """Accumulate GLS sufficient statistics across groups.

    Returns
    -------
    XtOiX : (k, k) accumulated X'Omega^{-1}X
    XtOiy : (k,) accumulated X'Omega^{-1}y
    log_det_omega : float, sum of log|Omega_g| (without sigma^2)
    omega_inv_list : list of per-group Omega^{-1} blocks (cached for RSS)
    """
    G = len(X_groups)
    k = X_groups[0].shape[1]

    # Check if balanced (all groups same size)
    sizes = {Xg.shape[0] for Xg in X_groups}
    balanced = len(sizes) == 1

    if balanced and G > 1:
        m = X_groups[0].shape[0]
        # Stack into 3D arrays for batched matmul
        X_3d = np.stack(X_groups)            # (G, m, k)
        y_3d = np.stack(y_groups)             # (G, m)
        Ri_3d = np.stack(corr_inverses)       # (G, m, m)
        W = np.stack(var_weights_groups)       # (G, m)

        # Build all Omega_inv blocks: diag(1/w) @ R_inv @ diag(1/w)
        safe_W = np.where(np.abs(W) < 1e-15, 1e-15, W)
        inv_W = 1.0 / safe_W                  # (G, m)
        # Omega_inv[g] = inv_W[g,:,None] * Ri_3d[g] * inv_W[g,None,:]
        OmegaInv_3d = (inv_W[:, :, None] * Ri_3d) * inv_W[:, None, :]  # (G, m, m)

        # X'Omega^{-1}X via batched matmul
        OiX = np.matmul(OmegaInv_3d, X_3d)   # (G, m, k)
        XtOiX = np.sum(
            np.matmul(X_3d.transpose(0, 2, 1), OiX), axis=0
        )                                      # (k, k)

        # X'Omega^{-1}y via batched matmul
        Oiy = np.squeeze(
            np.matmul(OmegaInv_3d, y_3d[:, :, None]), axis=-1
        )                                      # (G, m)
        XtOiy = np.sum(
            np.squeeze(
                np.matmul(X_3d.transpose(0, 2, 1), Oiy[:, :, None]),
                axis=-1,
            ),
            axis=0,
        )                                      # (k,)

        # log|Omega| = sum(logdet_R) + 2*sum(log(w))
        log_det_omega = sum(corr_logdets)
        for wg in var_weights_groups:
            log_det_omega += 2 * _safe_log_weights(wg)

        omega_inv_list = [OmegaInv_3d[g] for g in range(G)]
        return XtOiX, XtOiy, log_det_omega, omega_inv_list

    # Unbalanced or single group: sequential or threaded
    def _per_group(args):
        Xg, yg, Ri, logdet_R, wg = args
        Oi = _build_omega_inv_from_corr_inv(Ri, wg)
        xtox = Xg.T @ Oi @ Xg
        xtoy = Xg.T @ Oi @ yg
        ldo = logdet_R + 2 * _safe_log_weights(wg)
        return xtox, xtoy, ldo, Oi

    items = list(zip(X_groups, y_groups, corr_inverses, corr_logdets, var_weights_groups))

    if n_jobs > 1 and G > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_per_group, items))
    else:
        results = [_per_group(item) for item in items]

    XtOiX = np.zeros((k, k))
    XtOiy = np.zeros(k)
    log_det_omega = 0.0
    omega_inv_list = []
    for xtox, xtoy, ldo, Oi in results:
        XtOiX += xtox
        XtOiy += xtoy
        log_det_omega += ldo
        omega_inv_list.append(Oi)

    return XtOiX, XtOiy, log_det_omega, omega_inv_list


def _compute_rss(
    X_groups: list[NDArray],
    y_groups: list[NDArray],
    omega_inv_list: list[NDArray],
    beta_hat: NDArray,
    n_jobs: int = 1,
) -> float:
    """Compute weighted residual sum of squares using cached Omega_inv blocks.

    Returns
    -------
    float : sum_g (y_g - X_g beta)' Omega_inv_g (y_g - X_g beta)
    """
    G = len(X_groups)
    sizes = {Xg.shape[0] for Xg in X_groups}
    balanced = len(sizes) == 1

    if balanced and G > 1:
        m = X_groups[0].shape[0]
        X_3d = np.stack(X_groups)                              # (G, m, k)
        y_3d = np.stack(y_groups)                              # (G, m)
        OmegaInv_3d = np.stack(omega_inv_list)                 # (G, m, m)
        resid_3d = y_3d - np.squeeze(
            np.matmul(X_3d, beta_hat[None, :, None]), axis=-1
        )                                                       # (G, m)
        Oi_resid = np.squeeze(
            np.matmul(OmegaInv_3d, resid_3d[:, :, None]), axis=-1
        )                                                       # (G, m)
        return float(np.sum(resid_3d * Oi_resid))

    # Unbalanced or single group
    def _rss_group(args):
        Xg, yg, Oi = args
        resid = yg - Xg @ beta_hat
        return float(resid @ Oi @ resid)

    items = list(zip(X_groups, y_groups, omega_inv_list))

    if n_jobs > 1 and G > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            rss_parts = list(pool.map(_rss_group, items))
    else:
        rss_parts = [_rss_group(item) for item in items]

    return sum(rss_parts)


def profile_loglik_ml(
    X_groups: list[NDArray],
    y_groups: list[NDArray],
    corr_inverses: list[NDArray],
    corr_logdets: list[float],
    var_weights_groups: list[NDArray],
    nobs: int,
    n_jobs: int = 1,
) -> float:
    """Profile log-likelihood under ML estimation.

    Beta and sigma^2 are profiled out. The returned value is the
    concentrated log-likelihood as a function of correlation and variance
    parameters only.

    Parameters
    ----------
    X_groups : list of (m_g, k) design matrices per group.
    y_groups : list of (m_g,) response vectors per group.
    corr_inverses : list of (m_g, m_g) inverse correlation matrices per group.
    corr_logdets : list of float, log-determinants of correlation matrices.
    var_weights_groups : list of (m_g,) variance weight vectors per group.
    nobs : int, total number of observations.
    n_jobs : int, number of threads (1=sequential).

    Returns
    -------
    float : profile log-likelihood value.
    """
    N = nobs

    XtOiX, XtOiy, log_det_omega, omega_inv_list = _accumulate_gls_sums(
        X_groups, y_groups, corr_inverses, corr_logdets, var_weights_groups, n_jobs
    )

    try:
        beta_hat = np.linalg.solve(XtOiX, XtOiy)
    except np.linalg.LinAlgError:
        return -np.inf

    rss_weighted = _compute_rss(
        X_groups, y_groups, omega_inv_list, beta_hat, n_jobs
    )

    sigma2_hat = rss_weighted / N

    if sigma2_hat <= 0:
        return -np.inf

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
    corr_inverses: list[NDArray],
    corr_logdets: list[float],
    var_weights_groups: list[NDArray],
    nobs: int,
    n_jobs: int = 1,
) -> float:
    """Profile log-likelihood under REML estimation.

    Like ML, but integrates out the fixed effects for unbiased variance
    estimation. The REML adjustment adds -0.5 * log|X'Omega^{-1}X|.

    Parameters
    ----------
    X_groups : list of (m_g, k) design matrices per group.
    y_groups : list of (m_g,) response vectors per group.
    corr_inverses : list of (m_g, m_g) inverse correlation matrices per group.
    corr_logdets : list of float, log-determinants of correlation matrices.
    var_weights_groups : list of (m_g,) variance weight vectors per group.
    nobs : int, total number of observations.
    n_jobs : int, number of threads (1=sequential).

    Returns
    -------
    float : profile REML log-likelihood value.
    """
    k = X_groups[0].shape[1]
    N = nobs

    XtOiX, XtOiy, log_det_omega, omega_inv_list = _accumulate_gls_sums(
        X_groups, y_groups, corr_inverses, corr_logdets, var_weights_groups, n_jobs
    )

    try:
        beta_hat = np.linalg.solve(XtOiX, XtOiy)
    except np.linalg.LinAlgError:
        return -np.inf

    rss_weighted = _compute_rss(
        X_groups, y_groups, omega_inv_list, beta_hat, n_jobs
    )

    N_reml = N - k
    if N_reml <= 0:
        return -np.inf
    sigma2_hat = rss_weighted / N_reml

    if sigma2_hat <= 0:
        return -np.inf

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
    corr_inverses: list[NDArray],
    corr_logdets: list[float],
    var_weights_groups: list[NDArray],
    nobs: int,
    method: str = "REML",
    n_jobs: int = 1,
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

    XtOiX, XtOiy, log_det_omega, omega_inv_list = _accumulate_gls_sums(
        X_groups, y_groups, corr_inverses, corr_logdets, var_weights_groups, n_jobs
    )

    try:
        beta_hat = np.linalg.solve(XtOiX, XtOiy)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "X'Omega^{-1}X is singular. This may indicate perfect collinearity "
            "in the design matrix or a degenerate correlation structure."
        ) from e

    rss_weighted = _compute_rss(
        X_groups, y_groups, omega_inv_list, beta_hat, n_jobs
    )

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
            X_groups, y_groups, corr_inverses, corr_logdets,
            var_weights_groups, nobs, n_jobs
        )
    else:
        loglik = profile_loglik_ml(
            X_groups, y_groups, corr_inverses, corr_logdets,
            var_weights_groups, nobs, n_jobs
        )

    return beta_hat, cov_beta, sigma2_hat, loglik
