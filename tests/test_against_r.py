"""Validation tests: verify python_gls against known-correct results.

These tests validate statistical correctness by checking:
  a. OLS exact match with statsmodels (coefficients AND standard errors)
  b. ML vs REML sigma2 ordering (REML >= ML, same coefficients)
  c. AIC/BIC formula correctness (manual computation)
  d. Confidence interval coverage (Monte Carlo, ~95% of 95% CIs cover true value)
  e. AR(1) phi recovery (known phi -> estimated phi close)
  f. VarIdent variance ratio recovery (known heteroscedasticity)
  g. Likelihood improvement from correct correlation structure
  h. Coefficient consistency with large samples (within 3 SE of truth)
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from python_gls import GLS
from python_gls.correlation import CorAR1, CorCompSymm
from python_gls.variance import VarIdent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ar1_panel(
    n_subjects: int,
    n_times: int,
    phi: float,
    beta: np.ndarray,
    sigma: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate panel data with AR(1) errors.

    Parameters
    ----------
    n_subjects : number of subjects/groups
    n_times : observations per subject
    phi : AR(1) autocorrelation parameter
    beta : true coefficients [intercept, slope1, ...]
    sigma : error standard deviation
    seed : random seed
    """
    rng = np.random.RandomState(seed)
    N = n_subjects * n_times
    k = len(beta) - 1  # number of predictors (excluding intercept)

    subjects = np.repeat(np.arange(n_subjects), n_times)
    # Build design matrix: intercept + k predictors
    X_pred = rng.randn(N, k)
    X_full = np.column_stack([np.ones(N), X_pred])

    # AR(1) errors within each subject
    errors = np.zeros(N)
    innovation_sd = sigma * np.sqrt(1 - phi ** 2) if abs(phi) < 1 else sigma
    for s in range(n_subjects):
        idx = slice(s * n_times, (s + 1) * n_times)
        e = np.zeros(n_times)
        e[0] = rng.randn() * sigma
        for t in range(1, n_times):
            e[t] = phi * e[t - 1] + rng.randn() * innovation_sd
        errors[idx] = e

    y = X_full @ beta + errors

    df = pd.DataFrame({"y": y, "subject": subjects})
    for j in range(k):
        df[f"x{j + 1}"] = X_pred[:, j]
    return df


def make_hetero_panel(
    n_subjects_per_group: int,
    n_times: int,
    beta: np.ndarray,
    sigma_a: float,
    sigma_b: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate panel data with group-level heteroscedasticity.

    Group "A" has error std = sigma_a, group "B" has error std = sigma_b.
    """
    rng = np.random.RandomState(seed)
    n_subjects = 2 * n_subjects_per_group
    N = n_subjects * n_times
    k = len(beta) - 1

    subjects = np.repeat(np.arange(n_subjects), n_times)
    group = np.where(subjects < n_subjects_per_group, "A", "B")
    X_pred = rng.randn(N, k)
    X_full = np.column_stack([np.ones(N), X_pred])

    sigma_vec = np.where(group == "A", sigma_a, sigma_b)
    errors = rng.randn(N) * sigma_vec
    y = X_full @ beta + errors

    df = pd.DataFrame({"y": y, "subject": subjects, "group": group})
    for j in range(k):
        df[f"x{j + 1}"] = X_pred[:, j]
    return df


# ===========================================================================
# (a) OLS EXACT MATCH
# ===========================================================================

class TestOLSExactMatch:
    """GLS with no correlation/variance must match statsmodels OLS exactly."""

    def test_coefficients_machine_precision(self):
        """Coefficients must match to <1e-10."""
        rng = np.random.RandomState(42)
        N = 300
        x1 = rng.randn(N)
        x2 = rng.randn(N)
        y = 5.0 + 2.0 * x1 - 3.0 * x2 + rng.randn(N) * 0.8
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        r = GLS.from_formula("y ~ x1 + x2", data=df, method="REML").fit()
        X_sm = sm.add_constant(df[["x1", "x2"]])
        ols = sm.OLS(y, X_sm).fit()

        np.testing.assert_allclose(r.params.values, ols.params.values, atol=1e-10)

    def test_standard_errors_machine_precision(self):
        """Standard errors under REML must match OLS exactly.

        OLS uses sigma2 = RSS/(N-k), which is the REML estimator.
        """
        rng = np.random.RandomState(42)
        N = 300
        x1 = rng.randn(N)
        x2 = rng.randn(N)
        y = 5.0 + 2.0 * x1 - 3.0 * x2 + rng.randn(N) * 0.8
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        r = GLS.from_formula("y ~ x1 + x2", data=df, method="REML").fit()
        X_sm = sm.add_constant(df[["x1", "x2"]])
        ols = sm.OLS(y, X_sm).fit()

        np.testing.assert_allclose(r.bse.values, ols.bse.values, atol=1e-10)

    def test_sigma2_reml_equals_ols(self):
        """REML sigma2 = RSS/(N-k) = OLS MSE."""
        rng = np.random.RandomState(99)
        N = 150
        x = rng.randn(N)
        y = 1.0 + 0.5 * x + rng.randn(N) * 2.0
        df = pd.DataFrame({"y": y, "x": x})

        r = GLS.from_formula("y ~ x", data=df, method="REML").fit()
        X_sm = sm.add_constant(df[["x"]])
        ols = sm.OLS(y, X_sm).fit()

        np.testing.assert_allclose(r.sigma2, ols.mse_resid, atol=1e-10)


# ===========================================================================
# (b) ML vs REML
# ===========================================================================

class TestMLvsREML:
    """REML sigma2 >= ML sigma2, both produce same coefficients."""

    def test_reml_sigma2_larger(self):
        """REML uses N-k denominator so sigma2_reml > sigma2_ml."""
        df = make_ar1_panel(60, 5, phi=0.5, beta=np.array([1.0, 2.0]), seed=10)

        r_ml = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject", method="ML"
        ).fit()
        r_reml = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject", method="REML"
        ).fit()

        assert r_reml.sigma2 > r_ml.sigma2, (
            f"REML sigma2 ({r_reml.sigma2:.6f}) should exceed "
            f"ML sigma2 ({r_ml.sigma2:.6f})"
        )

    def test_same_coefficients(self):
        """ML and REML should converge to very similar beta."""
        df = make_ar1_panel(60, 5, phi=0.5, beta=np.array([1.0, 2.0]), seed=10)

        r_ml = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject", method="ML"
        ).fit()
        r_reml = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject", method="REML"
        ).fit()

        np.testing.assert_allclose(
            r_ml.params.values, r_reml.params.values, atol=0.05,
            err_msg="ML and REML coefficients should be very close"
        )


# ===========================================================================
# (c) AIC / BIC FORMULA
# ===========================================================================

class TestAICBICFormula:
    """Verify AIC = -2*loglik + 2*k and BIC = -2*loglik + k*log(n)."""

    def test_aic_formula_ols(self):
        """OLS case: k = n_coef + 1 (sigma2)."""
        rng = np.random.RandomState(7)
        N = 100
        x = rng.randn(N)
        y = 1.0 + x + rng.randn(N)
        df = pd.DataFrame({"y": y, "x": x})

        r = GLS.from_formula("y ~ x", data=df).fit()
        k = 2 + 1  # intercept + slope + sigma2
        expected_aic = -2 * r.loglik + 2 * k
        np.testing.assert_allclose(r.aic, expected_aic, atol=1e-10)

    def test_bic_formula_ols(self):
        """BIC = -2*loglik + k*log(n)."""
        rng = np.random.RandomState(7)
        N = 100
        x = rng.randn(N)
        y = 1.0 + x + rng.randn(N)
        df = pd.DataFrame({"y": y, "x": x})

        r = GLS.from_formula("y ~ x", data=df).fit()
        k = 2 + 1
        expected_bic = -2 * r.loglik + k * np.log(N)
        np.testing.assert_allclose(r.bic, expected_bic, atol=1e-10)

    def test_aic_with_correlation(self):
        """With AR(1), k includes 1 correlation parameter."""
        df = make_ar1_panel(40, 5, phi=0.4, beta=np.array([1.0, 0.5]), seed=3)

        r = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        # k = 2 coef + 1 corr param + 1 sigma2 = 4
        k = 4
        expected_aic = -2 * r.loglik + 2 * k
        np.testing.assert_allclose(r.aic, expected_aic, atol=1e-10)

    def test_bic_with_variance(self):
        """With VarIdent (2 groups), k includes 1 variance parameter."""
        df = make_hetero_panel(20, 4, beta=np.array([1.0, 0.5]), sigma_a=1.0, sigma_b=2.0, seed=5)
        N = len(df)

        r = GLS.from_formula(
            "y ~ x1", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()
        # k = 2 coef + 1 var param + 1 sigma2 = 4
        k = 4
        expected_bic = -2 * r.loglik + k * np.log(N)
        np.testing.assert_allclose(r.bic, expected_bic, atol=1e-10)


# ===========================================================================
# (d) CONFIDENCE INTERVAL COVERAGE
# ===========================================================================

class TestCICoverage:
    """Monte Carlo: 95% CIs should cover the true parameter ~95% of the time."""

    @pytest.mark.slow
    def test_coverage_ar1(self):
        """Simulate 200 datasets with AR(1) errors, fit GLS, check CI coverage.

        Allow 88-99% range for sampling variability with 200 simulations.
        """
        true_beta = np.array([3.0, 1.5])
        n_sim = 200
        covers = np.zeros((n_sim, len(true_beta)), dtype=bool)

        for i in range(n_sim):
            df = make_ar1_panel(
                n_subjects=30, n_times=5, phi=0.5,
                beta=true_beta, sigma=1.0, seed=i
            )
            r = GLS.from_formula(
                "y ~ x1", data=df, correlation=CorAR1(), groups="subject"
            ).fit()
            ci = r.conf_int(alpha=0.05)
            for j, name in enumerate(r.feature_names):
                covers[i, j] = ci.loc[name, "lower"] <= true_beta[j] <= ci.loc[name, "upper"]

        coverage = covers.mean(axis=0)
        for j, name in enumerate(["Intercept", "x1"]):
            assert 0.88 <= coverage[j] <= 0.99, (
                f"Coverage for {name}: {coverage[j]:.2%} outside [88%, 99%]"
            )


# ===========================================================================
# (e) KNOWN AR(1) RECOVERY
# ===========================================================================

class TestAR1Recovery:
    """Estimated phi should be close to the true generating phi."""

    def test_phi_recovery_moderate(self):
        """True phi=0.6, estimate should be within 0.15."""
        df = make_ar1_panel(
            n_subjects=80, n_times=8, phi=0.6,
            beta=np.array([1.0, 2.0]), sigma=1.0, seed=42
        )
        r = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        phi_est = r.correlation_params[0]
        assert abs(phi_est - 0.6) < 0.15, (
            f"Estimated phi={phi_est:.3f}, expected ~0.6"
        )

    def test_phi_recovery_weak(self):
        """True phi=0.2, should recover a small positive value."""
        df = make_ar1_panel(
            n_subjects=80, n_times=8, phi=0.2,
            beta=np.array([1.0, 2.0]), sigma=1.0, seed=99
        )
        r = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        phi_est = r.correlation_params[0]
        assert abs(phi_est - 0.2) < 0.15, (
            f"Estimated phi={phi_est:.3f}, expected ~0.2"
        )

    def test_phi_recovery_strong(self):
        """True phi=0.8, should recover a value close to 0.8."""
        df = make_ar1_panel(
            n_subjects=100, n_times=10, phi=0.8,
            beta=np.array([2.0, -1.0]), sigma=0.5, seed=77
        )
        r = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        phi_est = r.correlation_params[0]
        assert abs(phi_est - 0.8) < 0.15, (
            f"Estimated phi={phi_est:.3f}, expected ~0.8"
        )


# ===========================================================================
# (f) KNOWN VARIANCE RECOVERY
# ===========================================================================

class TestVarianceRecovery:
    """VarIdent should recover approximately correct variance ratios."""

    def test_variance_ratio_recovery(self):
        """True sigma ratio B/A = 3, VarIdent should recover log(3) ~ 1.1."""
        df = make_hetero_panel(
            n_subjects_per_group=40, n_times=6,
            beta=np.array([1.0, 2.0]),
            sigma_a=1.0, sigma_b=3.0, seed=55
        )
        r = GLS.from_formula(
            "y ~ x1", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()
        # VarIdent stores log(delta_B) where delta_B = sigma_B/sigma_A
        log_ratio_est = r.variance_params[0]
        # True log ratio = log(3) ~ 1.099
        assert abs(log_ratio_est - np.log(3.0)) < 0.3, (
            f"Estimated log(ratio)={log_ratio_est:.3f}, expected ~{np.log(3.0):.3f}"
        )

    def test_equal_variance_ratio_near_zero(self):
        """If sigma_A = sigma_B, the log ratio should be near 0."""
        df = make_hetero_panel(
            n_subjects_per_group=40, n_times=6,
            beta=np.array([1.0, 2.0]),
            sigma_a=1.0, sigma_b=1.0, seed=66
        )
        r = GLS.from_formula(
            "y ~ x1", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()
        log_ratio_est = r.variance_params[0]
        assert abs(log_ratio_est) < 0.3, (
            f"Estimated log(ratio)={log_ratio_est:.3f}, expected ~0.0"
        )


# ===========================================================================
# (g) LIKELIHOOD IMPROVEMENT
# ===========================================================================

class TestLikelihoodImprovement:
    """Correct correlation structure should improve log-likelihood over OLS."""

    def test_ar1_improves_over_ols(self):
        """AR(1) data fit with CorAR1 should have higher loglik than OLS."""
        df = make_ar1_panel(
            n_subjects=50, n_times=6, phi=0.7,
            beta=np.array([1.0, 1.5]), sigma=1.0, seed=12
        )
        r_ols = GLS.from_formula("y ~ x1", data=df).fit()
        r_ar1 = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert r_ar1.loglik > r_ols.loglik, (
            f"AR1 loglik ({r_ar1.loglik:.2f}) should exceed OLS ({r_ols.loglik:.2f})"
        )

    def test_comp_symm_improves_over_ols(self):
        """Data with exchangeable correlation should benefit from CorCompSymm.

        Generate data with compound symmetry (constant within-group correlation).
        """
        rng = np.random.RandomState(88)
        n_subjects = 50
        n_times = 5
        N = n_subjects * n_times
        rho = 0.5

        subjects = np.repeat(np.arange(n_subjects), n_times)
        x = rng.randn(N)

        # Compound symmetry errors: y_it = b0 + b1*x + u_i + e_it
        # where u_i ~ N(0, rho*sigma2) is shared within subject
        u = rng.randn(n_subjects) * np.sqrt(rho)
        e = rng.randn(N) * np.sqrt(1 - rho)
        errors = u[subjects] + e

        y = 2.0 + 1.0 * x + errors
        df = pd.DataFrame({"y": y, "x": x, "subject": subjects})

        r_ols = GLS.from_formula("y ~ x", data=df).fit()
        r_cs = GLS.from_formula(
            "y ~ x", data=df, correlation=CorCompSymm(), groups="subject"
        ).fit()
        assert r_cs.loglik > r_ols.loglik, (
            f"CompSymm loglik ({r_cs.loglik:.2f}) should exceed OLS ({r_ols.loglik:.2f})"
        )

    def test_varident_improves_over_ols(self):
        """Heteroscedastic data should benefit from VarIdent."""
        df = make_hetero_panel(
            n_subjects_per_group=30, n_times=5,
            beta=np.array([1.0, 2.0]),
            sigma_a=0.5, sigma_b=2.0, seed=33
        )
        r_ols = GLS.from_formula("y ~ x1", data=df).fit()
        r_vi = GLS.from_formula(
            "y ~ x1", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()
        assert r_vi.loglik > r_ols.loglik, (
            f"VarIdent loglik ({r_vi.loglik:.2f}) should exceed OLS ({r_ols.loglik:.2f})"
        )


# ===========================================================================
# (h) COEFFICIENT CONSISTENCY
# ===========================================================================

class TestCoefficientConsistency:
    """With large n, coefficients should be within 3 SE of true values."""

    def test_large_sample_ar1(self):
        """500 subjects, 6 time points: beta should be close to truth."""
        true_beta = np.array([3.0, -1.5])
        df = make_ar1_panel(
            n_subjects=500, n_times=6, phi=0.5,
            beta=true_beta, sigma=1.0, seed=0
        )
        r = GLS.from_formula(
            "y ~ x1", data=df, correlation=CorAR1(), groups="subject"
        ).fit()

        for j, name in enumerate(r.feature_names):
            se = r.bse[name]
            deviation = abs(r.params[name] - true_beta[j])
            assert deviation < 3 * se, (
                f"{name}: |est - true| = {deviation:.4f} > 3*SE = {3 * se:.4f}"
            )

    def test_large_sample_ols(self):
        """OLS on iid data: coefficients within 3 SE of truth."""
        rng = np.random.RandomState(1)
        N = 2000
        true_beta = np.array([10.0, -2.0, 0.5])
        x1 = rng.randn(N)
        x2 = rng.randn(N)
        y = true_beta[0] + true_beta[1] * x1 + true_beta[2] * x2 + rng.randn(N)
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        r = GLS.from_formula("y ~ x1 + x2", data=df).fit()

        for j, name in enumerate(r.feature_names):
            se = r.bse[name]
            deviation = abs(r.params[name] - true_beta[j])
            assert deviation < 3 * se, (
                f"{name}: |est - true| = {deviation:.4f} > 3*SE = {3 * se:.4f}"
            )

    def test_large_sample_hetero(self):
        """Large heteroscedastic sample: still consistent."""
        true_beta = np.array([0.0, 5.0])
        df = make_hetero_panel(
            n_subjects_per_group=100, n_times=6,
            beta=true_beta, sigma_a=0.5, sigma_b=3.0, seed=2
        )
        r = GLS.from_formula(
            "y ~ x1", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()

        for j, name in enumerate(r.feature_names):
            se = r.bse[name]
            deviation = abs(r.params[name] - true_beta[j])
            assert deviation < 3 * se, (
                f"{name}: |est - true| = {deviation:.4f} > 3*SE = {3 * se:.4f}"
            )
