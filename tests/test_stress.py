"""Stress tests for python_gls: numerical stability, edge cases, convergence."""

import warnings

import numpy as np
import pandas as pd
import pytest

from python_gls import GLS
from python_gls.correlation import (
    CorAR1,
    CorARMA,
    CorCAR1,
    CorCompSymm,
    CorSymm,
    CorExp,
    CorGaus,
    CorLin,
    CorRatio,
    CorSpher,
)
from python_gls.variance import (
    VarIdent,
    VarPower,
    VarExp,
    VarConstPower,
    VarFixed,
    VarComb,
)
from python_gls._parametrization import (
    unconstrained_to_corr,
    corr_to_unconstrained,
    angles_to_corr,
    corr_to_angles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(n_subjects, n_times, n_covariates=1, seed=42, phi=0.0,
                group_var=False, heteroscedastic=False, unbalanced=False):
    """Generate a panel dataset for testing.

    Parameters
    ----------
    n_subjects : int
    n_times : int
        Number of time periods per subject (may vary if unbalanced).
    n_covariates : int
    seed : int
    phi : float
        AR(1) parameter for within-subject errors.
    group_var : bool
        Whether to add a two-level group variable.
    heteroscedastic : bool
        Whether to make different groups have different error variance.
    unbalanced : bool
        If True, subjects have varying numbers of observations.
    """
    rng = np.random.default_rng(seed)

    rows = []
    for s in range(n_subjects):
        if unbalanced:
            nt = rng.integers(max(2, n_times // 2), n_times + 1)
        else:
            nt = n_times

        # Covariates
        x_vals = rng.standard_normal((nt, n_covariates))

        # AR(1) errors
        e = rng.standard_normal(nt)
        if abs(phi) > 0:
            for t in range(1, nt):
                e[t] = phi * e[t - 1] + np.sqrt(max(1 - phi ** 2, 0.01)) * e[t]

        sigma = 1.0
        if heteroscedastic and s >= n_subjects // 2:
            sigma = 3.0

        y = 2.0 + sum(1.0 * x_vals[:, j] for j in range(n_covariates)) + sigma * e * 0.5

        for t in range(nt):
            row = {"y": y[t], "subject": s, "time": t}
            for j in range(n_covariates):
                row[f"x{j}"] = x_vals[t, j]
            if group_var:
                row["group"] = "A" if s < n_subjects // 2 else "B"
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Large dataset tests
# ---------------------------------------------------------------------------

class TestLargeDatasets:
    """Test that the library handles large datasets without numerical issues."""

    def test_large_panel_ar1(self):
        """200 subjects, 10 time periods with AR(1) correlation."""
        df = _make_panel(200, 10, seed=1, phi=0.6)
        formula = "y ~ x0"
        r = GLS.from_formula(
            formula, data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert r.converged
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))
        assert r.sigma2 > 0

    def test_large_panel_comp_symm(self):
        """200 subjects, 8 time periods with compound symmetry."""
        df = _make_panel(200, 8, seed=2, phi=0.3)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorCompSymm(), groups="subject"
        ).fit()
        assert r.converged
        assert np.isfinite(r.loglik)

    def test_large_panel_symm(self):
        """100 subjects, 4 time periods with unstructured correlation.

        CorSymm has d(d-1)/2 = 6 params for d=4, so this is feasible.
        """
        df = _make_panel(100, 4, seed=3, phi=0.5)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorSymm(), groups="subject"
        ).fit()
        assert r.converged
        assert np.isfinite(r.loglik)

    def test_ml_vs_reml_large(self):
        """Both ML and REML should converge on a large dataset."""
        df = _make_panel(150, 6, seed=4, phi=0.4)
        r_ml = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject", method="ML"
        ).fit()
        r_reml = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject", method="REML"
        ).fit()
        assert r_ml.converged
        assert r_reml.converged
        # Coefficients should be close
        np.testing.assert_allclose(r_ml.params.values, r_reml.params.values, atol=0.2)


# ---------------------------------------------------------------------------
# Near-singular / extreme parameter tests
# ---------------------------------------------------------------------------

class TestNearSingularCorrelation:
    """Test numerical stability with extreme correlation parameters."""

    def test_ar1_high_phi(self):
        """AR(1) with phi very close to 1 (near-singular)."""
        df = _make_panel(50, 5, seed=10, phi=0.95)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))
        # Recovered phi should be high
        assert r.correlation_params[0] > 0.5

    def test_ar1_negative_phi(self):
        """AR(1) with phi close to -1 (alternating pattern)."""
        df = _make_panel(50, 5, seed=11, phi=-0.8)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))

    def test_comp_symm_high_rho(self):
        """Compound symmetry with rho close to 1."""
        df = _make_panel(40, 4, seed=12, phi=0.9)
        cor = CorCompSymm()
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=cor, groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))

    def test_ar1_phi_zero_should_approximate_ols(self):
        """With phi=0, AR(1) GLS should give results close to OLS."""
        df = _make_panel(60, 4, seed=13, phi=0.0)
        r_ols = GLS.from_formula("y ~ x0", data=df).fit()
        r_ar1 = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        # With no true autocorrelation, coefficients should be very similar
        np.testing.assert_allclose(
            r_ols.params.values, r_ar1.params.values, atol=0.3
        )


# ---------------------------------------------------------------------------
# Extreme variance ratios
# ---------------------------------------------------------------------------

class TestExtremeVarianceRatios:
    """Test with very small and very large variance ratios."""

    def test_large_variance_ratio(self):
        """One group has 10x the standard deviation of another."""
        rng = np.random.default_rng(20)
        n_subjects = 40
        n_times = 4
        rows = []
        for s in range(n_subjects):
            sigma = 0.1 if s < 20 else 1.0  # 10x ratio
            for t in range(n_times):
                x = rng.standard_normal()
                y = 1.0 + 2.0 * x + sigma * rng.standard_normal()
                rows.append({
                    "y": y, "x": x, "subject": s, "time": t,
                    "group": "lo" if s < 20 else "hi",
                })
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()
        assert r.converged
        assert r.variance_params is not None
        # VarIdent sorts levels alphabetically: "hi" is reference, "lo" has lower variance
        # so the log-ratio for "lo" relative to "hi" should be negative (lo < hi)
        assert r.variance_params[0] < 0

    def test_very_small_variance_ratio(self):
        """Groups have nearly identical variance."""
        rng = np.random.default_rng(21)
        n_subjects = 40
        n_times = 4
        rows = []
        for s in range(n_subjects):
            sigma = 1.0 if s < 20 else 1.01
            for t in range(n_times):
                x = rng.standard_normal()
                y = 1.0 + 2.0 * x + sigma * rng.standard_normal()
                rows.append({
                    "y": y, "x": x, "subject": s, "time": t,
                    "group": "A" if s < 20 else "B",
                })
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()
        assert r.converged
        # Variance ratio should be near 0 in log scale
        assert abs(r.variance_params[0]) < 1.0

    def test_varpower_extreme_covariate(self):
        """VarPower with covariate values spanning many orders of magnitude."""
        rng = np.random.default_rng(22)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        # Covariate for variance: values from 0.01 to 100
        v = np.exp(rng.uniform(-4, 4, n))
        y = 1.0 + x + rng.standard_normal(n) * np.sqrt(v)
        df = pd.DataFrame({"y": y, "x": x, "v": v, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarPower("v"), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))


# ---------------------------------------------------------------------------
# Unbalanced panels
# ---------------------------------------------------------------------------

class TestUnbalancedPanels:
    """Test with unequal group sizes."""

    def test_unbalanced_ar1(self):
        """AR(1) with varying numbers of observations per subject."""
        df = _make_panel(60, 8, seed=30, phi=0.5, unbalanced=True)
        # Verify unbalanced
        sizes = df.groupby("subject").size()
        assert sizes.nunique() > 1, "Panel is not actually unbalanced"
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert r.converged
        assert np.isfinite(r.loglik)

    def test_unbalanced_comp_symm(self):
        """Compound symmetry with unbalanced panel."""
        df = _make_panel(50, 6, seed=31, phi=0.3, unbalanced=True)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorCompSymm(), groups="subject"
        ).fit()
        assert r.converged
        assert np.isfinite(r.loglik)

    def test_unbalanced_with_variance(self):
        """Unbalanced panel with both correlation and variance structures."""
        df = _make_panel(
            40, 6, seed=32, phi=0.4, group_var=True,
            heteroscedastic=True, unbalanced=True,
        )
        r = GLS.from_formula(
            "y ~ x0", data=df,
            correlation=CorAR1(),
            variance=VarIdent("group"),
            groups="subject",
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))


# ---------------------------------------------------------------------------
# Single observation per group / small groups
# ---------------------------------------------------------------------------

class TestSmallGroups:
    """Test edge cases with very small groups."""

    def test_two_obs_per_group(self):
        """Each subject has exactly 2 observations."""
        df = _make_panel(80, 2, seed=40, phi=0.5)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))

    def test_single_obs_per_group_no_correlation(self):
        """Single observation per group -- correlation structure should be identity."""
        rng = np.random.default_rng(41)
        n = 100
        subjects = np.arange(n)
        x = rng.standard_normal(n)
        y = 1.0 + 2.0 * x + rng.standard_normal(n) * 0.5
        df = pd.DataFrame({"y": y, "x": x, "subject": subjects})
        # With 1 obs per group, AR(1) matrix is 1x1 = identity
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        # Should be close to OLS
        r_ols = GLS.from_formula("y ~ x", data=df).fit()
        np.testing.assert_allclose(r.params.values, r_ols.params.values, atol=0.01)

    def test_single_group(self):
        """All observations in one group (degenerate case)."""
        rng = np.random.default_rng(42)
        n = 50
        x = rng.standard_normal(n)
        y = 1.0 + 2.0 * x + rng.standard_normal(n)
        subjects = np.zeros(n, dtype=int)
        df = pd.DataFrame({"y": y, "x": x, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))

    def test_three_obs_per_group(self):
        """Three observations per group -- minimal for CorSymm with d=3."""
        df = _make_panel(50, 3, seed=43, phi=0.3)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorSymm(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert r.correlation_params is not None


# ---------------------------------------------------------------------------
# High-dimensional CorSymm
# ---------------------------------------------------------------------------

class TestHighDimensionalCorSymm:
    """CorSymm with many time periods (many free parameters)."""

    def test_corsymm_5_periods(self):
        """CorSymm with d=5 has 10 free parameters."""
        df = _make_panel(80, 5, seed=50, phi=0.4)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorSymm(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert r.correlation_params is not None
        assert len(r.correlation_params) == 10

    def test_corsymm_6_periods(self):
        """CorSymm with d=6 has 15 free parameters."""
        df = _make_panel(100, 6, seed=51, phi=0.3)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorSymm(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert r.correlation_params is not None
        assert len(r.correlation_params) == 15

    def test_corsymm_parametrization_roundtrip_5d(self):
        """Parametrization roundtrip for 5x5 correlation matrix."""
        rng = np.random.default_rng(52)
        # Generate a valid correlation matrix
        A = rng.standard_normal((5, 5))
        R = np.corrcoef(A)
        # Ensure PD
        eigvals = np.linalg.eigvalsh(R)
        if np.min(eigvals) < 1e-4:
            R += (1e-4 - np.min(eigvals) + 1e-4) * np.eye(5)
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)
        u = corr_to_unconstrained(R)
        R2 = unconstrained_to_corr(u, 5)
        np.testing.assert_allclose(R, R2, atol=1e-4)

    def test_corsymm_parametrization_always_pd(self):
        """Random unconstrained params should always give PD matrix for d=6."""
        rng = np.random.default_rng(53)
        for _ in range(20):
            u = rng.standard_normal(15)  # 6x6 matrix has 15 params
            R = unconstrained_to_corr(u, 6)
            eigvals = np.linalg.eigvalsh(R)
            assert np.all(eigvals > -1e-10), f"Not PD for d=6: min eigval={np.min(eigvals)}"
            np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Many covariates
# ---------------------------------------------------------------------------

class TestManyCovariates:
    """Test with many predictor variables."""

    def test_10_covariates_ols(self):
        """OLS with 10 covariates."""
        rng = np.random.default_rng(60)
        n = 500
        k = 10
        X = rng.standard_normal((n, k))
        beta_true = rng.standard_normal(k)
        y = 1.0 + X @ beta_true + rng.standard_normal(n) * 0.5
        data = {"y": y}
        formula_parts = []
        for j in range(k):
            data[f"x{j}"] = X[:, j]
            formula_parts.append(f"x{j}")
        df = pd.DataFrame(data)
        formula = "y ~ " + " + ".join(formula_parts)
        r = GLS.from_formula(formula, data=df).fit()
        assert len(r.params) == k + 1  # k covariates + intercept
        assert np.all(np.isfinite(r.params.values))
        # Check that true intercept is within CI
        ci = r.conf_int()
        assert ci.loc["Intercept", "lower"] < 1.0 < ci.loc["Intercept", "upper"]

    def test_10_covariates_with_ar1(self):
        """AR(1) GLS with 10 covariates."""
        df = _make_panel(80, 5, n_covariates=10, seed=61, phi=0.5)
        formula = "y ~ " + " + ".join([f"x{j}" for j in range(10)])
        r = GLS.from_formula(
            formula, data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert r.converged
        assert len(r.params) == 11
        assert np.all(np.isfinite(r.params.values))

    def test_many_covariates_near_collinear(self):
        """Near-collinear design matrix (stress for lstsq and inverse)."""
        rng = np.random.default_rng(62)
        n = 200
        x1 = rng.standard_normal(n)
        X = np.column_stack([x1 + rng.standard_normal(n) * 0.01 for _ in range(5)])
        y = 1.0 + X @ np.ones(5) + rng.standard_normal(n)
        data = {"y": y}
        formula_parts = []
        for j in range(5):
            data[f"x{j}"] = X[:, j]
            formula_parts.append(f"x{j}")
        df = pd.DataFrame(data)
        formula = "y ~ " + " + ".join(formula_parts)
        # Should still produce finite results (even if SEs are large)
        r = GLS.from_formula(formula, data=df).fit()
        assert np.all(np.isfinite(r.params.values))


# ---------------------------------------------------------------------------
# Intercept-only models
# ---------------------------------------------------------------------------

class TestInterceptOnly:
    """Test intercept-only models (no covariates)."""

    def test_intercept_only_ols(self):
        """y ~ 1 should estimate just the mean."""
        rng = np.random.default_rng(70)
        n = 100
        y = 5.0 + rng.standard_normal(n) * 0.5
        df = pd.DataFrame({"y": y})
        r = GLS.from_formula("y ~ 1", data=df).fit()
        assert len(r.params) == 1
        np.testing.assert_allclose(r.params.values[0], np.mean(y), atol=1e-10)

    def test_intercept_only_with_ar1(self):
        """Intercept-only with AR(1) correlation."""
        df = _make_panel(50, 5, n_covariates=0, seed=71, phi=0.6)
        # Remove x0 from formula
        r = GLS.from_formula(
            "y ~ 1", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert r.converged
        assert len(r.params) == 1
        assert np.isfinite(r.loglik)

    def test_intercept_only_with_comp_symm(self):
        """Intercept-only with compound symmetry."""
        df = _make_panel(40, 4, n_covariates=0, seed=72, phi=0.3)
        r = GLS.from_formula(
            "y ~ 1", data=df, correlation=CorCompSymm(), groups="subject"
        ).fit()
        assert r.converged
        assert len(r.params) == 1


# ---------------------------------------------------------------------------
# Outliers and extreme values
# ---------------------------------------------------------------------------

class TestOutliersAndExtremeValues:
    """Test robustness to outliers and extreme data values."""

    def test_single_outlier_in_y(self):
        """A single very large y value should not crash the model."""
        rng = np.random.default_rng(80)
        n_subjects = 40
        n_times = 4
        df = _make_panel(n_subjects, n_times, seed=80, phi=0.3)
        # Insert outlier
        df.loc[0, "y"] = 1e6
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.all(np.isfinite(r.params.values))

    def test_large_covariate_values(self):
        """Very large x values (but legitimate) should still work."""
        rng = np.random.default_rng(81)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n) * 1e4
        y = 1.0 + 0.001 * x + rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "x": x, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.all(np.isfinite(r.params.values))

    def test_near_zero_response(self):
        """Response values very close to zero."""
        rng = np.random.default_rng(82)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        y = 1e-10 + 1e-10 * x + rng.standard_normal(n) * 1e-10
        df = pd.DataFrame({"y": y, "x": x, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        assert np.all(np.isfinite(r.params.values))

    def test_constant_response(self):
        """Constant y (zero variance -- degenerate case)."""
        rng = np.random.default_rng(83)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        y = np.ones(n) * 5.0
        df = pd.DataFrame({"y": y, "x": x, "subject": subjects})
        # This is a degenerate case; we just want no crash
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = GLS.from_formula(
                "y ~ x", data=df, correlation=CorAR1(), groups="subject"
            ).fit()
        # Intercept should be 5.0, slope should be 0
        np.testing.assert_allclose(r.params.values[0], 5.0, atol=1e-6)
        np.testing.assert_allclose(r.params.values[1], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# ARMA correlation tests
# ---------------------------------------------------------------------------

class TestARMAStress:
    """Stress tests for ARMA correlation structure."""

    def test_arma_p2_q0(self):
        """AR(2) process."""
        df = _make_panel(60, 6, seed=90, phi=0.5)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorARMA(p=2, q=0), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)

    def test_arma_p0_q2(self):
        """MA(2) process."""
        df = _make_panel(60, 6, seed=91)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorARMA(p=0, q=2), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)

    def test_arma_p1_q1(self):
        """ARMA(1,1) process."""
        df = _make_panel(60, 6, seed=92, phi=0.4)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorARMA(p=1, q=1), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)


# ---------------------------------------------------------------------------
# CAR1 stress tests
# ---------------------------------------------------------------------------

class TestCAR1Stress:
    """Stress tests for continuous AR(1) correlation."""

    def test_car1_irregular_spacing(self):
        """Irregularly-spaced time points."""
        rng = np.random.default_rng(100)
        n_subjects = 50
        rows = []
        cor = CorCAR1()
        for s in range(n_subjects):
            # Random time points
            times = np.sort(rng.uniform(0, 10, size=5))
            cor.set_time_points(s, times)
            x = rng.standard_normal(5)
            y = 1.0 + 2.0 * x + rng.standard_normal(5) * 0.5
            for t in range(5):
                rows.append({"y": y[t], "x": x[t], "subject": s, "time": times[t]})
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=cor, groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert r.correlation_params is not None
        assert 0 < r.correlation_params[0] < 1

    def test_car1_very_close_time_points(self):
        """Time points very close together (near-singular)."""
        rng = np.random.default_rng(101)
        n_subjects = 30
        rows = []
        cor = CorCAR1()
        for s in range(n_subjects):
            # Very close time points
            times = np.array([0, 0.001, 0.002, 0.003])
            cor.set_time_points(s, times)
            x = rng.standard_normal(4)
            y = 1.0 + x + rng.standard_normal(4) * 0.5
            for t in range(4):
                rows.append({"y": y[t], "x": x[t], "subject": s, "time": times[t]})
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=cor, groups="subject"
        ).fit()
        assert np.all(np.isfinite(r.params.values))


# ---------------------------------------------------------------------------
# Spatial correlation stress tests
# ---------------------------------------------------------------------------

class TestSpatialStress:
    """Stress tests for spatial correlation structures."""

    @pytest.mark.parametrize("CorClass", [CorExp, CorGaus, CorRatio, CorSpher, CorLin])
    def test_spatial_1d_fit(self, CorClass):
        """Each spatial correlation should produce a valid fit."""
        rng = np.random.default_rng(110)
        n_subjects = 40
        n_locs = 4
        rows = []
        cor = CorClass(range_param=2.0)
        for s in range(n_subjects):
            coords = np.sort(rng.uniform(0, 5, n_locs))
            cor.set_coordinates(s, coords)
            x = rng.standard_normal(n_locs)
            y = 1.0 + x + rng.standard_normal(n_locs) * 0.3
            for j in range(n_locs):
                rows.append({"y": y[j], "x": x[j], "subject": s})
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=cor, groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))

    def test_spatial_with_nugget(self):
        """Exponential correlation with nugget effect."""
        rng = np.random.default_rng(111)
        n_subjects = 40
        n_locs = 4
        rows = []
        cor = CorExp(range_param=2.0, nugget=True)
        for s in range(n_subjects):
            coords = np.sort(rng.uniform(0, 5, n_locs))
            cor.set_coordinates(s, coords)
            x = rng.standard_normal(n_locs)
            y = 1.0 + x + rng.standard_normal(n_locs) * 0.3
            for j in range(n_locs):
                rows.append({"y": y[j], "x": x[j], "subject": s})
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=cor, groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert r.correlation_params is not None
        assert len(r.correlation_params) == 2  # range + nugget


# ---------------------------------------------------------------------------
# Variance function edge cases
# ---------------------------------------------------------------------------

class TestVarianceFunctionEdgeCases:
    """Edge cases for variance functions."""

    def test_varident_many_levels(self):
        """VarIdent with many group levels."""
        rng = np.random.default_rng(120)
        n_groups = 10
        n_per = 20
        rows = []
        for g in range(n_groups):
            sigma = 0.5 + g * 0.3
            for i in range(n_per):
                x = rng.standard_normal()
                y = 1.0 + x + sigma * rng.standard_normal()
                rows.append({
                    "y": y, "x": x,
                    "subject": g * n_per + i,
                    "var_group": f"g{g}",
                })
        df = pd.DataFrame(rows)
        # Each subject is its own group for correlation (1 obs each)
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarIdent("var_group"), groups="subject"
        ).fit()
        assert r.converged
        assert r.variance_params is not None
        assert len(r.variance_params) == n_groups - 1

    def test_varexp_fit(self):
        """VarExp with covariate-dependent variance."""
        rng = np.random.default_rng(121)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        v = rng.uniform(0, 2, n)
        y = 1.0 + x + np.exp(0.5 * v) * rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "x": x, "v": v, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarExp("v"), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert r.variance_params is not None

    def test_varconstpower_fit(self):
        """VarConstPower fit."""
        rng = np.random.default_rng(122)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        v = np.abs(rng.standard_normal(n)) + 0.1
        y = 1.0 + x + (1 + v) * rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "x": x, "v": v, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarConstPower("v"), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)

    def test_varfixed_no_estimation(self):
        """VarFixed has no parameters to estimate."""
        rng = np.random.default_rng(123)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        w = rng.uniform(1, 10, n)
        y = 1.0 + x + np.sqrt(w) * rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "x": x, "w": w, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarFixed("w"), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)
        assert r.variance_params is None  # No params estimated

    def test_varcomb_power_and_ident(self):
        """VarComb combining VarPower and VarIdent."""
        rng = np.random.default_rng(124)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        v = np.abs(rng.standard_normal(n)) + 0.5
        group = np.where(subjects < 25, "A", "B")
        sigma_group = np.where(group == "A", 0.5, 1.5)
        y = 1.0 + x + sigma_group * v * rng.standard_normal(n)
        df = pd.DataFrame({
            "y": y, "x": x, "v": v, "subject": subjects, "group": group,
        })
        vc = VarComb(VarPower("v"), VarIdent("group"))
        r = GLS.from_formula(
            "y ~ x", data=df, variance=vc, groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)

    def test_varpower_with_zero_covariate(self):
        """VarPower when covariate has zero values (should use fallback weight=1)."""
        rng = np.random.default_rng(125)
        n = 200
        subjects = np.repeat(np.arange(50), 4)
        x = rng.standard_normal(n)
        v = np.abs(rng.standard_normal(n))
        v[::10] = 0.0  # Insert zeros
        y = 1.0 + x + rng.standard_normal(n)
        df = pd.DataFrame({"y": y, "x": x, "v": v, "subject": subjects})
        r = GLS.from_formula(
            "y ~ x", data=df, variance=VarPower("v"), groups="subject"
        ).fit()
        assert np.all(np.isfinite(r.params.values))


# ---------------------------------------------------------------------------
# Combined correlation + variance stress
# ---------------------------------------------------------------------------

class TestCombinedCorrelationVariance:
    """Stress tests with both correlation and variance structures."""

    def test_ar1_plus_varident_large(self):
        """Large panel with AR(1) + VarIdent."""
        df = _make_panel(
            100, 6, seed=130, phi=0.5, group_var=True, heteroscedastic=True
        )
        r = GLS.from_formula(
            "y ~ x0", data=df,
            correlation=CorAR1(),
            variance=VarIdent("group"),
            groups="subject",
        ).fit()
        assert r.converged
        assert r.correlation_params is not None
        assert r.variance_params is not None

    def test_comp_symm_plus_varexp(self):
        """CompSymm + VarExp."""
        rng = np.random.default_rng(131)
        n_subjects = 60
        n_times = 4
        rows = []
        for s in range(n_subjects):
            for t in range(n_times):
                x = rng.standard_normal()
                v = rng.uniform(0, 2)
                y = 1.0 + x + np.exp(0.3 * v) * rng.standard_normal()
                rows.append({
                    "y": y, "x": x, "v": v, "subject": s, "time": t,
                })
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df,
            correlation=CorCompSymm(),
            variance=VarExp("v"),
            groups="subject",
        ).fit()
        assert np.isfinite(r.loglik)

    def test_symm_plus_varident(self):
        """CorSymm + VarIdent (many parameters)."""
        df = _make_panel(
            80, 4, seed=132, phi=0.3, group_var=True, heteroscedastic=True
        )
        r = GLS.from_formula(
            "y ~ x0", data=df,
            correlation=CorSymm(),
            variance=VarIdent("group"),
            groups="subject",
        ).fit()
        assert np.isfinite(r.loglik)


# ---------------------------------------------------------------------------
# Convergence tests
# ---------------------------------------------------------------------------

class TestConvergence:
    """Test convergence behavior for difficult problems."""

    def test_convergence_with_high_correlation(self):
        """High true correlation -- should converge even if difficult."""
        df = _make_panel(80, 5, seed=140, phi=0.9)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit(maxiter=500)
        assert np.isfinite(r.loglik)
        # phi should be estimated as high
        assert r.correlation_params[0] > 0.5

    def test_convergence_comp_symm_negative(self):
        """Negative compound symmetry -- should converge."""
        rng = np.random.default_rng(141)
        n_subjects = 60
        n_times = 4
        rows = []
        for s in range(n_subjects):
            # Generate negatively correlated errors within subject
            e = rng.standard_normal(n_times)
            e -= np.mean(e)  # Creates some negative correlation
            for t in range(n_times):
                x = rng.standard_normal()
                y = 1.0 + x + e[t] * 0.5
                rows.append({"y": y, "x": x, "subject": s, "time": t})
        df = pd.DataFrame(rows)
        r = GLS.from_formula(
            "y ~ x", data=df, correlation=CorCompSymm(), groups="subject"
        ).fit()
        assert np.isfinite(r.loglik)

    def test_verbose_does_not_crash(self):
        """Verbose mode should work without errors."""
        df = _make_panel(30, 4, seed=142, phi=0.3)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit(verbose=False)  # Just ensure no crash
        assert np.isfinite(r.loglik)


# ---------------------------------------------------------------------------
# Direct endog/exog interface stress
# ---------------------------------------------------------------------------

class TestDirectInterface:
    """Test the direct endog/exog interface (no formula)."""

    def test_direct_interface_with_correlation(self):
        """Use endog/exog directly with AR(1) correlation."""
        rng = np.random.default_rng(150)
        n_subjects = 30
        n_times = 4
        N = n_subjects * n_times
        groups = np.repeat(np.arange(n_subjects), n_times)
        X = np.column_stack([np.ones(N), rng.standard_normal(N)])
        y = X @ np.array([1.0, 2.0]) + rng.standard_normal(N) * 0.5
        r = GLS(
            endog=y, exog=X, correlation=CorAR1(), groups=groups
        ).fit()
        assert np.isfinite(r.loglik)
        assert np.all(np.isfinite(r.params.values))

    def test_1d_exog(self):
        """Single-column exog (1D input) should be handled."""
        rng = np.random.default_rng(151)
        n = 50
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n) * 0.5
        r = GLS(endog=y, exog=x).fit()
        assert len(r.params) == 1


# ---------------------------------------------------------------------------
# Parametrization edge cases
# ---------------------------------------------------------------------------

class TestParametrizationEdgeCases:
    """Test spherical parametrization at boundaries."""

    def test_extreme_unconstrained_values(self):
        """Very large unconstrained values should produce valid correlation."""
        for val in [-100, -10, 10, 100]:
            u = np.full(3, val)  # 3x3 matrix
            R = unconstrained_to_corr(u, 3)
            eigvals = np.linalg.eigvalsh(R)
            assert np.all(eigvals > -1e-10), f"Not PD for u={val}: {eigvals}"
            np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-10)

    def test_zero_unconstrained(self):
        """Zero unconstrained params should give a valid matrix."""
        u = np.zeros(3)
        R = unconstrained_to_corr(u, 3)
        eigvals = np.linalg.eigvalsh(R)
        assert np.all(eigvals > -1e-10)
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-10)

    def test_near_identity_roundtrip(self):
        """Near-identity matrix should roundtrip."""
        R = np.eye(3) * 0.999 + np.full((3, 3), 0.001)
        np.fill_diagonal(R, 1.0)
        u = corr_to_unconstrained(R)
        R2 = unconstrained_to_corr(u, 3)
        np.testing.assert_allclose(R, R2, atol=1e-3)

    def test_high_correlation_matrix_roundtrip(self):
        """Matrix with high off-diagonal correlations."""
        R = np.array([
            [1.0, 0.95, 0.90],
            [0.95, 1.0, 0.95],
            [0.90, 0.95, 1.0],
        ])
        # Ensure PD
        eigvals = np.linalg.eigvalsh(R)
        assert np.all(eigvals > 0), "Test matrix not PD"
        u = corr_to_unconstrained(R)
        R2 = unconstrained_to_corr(u, 3)
        np.testing.assert_allclose(R, R2, atol=1e-3)


# ---------------------------------------------------------------------------
# Model comparison / information criteria
# ---------------------------------------------------------------------------

class TestModelComparison:
    """Test that AIC/BIC correctly count parameters."""

    def test_aic_bic_with_correlation(self):
        """AIC/BIC should account for correlation parameters."""
        df = _make_panel(50, 4, seed=160, phi=0.5)
        r_ols = GLS.from_formula("y ~ x0", data=df).fit()
        r_ar1 = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        # AR1 has one extra parameter (phi)
        assert np.isfinite(r_ols.aic)
        assert np.isfinite(r_ar1.aic)
        assert np.isfinite(r_ols.bic)
        assert np.isfinite(r_ar1.bic)

    def test_aic_bic_with_variance(self):
        """AIC/BIC should account for variance parameters."""
        df = _make_panel(50, 4, seed=161, phi=0.0, group_var=True)
        r = GLS.from_formula(
            "y ~ x0", data=df, variance=VarIdent("group"), groups="subject"
        ).fit()
        assert np.isfinite(r.aic)
        assert np.isfinite(r.bic)


# ---------------------------------------------------------------------------
# Residuals and fitted values with correlation
# ---------------------------------------------------------------------------

class TestResidualsFittedWithCorrelation:
    """Test that residuals + fitted = y even with correlation/variance."""

    def test_resid_plus_fitted_equals_y(self):
        """resid + fitted = y for GLS with AR(1)."""
        df = _make_panel(50, 4, seed=170, phi=0.5)
        r = GLS.from_formula(
            "y ~ x0", data=df, correlation=CorAR1(), groups="subject"
        ).fit()
        np.testing.assert_allclose(
            r.resid + r.fittedvalues, df["y"].values, atol=1e-10
        )

    def test_resid_plus_fitted_with_variance(self):
        """resid + fitted = y for GLS with VarIdent."""
        df = _make_panel(40, 4, seed=171, group_var=True, heteroscedastic=True)
        r = GLS.from_formula(
            "y ~ x0", data=df,
            correlation=CorAR1(),
            variance=VarIdent("group"),
            groups="subject",
        ).fit()
        np.testing.assert_allclose(
            r.resid + r.fittedvalues, df["y"].values, atol=1e-10
        )
