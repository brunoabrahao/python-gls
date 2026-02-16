"""End-to-end model tests."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from python_gls import GLS
from python_gls.correlation import CorAR1, CorCompSymm, CorSymm
from python_gls.variance import VarIdent


class TestOLSBaseline:
    """Verify OLS mode matches statsmodels exactly."""

    def test_ols_coefficients(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data, method="REML").fit()
        X = sm.add_constant(simple_data[["x1", "x2"]])
        ols = sm.OLS(simple_data["y"], X).fit()
        np.testing.assert_allclose(r.params.values, ols.params.values, atol=1e-10)

    def test_ols_standard_errors(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data, method="REML").fit()
        X = sm.add_constant(simple_data[["x1", "x2"]])
        ols = sm.OLS(simple_data["y"], X).fit()
        np.testing.assert_allclose(r.bse.values, ols.bse.values, atol=1e-10)

    def test_ols_tvalues(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data, method="REML").fit()
        X = sm.add_constant(simple_data[["x1", "x2"]])
        ols = sm.OLS(simple_data["y"], X).fit()
        np.testing.assert_allclose(r.tvalues.values, ols.tvalues.values, atol=1e-8)

    def test_conf_int(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data, method="REML").fit()
        ci = r.conf_int()
        # Check that true values fall within 95% CI
        assert ci.loc["Intercept", "lower"] < 3.0 < ci.loc["Intercept", "upper"]
        assert ci.loc["x1", "lower"] < 1.5 < ci.loc["x1", "upper"]
        assert ci.loc["x2", "lower"] < -2.0 < ci.loc["x2", "upper"]


class TestGLSWithCorrelation:
    """Test GLS with various correlation structures."""

    def test_ar1_fit(self, panel_data):
        r = GLS.from_formula(
            "y ~ x", data=panel_data, correlation=CorAR1(), groups="subject"
        ).fit()
        assert r.converged
        # AR1 parameter should be positive (true phi=0.6)
        assert r.correlation_params[0] > 0
        assert r.nobs == len(panel_data)

    def test_ar1_improves_loglik(self, panel_data):
        r_ols = GLS.from_formula("y ~ x", data=panel_data).fit()
        r_ar1 = GLS.from_formula(
            "y ~ x", data=panel_data, correlation=CorAR1(), groups="subject"
        ).fit()
        assert r_ar1.loglik > r_ols.loglik

    def test_comp_symm_fit(self, panel_data):
        r = GLS.from_formula(
            "y ~ x", data=panel_data, correlation=CorCompSymm(), groups="subject"
        ).fit()
        assert r.converged
        assert abs(r.correlation_params[0]) < 1  # rho in (-1, 1)

    def test_symm_fit(self, panel_data):
        r = GLS.from_formula(
            "y ~ x", data=panel_data, correlation=CorSymm(), groups="subject"
        ).fit()
        assert r.converged
        assert r.correlation_params is not None


class TestGLSWithVariance:
    """Test GLS with variance functions."""

    def test_varident_fit(self, hetero_data):
        r = GLS.from_formula(
            "y ~ x",
            data=hetero_data,
            variance=VarIdent("group"),
            groups="subject",
        ).fit()
        assert r.converged
        assert r.variance_params is not None
        # Group B has 3x the variance, so log ratio should be positive
        assert r.variance_params[0] > 0


class TestGLSCombined:
    """Test GLS with both correlation and variance structures."""

    def test_ar1_plus_varident(self, panel_data):
        panel_data = panel_data.copy()
        panel_data["group"] = np.where(panel_data["subject"] < 20, "A", "B")
        r = GLS.from_formula(
            "y ~ x",
            data=panel_data,
            correlation=CorAR1(),
            variance=VarIdent("group"),
            groups="subject",
        ).fit()
        assert r.converged
        assert r.correlation_params is not None
        assert r.variance_params is not None


class TestResultsInterface:
    """Test results object properties and methods."""

    def test_summary_string(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data).fit()
        s = r.summary()
        assert "Generalized Least Squares" in s
        assert "Intercept" in s
        assert "x1" in s
        assert "x2" in s

    def test_aic_bic(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data).fit()
        assert np.isfinite(r.aic)
        assert np.isfinite(r.bic)
        assert r.bic > r.aic  # For n > e^2 ≈ 7.4

    def test_residuals(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data).fit()
        resid = r.resid
        assert len(resid) == len(simple_data)
        # Residuals should have mean close to zero
        assert abs(np.mean(resid)) < 0.1

    def test_fittedvalues(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data).fit()
        fitted = r.fittedvalues
        assert len(fitted) == len(simple_data)
        np.testing.assert_allclose(
            r.resid + r.fittedvalues,
            simple_data["y"].values,
            atol=1e-10,
        )

    def test_cov_params_dataframe(self, simple_data):
        r = GLS.from_formula("y ~ x1 + x2", data=simple_data).fit()
        cov = r.cov_params_func()
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (3, 3)
        # Covariance matrix should be positive definite
        eigvals = np.linalg.eigvalsh(cov.values)
        assert np.all(eigvals > 0)

    def test_method_ml_vs_reml(self, panel_data):
        r_ml = GLS.from_formula(
            "y ~ x", data=panel_data, correlation=CorAR1(), groups="subject", method="ML"
        ).fit()
        r_reml = GLS.from_formula(
            "y ~ x", data=panel_data, correlation=CorAR1(), groups="subject", method="REML"
        ).fit()
        # Coefficients should be similar but not identical
        assert np.allclose(r_ml.params.values, r_reml.params.values, atol=0.1)
        # REML sigma^2 should be slightly larger (unbiased)
        assert r_reml.sigma2 >= r_ml.sigma2 - 0.01


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_method(self, simple_data):
        with pytest.raises(ValueError, match="method must be"):
            GLS.from_formula("y ~ x1", data=simple_data, method="INVALID")

    def test_single_predictor(self, simple_data):
        r = GLS.from_formula("y ~ x1", data=simple_data).fit()
        assert len(r.params) == 2  # Intercept + x1

    def test_no_data_raises(self):
        with pytest.raises(ValueError):
            GLS().fit()

    def test_small_sample(self):
        df = pd.DataFrame({"y": [1, 2, 3], "x": [1, 2, 3]})
        r = GLS.from_formula("y ~ x", data=df).fit()
        assert r.converged

    def test_endog_exog_interface(self, simple_data):
        """Test direct endog/exog interface (no formula)."""
        y = simple_data["y"].values
        X = np.column_stack([np.ones(len(y)), simple_data[["x1", "x2"]].values])
        r = GLS(endog=y, exog=X).fit()
        assert len(r.params) == 3
