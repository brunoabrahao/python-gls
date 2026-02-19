"""Comparison tests: python_gls vs R nlme::gls() reference values.

Each scenario uses data generated in R (deterministic via set.seed), fitted
with nlme::gls(), and stored in tests/fixtures/r_reference.json.  The same
data is loaded here, fitted with python_gls, and the results are compared.

Regenerate the fixture with:
    Rscript tests/generate_r_reference.R
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from python_gls import GLS
from python_gls.correlation import CorAR1, CorCompSymm
from python_gls.variance import VarIdent

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "r_reference.json"

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
COEF_ATOL = 0.05
SE_RTOL = 0.10
SIGMA_RTOL = 0.05
LOGLIK_ATOL = 1.0
AIC_ATOL = 2.0
BIC_ATOL = 2.0
COR_PARAM_ATOL = 0.05
VAR_PARAM_RTOL = 0.15


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def r_reference():
    """Load R reference fixture; skip if not generated."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            "R reference fixture not found. Run: Rscript tests/generate_r_reference.R"
        )
    return json.loads(FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_dataframe(data_dict: dict) -> pd.DataFrame:
    """Reconstruct a DataFrame from the JSON data dict."""
    # Convert lists to arrays; handle mixed types (subject may be int/str)
    return pd.DataFrame(data_dict)


def _fit_scenario(scenario: dict) -> "GLSResults":
    """Fit python_gls on a single scenario from the fixture."""
    df = _build_dataframe(scenario["data"])
    cfg = scenario["config"]

    formula = cfg["formula"]
    method = cfg["method"]

    # Correlation
    cor_type = cfg.get("correlation", "none")
    correlation = None
    if cor_type == "AR1":
        correlation = CorAR1()
    elif cor_type == "CompSymm":
        correlation = CorCompSymm()

    # Variance
    var_type = cfg.get("variance", "none")
    variance = None
    var_group = cfg.get("variance_group")
    if var_type == "VarIdent" and var_group:
        variance = VarIdent(var_group)

    # Groups: needed whenever correlation or variance is present
    groups = "subject" if (correlation is not None or variance is not None) else None

    model = GLS.from_formula(
        formula, data=df, correlation=correlation, variance=variance,
        groups=groups, method=method,
    )
    return model.fit(maxiter=200)


def _compare_coefficients(result, r_results):
    """Compare coefficient estimates."""
    r_coefs = r_results["coefficients"]
    for name, r_val in r_coefs.items():
        # R uses "(Intercept)", python_gls uses "Intercept"
        py_name = name.replace("(Intercept)", "Intercept")
        assert py_name in result.params.index, (
            f"Coefficient '{py_name}' not found in python result; "
            f"available: {list(result.params.index)}"
        )
        np.testing.assert_allclose(
            result.params[py_name], r_val, atol=COEF_ATOL,
            err_msg=f"Coefficient '{py_name}'",
        )


def _compare_std_errors(result, r_results):
    """Compare standard errors."""
    r_ses = r_results["std_errors"]
    for name, r_val in r_ses.items():
        py_name = name.replace("(Intercept)", "Intercept")
        np.testing.assert_allclose(
            result.bse[py_name], r_val, rtol=SE_RTOL,
            err_msg=f"Std error '{py_name}'",
        )


def _compare_sigma(result, r_results):
    """Compare residual standard deviation (sigma)."""
    r_sigma = r_results["sigma"]
    py_sigma = np.sqrt(result.sigma2)
    np.testing.assert_allclose(
        py_sigma, r_sigma, rtol=SIGMA_RTOL,
        err_msg="Sigma (residual std dev)",
    )


def _compare_loglik(result, r_results):
    """Compare log-likelihood."""
    np.testing.assert_allclose(
        result.loglik, r_results["loglik"], atol=LOGLIK_ATOL,
        err_msg="Log-likelihood",
    )


def _compare_aic_bic(result, r_results):
    """Compare AIC and BIC."""
    np.testing.assert_allclose(
        result.aic, r_results["aic"], atol=AIC_ATOL,
        err_msg="AIC",
    )
    np.testing.assert_allclose(
        result.bic, r_results["bic"], atol=BIC_ATOL,
        err_msg="BIC",
    )


def _compare_correlation_params(result, r_results):
    """Compare correlation structure parameters."""
    r_cor = r_results.get("correlation_params")
    if r_cor is None:
        return
    # R returns dict like {"Phi": 0.48} or {"Rho": 0.30}
    r_values = list(r_cor.values())
    py_values = result.correlation_params
    assert py_values is not None, "Python result has no correlation params"
    np.testing.assert_allclose(
        np.asarray(py_values), np.asarray(r_values), atol=COR_PARAM_ATOL,
        err_msg="Correlation parameters",
    )


def _compare_variance_params(result, r_results):
    """Compare variance structure parameters.

    R reports variance ratios on natural scale (delta).
    Python stores log(delta).  We transform Python's to natural scale.
    """
    r_var = r_results.get("variance_params")
    if r_var is None:
        return
    r_values = np.asarray(list(r_var.values()))
    py_values = result.variance_params
    assert py_values is not None, "Python result has no variance params"
    py_natural = np.exp(np.asarray(py_values))
    np.testing.assert_allclose(
        py_natural, r_values, rtol=VAR_PARAM_RTOL,
        err_msg="Variance parameters (natural scale)",
    )


def _compare_all(result, r_results):
    """Run all comparisons for a scenario."""
    _compare_coefficients(result, r_results)
    _compare_std_errors(result, r_results)
    _compare_sigma(result, r_results)
    _compare_loglik(result, r_results)
    _compare_aic_bic(result, r_results)
    _compare_correlation_params(result, r_results)
    _compare_variance_params(result, r_results)


# ===========================================================================
# Test Classes
# ===========================================================================

class TestOLSvsR:
    """Scenario 1: OLS baseline (no correlation, no variance)."""

    def test_ols_baseline(self, r_reference):
        sc = r_reference["ols_baseline"]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])


class TestAR1vsR:
    """Scenarios 2-5: AR(1) with various phi and methods."""

    @pytest.mark.parametrize("scenario_name", [
        "ar1_moderate_reml",
        "ar1_moderate_ml",
        "ar1_high_reml",
        "ar1_negative_reml",
    ])
    def test_ar1(self, r_reference, scenario_name):
        sc = r_reference[scenario_name]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])


class TestCompSymmvsR:
    """Scenarios 6-7: Compound symmetry."""

    @pytest.mark.parametrize("scenario_name", [
        "compsymm_reml",
        "compsymm_ml",
    ])
    def test_compsymm(self, r_reference, scenario_name):
        sc = r_reference[scenario_name]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])


class TestVarIdentvsR:
    """Scenario 8: VarIdent with 2 groups."""

    def test_varident(self, r_reference):
        sc = r_reference["varident_reml"]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])


class TestCombinedvsR:
    """Scenario 9: AR(1) + VarIdent combined."""

    def test_ar1_varident(self, r_reference):
        sc = r_reference["ar1_varident_reml"]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])


class TestMultiplePredictorsvsR:
    """Scenario 10: Multiple predictors with AR(1)."""

    def test_multi_predictor(self, r_reference):
        sc = r_reference["multi_predictor_ar1"]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])


class TestUnbalancedvsR:
    """Scenario 11: Unbalanced panels with AR(1)."""

    def test_unbalanced(self, r_reference):
        sc = r_reference["unbalanced_ar1"]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])


class TestInterceptOnlyvsR:
    """Scenario 12: Intercept-only model with CompSymm."""

    def test_intercept_only_cs(self, r_reference):
        sc = r_reference["intercept_only_cs"]
        result = _fit_scenario(sc)
        _compare_all(result, sc["results"])
