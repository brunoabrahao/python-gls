"""Shared test fixtures for python_gls tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_data():
    """Simple dataset with known coefficients for basic testing."""
    np.random.seed(42)
    N = 200
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    y = 3.0 + 1.5 * x1 - 2.0 * x2 + np.random.randn(N) * 0.5
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


@pytest.fixture
def panel_data():
    """Panel dataset with subjects and time periods."""
    np.random.seed(123)
    n_subjects = 40
    n_times = 5
    N = n_subjects * n_times

    subjects = np.repeat(np.arange(n_subjects), n_times)
    times = np.tile(np.arange(n_times), n_subjects)
    x = np.random.randn(N)
    treatment = np.random.choice(["control", "treated"], n_subjects)[subjects]

    # AR(1) correlated errors within subject
    phi_true = 0.6
    errors = np.zeros(N)
    for s in range(n_subjects):
        idx = slice(s * n_times, (s + 1) * n_times)
        e = np.random.randn(n_times)
        for t in range(1, n_times):
            e[t] = phi_true * e[t - 1] + np.sqrt(1 - phi_true**2) * e[t]
        errors[idx] = e * 0.5

    y = 2.0 + 1.0 * x + errors

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "subject": subjects,
            "time": times,
            "treatment": treatment,
        }
    )


@pytest.fixture
def hetero_data():
    """Dataset with heteroscedastic errors by group."""
    np.random.seed(456)
    n_subjects = 30
    n_times = 4
    N = n_subjects * n_times

    subjects = np.repeat(np.arange(n_subjects), n_times)
    group = np.where(subjects < 15, "A", "B")
    x = np.random.randn(N)

    # Different variance by group
    sigma = np.where(group == "A", 0.5, 1.5)
    y = 1.0 + 2.0 * x + np.random.randn(N) * sigma

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "subject": subjects,
            "group": group,
        }
    )
